import cv2 as cv
import numpy as np
from os import listdir
#肤色提取
def skinMask(img):
    skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
    cv.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0, 360, (255,255,255), -1) #绘制椭圆弧线
    YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB) #转换至YCrCb空间
    (y,Cr,Cb) = cv.split(YCrCb) #拆分出Y,Cr,Cb值
    skin = np.zeros(Cr.shape, dtype = np.uint8) #掩膜
    (x,y) = Cr.shape
    for i in range(0, x):
        for j in range(0, y):
            if skinCrCbHist [Cr[i][j], Cb[i][j]] > 0: #若不在椭圆区间中
                skin[i][j] = 255
    res = cv.bitwise_and(img,img, mask = skin)
    return res

#帧处理
def DealImage(Image):
    #噪声处理部分
    blur=cv.medianBlur(Image,3)
    blur=cv.GaussianBlur(blur,(3,3),1)
    #cv.imshow('Noisy_dealing',blur)
    #肤色提取
    skin=skinMask(blur)
    #cv.imshow('skin',skin)
    #形态学处理,开操作
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(10,10))
    open=cv.morphologyEx(skin,cv.MORPH_OPEN,kernel)
    erode=cv.morphologyEx(open,cv.MORPH_ERODE,cv.getStructuringElement(cv.MORPH_RECT,(4,4)))
    #cv.imshow('OPEN',erode)

    #图像灰度化二值化
    gray=cv.cvtColor(erode,cv.COLOR_BGR2GRAY)
    kernel3 = np.array([
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 8, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1]]) / 8.0
    great3 = cv.filter2D(gray, -1, kernel3)
    #cv.imshow('锐化3', great3) #锐化处理后，仅仅作为测试使用
    _, binary = cv.threshold(great3, 0, 255, cv.THRESH_BINARY)
    #cv.imshow('二值化', binary)

    #绘制轮廓
    ret_np = np.ones(binary.shape, np.uint8)  # 创建黑色幕布
    #找轮廓
    #try...except...的原因：万一检测不到边界，也不会因为保存抛出xx异常
    try:
        contours=cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        contour = contours[0]
        contour = sorted(contour, key=cv.contourArea, reverse=True)
        contour_array = contour[0][:, 0, :]  # 注意这里只保留区域面积最大的轮廓点坐标
        #这里其实就是排除干扰区域，只留下手的轮廓，但是这样做万一其他肤色区域比手大就会出错
        #因此，这部分还有待优化
        ret = cv.drawContours(ret_np, contour[0], -1, (255, 255, 255), 1)  # 绘制白色轮廓
    except:
        ret=ret_np
    return binary,ret
#计算傅里叶算子
def FourierDesciptor(res,dims):
    descirptor_in_use=[]
    ret_np=np.ones(res.shape,dtype=np.uint8)
    Canfind=False
    try:
        contours = cv.findContours(res, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour = contours[0]
        contour = sorted(contour, key=cv.contourArea, reverse=True)
        contour_array = contour[0][:, 0, :]  # 注意这里只保留区域面积最大的轮廓点坐标
        ret = cv.drawContours(ret_np, contour[0], -1, (255, 255, 255), 1)  # 绘制白色轮廓
    except:
        ret=ret_np
        return Canfind,ret,descirptor_in_use
    Canfind=True
    contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contours_complex.real = contour_array[:,0]#横坐标作为实数部分
    contours_complex.imag = contour_array[:,1]#纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)#进行傅里叶变换
    #fourier_result = np.fft.fftshift(fourier_result)
    descirptor_in_use = truncate_descriptor(fourier_result,dims)#截短傅里叶描述子
    return Canfind,ret,descirptor_in_use


def truncate_descriptor(fourier_result,dims):
    MIN_DESCRIPTOR = dims
    descriptors_in_use = np.fft.fftshift(fourier_result)
    # 取中间的MIN_DESCRIPTOR项描述子
    center_index = int(len(descriptors_in_use) / 2)
    low, high = center_index - int(MIN_DESCRIPTOR / 2), center_index + int(MIN_DESCRIPTOR / 2)
    descriptors_in_use = descriptors_in_use[low:high]

    descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
    return descriptors_in_use

#Input:ImagesPath,FilePath
def LoadData(ImagesPath,FilePath):
    files=listdir(ImagesPath)
    FilesNum=0
    for file in files:
        FilesNum+=1
        images=listdir(ImagesPath+file+'/')
        ImagesCount=0
        for image in images:
            ImagesCount+=1
            ImageDealing=cv.imread(ImagesPath+file+'/'+image,cv.IMREAD_UNCHANGED)
            ffdp=FourierDesciptor(ImageDealing)
            descirptor_in_use = abs(ffdp[1])
            FileName= FilePath +str(FilesNum)+'_'+str(ImagesCount) + '.txt'
            with open(FileName,'w',encoding='utf-8') as f:
                temp = descirptor_in_use[1]
                for k in range(1, len(descirptor_in_use)):
                    x_record = int(100 * descirptor_in_use[k] / temp)
                    f.write(str(x_record))
                    f.write(' ')
                f.write('\n')
