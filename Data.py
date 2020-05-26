#--* conding: utf-8 *--
#GUI封装 部分
#调用需要的库
from PySide2.QtWidgets import QApplication,QMessageBox,QWidget,QLabel
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile,QSize,QTimer
from PySide2.QtGui import QImage,QPixmap
from pygame import mixer
import os
import shutil
import cv2 as cv
import MyFuncs as mf
import time
import numpy as np

from sklearn.externals import joblib
from functools import reduce
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,f1_score,log_loss

import matplotlib.pyplot as plt


import seaborn as sn
#--********--

#--configure settings--#
Cap=0
files_Cannot_delte=['songs']
Files_Init=['data_images','test_images','data_features','test_features','model']
SavingImages=False
#######################
######################
#文件夹初始化，避免原数据的干扰
def Init_Files():
    global files_Cannot_delte
    rootdir='./Source'
    fileslist=os.listdir(rootdir)
    for file in fileslist:
        FilePath=os.path.join(rootdir,file)
        if os.path.isfile(FilePath):
            os.remove(FilePath)
        elif file in files_Cannot_delte:
            continue
        elif os.path.isdir(FilePath):
            shutil.rmtree(FilePath)
    for file in Files_Init:
        path='./Source/'+file+''
        folder=os.path.exists(path)
        if not folder:
            os.mkdir(path)
##########
class Stats(QWidget):
    def __init__(self):
        super().__init__()
        qfile_stats=QFile('./Data.ui')  #加载ui文件
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        #----configure settings
        self.DataImagesPath='./Source/data_images/'   #数据集图片保存路径
        self.TestImagesPath='./Source/test_images/'   #测试集图片保存路径
        self.CurrentPath=self.DataImagesPath          #当前的图片保存路径
        self.DataFeaturePath='./Source/data_features/'   #数据集特征路径
        self.TestFeaturePath='./Source/test_features/'   #测试集特征路径
        self.FilePath=self.DataFeaturePath  #当前的文件保存路径
        self.Cap=cv.VideoCapture(0)   #摄像头，0是笔记本自带，1是外接摄像头
        self.x0=50   #选取框的左上角横坐标
        self.y0=50   #选取框的左上角纵坐标
        self.width=300    #选取框的宽度
        self.height=400   #选取框的高度
        self.MaxKind=0    #输入的最大种类数
        self.CurrentKind=0   #当前的手势数
        self.MaxImages=0    #规定的保存的图片数
        self.CurrentNumOfImages=0  #当前已经保存的图片数
        self.CurrentPosName=''  #当前需要保存的手势名
        self.SavingImages=False    #自动保存图片开关
        self.CameraStatus=False    #摄像头状态：开 and 关
        self.TestControl=False     #是否测试： 开 and 关
        self.SavingSingleImage=False #是否打开单张保存开关
        self.CheckErrorRio=False     #检查错误率开关
        self.BinaryImage=''
        self.ConvasImage=''
        self.LockShow=False   #显示锁定
        self.KindFiles=[]   #手势名
        self.AllTxts=0
        #-------****------------
        self.ui=QUiLoader().load(qfile_stats)  #加载UI文件
        self.Dims=int(self.ui.DimsBox.currentText())
        self.TipsBox=QMessageBox()  #定义消息提示框
        #warning: 警告  critical: 错误报告   infomation，about: 消息提示   question: 确认继续
        self.ui.InputBox.setEnabled(False)
        self.ui.InputBoxButton.setEnabled(False)
        self.ui.InputEditButton.setEnabled(False)
        self.ui.InputEdit.setEnabled(False)
        self.ui.SaveButton.setEnabled(False)
        #----------*****----------DimsBox
        self.ui.chooseFunc.currentIndexChanged.connect(self.handleSelectionChange)
        self.ui.DimsBox.currentIndexChanged.connect(self.ChangeDims)
        self.ui.CameraButton.clicked.connect(self.OpenCamera)
        self.ui.InputBoxButton.clicked.connect(self.GetInputBox)
        self.ui.InputEditButton.clicked.connect(self.GetInputEdit)
        self.ui.SaveButton.clicked.connect(self.SaveImg)
        self.ui.SaveSingleButton.clicked.connect(self.SaveSingleImg)
        self.ui.ExitButton.clicked.connect(self.ExitSystem)
        self.ui.SVMTrainButton.clicked.connect(self.SVMTrain)
        self.ui.SVMTestButton.clicked.connect(self.SVMTest)
        self.ui.SVMErrorButton.clicked.connect(self.SVMError)
        self.ui.LoadButton.clicked.connect(self.LoadData)
        #----------提示信息----------------
        self.ui.InfoBrower.append('1.请选择样本或者测试集')
        self.ui.InfoBrower.append('2.打开摄像头')
        #-----****-------------
        self.timer = QTimer()  #计时器，循环执行VideoShowing
        self.timer.timeout.connect(self.VideoShowing)
        self.timer.start(1)
    #--------****-----------------
    #  复选框的信号函数,根据选择更改文件的读取路径和保存路径
    def handleSelectionChange(self):
        method=self.ui.chooseFunc.currentText()
        if method=='样本数据集':
            self.CurrentPath=self.DataImagesPath
            self.FilePath=self.DataFeaturePath
        elif method=="测试数据集":
            self.CurrentPath=self.TestImagesPath
            self.FilePath=self.TestFeaturePath
    #改变算子的计算维度
    def ChangeDims(self):
        self.Dims=int(self.ui.DimsBox.currentText())

    #摄像头的信号函数
    def OpenCamera(self):  #打开摄像头
        self.Cap=cv.VideoCapture(0)
        self.CameraStatus=True
        self.ui.chooseFunc.setEnabled(False)
        self.ui.InputBoxLabel.setText('请输入手势种数')
        self.ui.InputBox.setEnabled(True)
        self.ui.InputBoxButton.setEnabled(True)
        self.ui.CameraButton.setEnabled(False)

    #数字框输入后确定按钮后信号函数
    def GetInputBox(self):
        value=self.ui.InputBox.value()
        Text=self.ui.InputBoxLabel.text()
        if Text== '请输入手势种数':
            self.MaxKind=value
            self.ui.InfoBrower.append('手势总数:'+str(self.MaxKind))
            self.ui.InfoBrower.append('请输入手势名,创建文件夹')
        elif Text == '请输入图片最大数':
            self.MaxImages=value
            self.ui.InfoBrower.append('规定'+self.CurrentPosName+"的图片数量："+str(self.MaxImages))
            self.ui.InfoBrower.append('图片保存路径:'+self.CurrentPath+self.CurrentPosName+'/')
        self.ui.InputBox.setEnabled(False)
        self.ui.InputEditLabel.setText('请输入手势名')
        self.ui.InputEdit.setEnabled(True)
        self.ui.InputEditButton.setEnabled(True)
        self.ui.InputBoxButton.setEnabled(False)

    #文本输入框的确认按钮后的信号函数
    def GetInputEdit(self):
        self.CurrentPosName=self.ui.InputEdit.text()
        folder_path=self.CurrentPath+'/'+self.CurrentPosName
        #print(folder_path)
        folder=os.path.exists(folder_path)
        if not folder:
            os.mkdir(folder_path)
        self.TipsBox.information(
            self.ui,
            "正确",
            "文件夹创建成功"
        )
        self.ui.InputBox.setEnabled(True)
        self.ui.InputBoxLabel.setText('请输入图片最大数')
        self.ui.InputBoxButton.setEnabled(True)
        self.ui.InputEdit.setEnabled(False)
        self.ui.InputEditButton.setEnabled(False)
        self.ui.SaveButton.setEnabled(True)
    #打开自动保存图片开关，连续地自动保存函数
    def SaveImg(self):
        self.SavingImages=True
        self.CurrentKind=self.CurrentKind+1
        self.ui.InfoBrower.append('总手势数:'+str(self.MaxKind))
        self.ui.InfoBrower.append('当前第:'+str(self.CurrentKind)+"种")
        self.ui.InfoBrower.append('此手势名:'+self.CurrentPosName)
         
        self.ui.SaveButton.setEnabled(False)
    #单张图片保存，可以预览效果再保存
    def SaveSingleImg(self):
        self.SavingSingleImage=True
        self.LockShow=True
        #self.ui.SaveSingleButton.setEnabled(True)

    def ExitSystem(self):
        exit(0)

    #训练支持向量机模型
    def SVMTrain(self):
        start=time.time()
        self.ui.InfoBrower.append('开始训练向量机')
        self.ui.InfoBrower.append('特征维度%d'%(self.Dims))
        svc=SVC()
        Files=os.listdir('./Source/data_features/')  #列出所有的文件
        NumOfFiles=len(Files) #文件数
        alltxts=0
        for file in Files:  #计算所有的txt文件数
            alltxts+=len(os.listdir('./Source/data_features/'+file+'/'))
        print('alltxts=%d'%(alltxts))
        TrainMat=np.zeros((alltxts,self.Dims-1))  #初始化数据矩阵
        SVMLabels=[] #标签矩阵
        parameters = {'kernel':('linear', 'rbf'),
	              'C':[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
	              'gamma':[0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}#预设置一些参数值
        ###导入数据###
        Cnt=0
        for file in Files:
            txts=os.listdir('./Source/data_features/'+file+'/')
            for txt in txts:
                ClassNumStr=int(txt.split('_')[0])  #提取标签
                SVMLabels.append(ClassNumStr) #加载标签
                #将数据集加载为矩阵
                returnVec = np.zeros((1, self.Dims-1))
                fr = open('./Source/data_features/'+file+'/'+txt) #打开文件
                lineStr = fr.readline()  #读取一整行
                lineStr = lineStr.split(' ') #除去空格
                for i in range(self.Dims-1):
                    returnVec[0, i] = int(lineStr[i])  #数据转换为整数
                TrainMat[Cnt,:]=returnVec  #加载矩阵
                Cnt+=1
        #-----PCA---------
        #n_comp=50
        #pca=PCA(n_comp)
        #pca.fit(TrainMat)
        #TrainMat=pca.transform(TrainMat)
        #-----------------
        print('数据加载完毕')
        self.ui.InfoBrower.append('数据加载完毕,开始训练')
        clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)  # 网格搜索法，设置5-折交叉验证
        clf.fit(TrainMat, SVMLabels)
        #self.ui.InfoBrower.append('打印最好的结果')
        print(clf.return_train_score) #打印得分
        print(clf.best_params_)  # 打印出最好的结果
        best_model = clf.best_estimator_
        end=time.time()
        period=end-start
        self.ui.InfoBrower.append('正在保存模型...')
        save_path = "./Source/model/" + "svm_efd_" + "train_model.m"
        joblib.dump(best_model, save_path)  # 保存最好的模型
        self.ui.InfoBrower.append('训练用时：%d 秒'%(period))
        self.ui.InfoBrower.append('模型训练成功！')

    #打开测试开关
    def SVMTest(self):
        self.TestControl=not self.TestControl

    #错误率测试函数，按下错误率按钮的信号函数
    def SVMError(self):
        FilePath='./Source/data_features/'  #路径
        TestFiles=os.listdir(FilePath)  #读取子文件夹
        AllKinds=len(TestFiles)
        Confution_matrixs=np.zeros((AllKinds,AllKinds))
        index_x,index_y=0,0
        AllSum=0   #所有的文件
        AllError=0
        clf=joblib.load('./Source/model/' + "svm_efd_" + "train_model.m")
        CNT=0
        TrueLabels=[]
        PreLabels=[]
        for TestFile in TestFiles:
            Files=os.listdir(FilePath+TestFile+'/')  #每种手势的特征文件数，txt文件数
            self.KindFiles=os.listdir("./Source/data_images/")
            print()
            NumOfFiles=len(Files)
            ErrorCount=0  #无误数据
            ErrorFilesName=[]  #错误的文件名
            for i in range(NumOfFiles):
                AllSum+=1
                FileName='./Source/data_features/'+TestFile+'/'+Files[i]  #文件路径
                TrueLabel=int(Files[i].split('_')[0])  #真实标签
                TrueLabels.append(TrueLabel)
                Vector=np.zeros((1,self.Dims-1))  #事先定义特征向量的大小
                fr=open(FileName)  #打开文件
                LineStr=fr.readline()  #读取一行
                LineStr=LineStr.split(' ')  #滤除空格
                fr.close()
                for j in range(self.Dims-1):  #数据变换
                    Vector[0,j]=int(LineStr[j])
                    #print(Vector[0,j])   #调试打开
                    #print(Vector)
                TestLabel=clf.predict(Vector)  #测试标签
                PreLabels.append(TestLabel)
                if TrueLabel!=TestLabel:
                    ErrorCount+=1  #错误数
                    ErrorFilesName.append(Files[i])
                    AllError+=1  #数据集的错误总数
                Confution_matrixs[index_x,TestLabel[0]-1]+=1
            index_x+=1
            #ErrorRio=str(ErrorCount/NumOfFiles)+'%'  #错误率
            self.ui.InfoBrower.append('手势名:%s,总测试数:%d' %(self.KindFiles[CNT],NumOfFiles))
            self.ui.InfoBrower.append('错误率:%f%%,错误个数:%d' %(100*ErrorCount/NumOfFiles,ErrorCount))
            print('错误文件=',ErrorFilesName)
            CNT+=1
        self.ui.InfoBrower.append('错误率计算完毕！')
        CM=confusion_matrix(TrueLabels,PreLabels,labels=[1,2,3,4,5,6,7,8,9,10])
        row_sum=sum(CM)
        #-----显示混淆矩阵------
        print('--------混淆矩阵---------')
        print(CM)
        CM_T=CM.T
        column_sum=sum(CM_T)
        print('--------矩阵列求和值---------')
        print(row_sum)
        print('---------矩阵行求和值---------')
        print(column_sum)
        print('--------准确度-------\n',accuracy_score(TrueLabels,PreLabels))
        print('--------精确度-------\n',precision_score(TrueLabels,PreLabels,average=None))
        print('--------召回率--------\n',recall_score(TrueLabels,PreLabels,average=None))
        print('--------F1-Measure----\n',f1_score(TrueLabels,PreLabels,average=None))
        #绘制热力图
        sn.set()  #初始化热力图
        f,ax=plt.subplots()
        sn.heatmap(CM,annot=True,fmt='.20g',ax=ax,cmap="Blues")  #蓝色，保证不以科学计数法的形式出现
        ax.set_title('ConfusionMatrix,Dimension=%d'%(self.Dims))
        ax.set_xlabel('Predict')
        ax.set_ylabel('True')
        plt.show()

    def VideoShowing(self):
        if self.CameraStatus==True:
            _,frame=self.Cap.read()  #读取摄像头的一帧
            #frame=cv.cvtColor(frame,cv.COLOR_RGB2BGR)
            frame=cv.flip(frame,1)  #翻转
            #cv.rectangle(frame,(self.x0,self.y0),(self.x0+self.width,self.y0+self.height),(0,0,255),1)  #矩形加框，暂时不用了
            roi=frame[self.y0:self.y0+self.height,self.x0:self.x0+self.width]  #ROI的提取
            binary,convas=mf.DealImage(roi)     #二值化和轮廓
            if self.LockShow==False:  #非锁定更新
                self.BinaryImage=binary
                self.ConvasImage=convas
            self.ui.ROILabel.setText('原图')
            roi=cv.cvtColor(roi,cv.COLOR_RGB2BGR)
            ROILabelShowing=QImage(roi.data,roi.shape[1],roi.shape[0],QImage.Format_RGB888)
            self.ui.roi_show_label.setPixmap(QPixmap.fromImage(ROILabelShowing))
            self.ui.ConvasLabel.setText('轮廓图')
            ConvaShowing=QImage(self.ConvasImage.data,self.ConvasImage.shape[1],self.ConvasImage.shape[0],QImage.Format_Grayscale8)
            self.ui.convas_show_label.setPixmap(QPixmap.fromImage(ConvaShowing))
            #----连续保存----
            if self.SavingImages==True:
                if self.CurrentNumOfImages < self.MaxImages:
                    self.CurrentNumOfImages=self.CurrentNumOfImages+1
                    cv.imwrite(self.CurrentPath+'/'+self.CurrentPosName+'/'+str(self.CurrentNumOfImages)+'.jpg',binary)
                    self.ui.InfoBrower.append(str(self.CurrentNumOfImages)+'.jpg保存成功')
                    time.sleep(1)
                else:
                    self.SavingImages=False
                    self.MaxImages=0
                    self.CurrentNumOfImages=0
                    self.ui.InfoBrower.append('当前种类图片保存完成！')
                    if self.MaxKind==self.CurrentKind:
                        self.ui.InfoBrower.append('所有的图片保存完成！')
            #----单张保存-----
            if self.SavingSingleImage==True:
                cv.imshow('保存图片',self.ConvasImage)
                self.ui.InfoBrower.append('准备保存'+str(self.CurrentNumOfImages+1)+'.jpg...')
                choice=self.TipsBox.question(
                    self.ui,
                    '保存图片',
                    '你是否要保存此图'
                )
                if choice == QMessageBox.Yes:
                    self.CurrentNumOfImages+=1
                    cv.imwrite(self.CurrentPath+'/'+self.CurrentPosName+'/'+str(self.CurrentNumOfImages)+'.jpg',binary)
                    self.ui.InfoBrower.append(str(self.CurrentNumOfImages)+'.jpg保存成功')
                    if self.CurrentNumOfImages>=self.MaxImages:
                        self.ui.SaveSingleButton.setEnabled(False)
                elif choice == QMessageBox.No:
                    self.ui.InfoBrower.append('放弃保存')
                self.SavingSingleImage=False
                self.LockShow=False
                cv.destroyAllWindows()
            #测试
            if self.TestControl==True:
                Names=os.listdir('./Source/data_images/')
                Cnt=0
                LabelName={}
                for Name in Names:
                    Cnt+=1
                    LabelName[str(Cnt)]=Name
                ffdp=mf.FourierDesciptor(binary,self.Dims)
                descriptor_in_use = abs(ffdp[2])
                temp = descriptor_in_use[1]
                test_num=np.zeros((1,self.Dims-1),dtype=int)
                for i in range(1,len(descriptor_in_use)):
                    test=int(100*descriptor_in_use[i]/temp)
                    test_num[0][i-1]=test
                #print('test_num=',test_num)   #统计出该帧的中手势的傅里叶算子,测试使用
                #print("now,test_num=",test_num)
                clf = joblib.load('./Source/model/' + "svm_efd_" + "train_model.m")
                label=clf.predict(test_num)
                self.ui.InfoBrower.append('分类结果:'+LabelName[str(label[0])])
        else:
            black=np.ones((350,431),dtype=np.uint8)
            Roi_showing=QImage(black.data,black.shape[1],black.shape[0],QImage.Format_Grayscale8)
            self.ui.roi_show_label.setPixmap(QPixmap.fromImage(Roi_showing))
            Convas_showing=QImage(black.data,black.shape[1],black.shape[0],QImage.Format_Grayscale8)
            self.ui.convas_show_label.setPixmap(QPixmap.fromImage(Roi_showing))
######################

    def LoadData(self):
        #self.ui.DimsBox.setEnabled(False)
        self.ui.InfoBrower.append('数据加载中...')
        Path='./Source/data_images/'
        self.KindFiles=os.listdir(Path)  #读取所有的手势名
        NumofKind=len(self.KindFiles)  #总数
        Cnt=0
        AllLoadNum=0
        for Cnt in range(NumofKind):
            images=os.listdir(Path+self.KindFiles[Cnt]+'/')
            ImagesCount=0
            FolderName='./Source/data_features/'+str(Cnt+1)
            if os.path.exists(FolderName)==False:
                os.mkdir(FolderName)
            for image in images:
                AllLoadNum+=1   #加载数+1
                ImagesCount+=1
                ImageDealing=cv.imread(Path+self.KindFiles[Cnt]+'/'+image,cv.IMREAD_UNCHANGED)
                ffdp=mf.FourierDesciptor(ImageDealing,self.Dims)
                #print('ffdp=',ffdp)
                descriptor_in_use = abs(ffdp[2])
                #print('descriptor_in_use=',descriptor_in_use)
                FileName= './Source/data_features/'+str(Cnt+1)+'/'+str(Cnt+1)+'_'+str(ImagesCount)+'.txt'
                with open(FileName,'w',encoding='utf-8') as f:
                    temp = descriptor_in_use[1]
                    #print('temp=',temp)
                    for k in range(1, len(descriptor_in_use)):
                        x_record = int(100 * descriptor_in_use[k] / temp)
                        #print('x_record',x_record)
                        f.write(str(x_record))
                        f.write(' ')
                    f.write('\n')
                    self.ui.InfoBrower.append(str(Cnt+1)+'_'+str(ImagesCount)+'.txt'+' 加载完毕')
        self.ui.InfoBrower.append('数据加载完毕，总加载%s项!'%(AllLoadNum))
        self.AllTxts=AllLoadNum
if __name__=="__main__":
    Init_Files()
    app=QApplication([])
    Stats=Stats()
    Stats.ui.show()
    app.exec_()