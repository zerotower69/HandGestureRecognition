#--* conding: utf-8 *--
#GUI封装 部分
#调用需要的库
from PySide2.QtWidgets import QApplication,QMessageBox,QWidget,QLabel
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile,QSize,QTimer
from PySide2.QtGui import QImage,QPixmap
from pygame import mixer

from sklearn.externals import joblib
from functools import reduce
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os
import cv2 as cv
import numpy as np
import MyFuncs as mf

################
class Stats(QWidget):
    def __init__(self):
        qfile_stats=QFile('./play.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()

        #------*Configure*------
        self.DataImagesFolder='data_images'
        self.DataFeatureFolder='data_feature'
        self.TestImagesFolder='test_images'
        self.TestFeatureFolder='test_feature'

        self.CurrentSong=0
        self.Songs=os.listdir('./Source/songs')
        self.PlayMusic=False
        self.MusicVolumn=0.2
        self.NumOfSongs=len(self.Songs)
        mixer.init()
        mixer.music.load('./Source/songs/'+self.Songs[self.CurrentSong])
        mixer.music.play()

        self.Controller=0
        self.x0=50
        self.y0=50
        self.width=300
        self.height=400

        self.ui=QUiLoader().load(qfile_stats)
        #self.ui.label_3.setFixedSize(self.video_size)

        self.ui.play_button.clicked.connect(self.play_button)
        self.ui.last_song_button.clicked.connect(self.last_song_button)
        self.ui.next_song_button.clicked.connect(self.next_song_button)
        self.capture = cv.VideoCapture(0)
        #self.capture.set(cv.CAP_PROP_FRAME_WIDTH , self.video_size.width())
        #self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.video_size.height())
 
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

        self.timer2=QTimer()
        self.timer2.timeout.connect(self.ControlMusic)
        self.timer2.start(7000)

    def play_button(self):
        self.PlayMusic= not self.PlayMusic
        mixer.init()
        mixer.music.load('./Source/songs/'+self.Songs[self.CurrentSong])
        if self.PlayMusic==True:
            mixer.music.play()
        else:
            mixer.music.stop()
    def next_song_button(self):
        if self.CurrentSong==self.NumOfSongs-1:
            self.CurrentSong=0
        else:
            self.CurrentSong=self.CurrentSong+1
        mixer.init()
        mixer.music.stop()
        mixer.music.load('./Source/songs/'+self.Songs[self.CurrentSong])
        mixer.music.play()
    def last_song_button(self):
        if self.CurrentSong==0:
            self.CurrentSong=self.NumOfSongs-1
        else:
            self.CurrentSong=self.CurrentSong-1
        mixer.init()
        mixer.music.stop()
        mixer.music.load('./Source/songs/'+self.Songs[self.CurrentSong])
        mixer.music.play()
    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        frame = cv.flip(frame, 1)
        roi=frame[self.y0:self.y0+self.height,self.x0:self.x0+self.width]
        binary,convas=mf.DealImage(roi)
        #cv.imshow('binary',binary)
        Canfind,ret,fourier_descriptor_in_use=mf.FourierDesciptor(binary,32)
        if Canfind==True:
            fd_in_use=abs(fourier_descriptor_in_use)
            #print('abs(descriptor)=',fd_in_use)
            temp=fd_in_use[1]
            #print("temp=",temp)
            test_num=np.zeros((1,31),dtype=int)
            #print(test_num)
            #test_num.
            for i in range(1,len(fd_in_use)):
                test=int(100*fd_in_use[i]/temp)
                test_num[0][i-1]=test
            #print('test_num=',test_num)   #统计出该帧的中手势的傅里叶算子,测试使用
            #print("now,test_num=",test_num)
            ModelPath='./model/32/'
            clf = joblib.load(ModelPath + "svm_efd_" + "train_model.m")
            label=clf.predict(test_num)
            print('分类结果：',label[0])
            self.Controller=label[0]
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       QImage.Format_RGB888)
        self.ui.label_3.setPixmap(QPixmap.fromImage(image))

    def ControlMusic(self):
        if self.Controller==2:
            mixer.music.pause()
        elif self.Controller==3:
            mixer.music.unpause()
        #elif self.Controller==3:
            #last_song_button()
        print('执行')

if __name__=="__main__":
    app=QApplication([])
    Stats=Stats()
    Stats.ui.show()
    app.exec_()