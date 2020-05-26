#####----------------项目说明-------------------########
#####版本0.1
####创建日期：2020-4-22
###功能：可以通过摄像头来建立自己的数据库进行手势分类识别(静态)
此版本是最初版本，刚刚做到简单的验证，
###-----------------使用说明-----------------####
##1.首先运行Create_collections.py文件生成自己的数据库，注意，此文件需要运行两次，分别获得数据库建立样本和测试样本
##2.运行data_better.py文件，对数据库样本图片进行扩容
##3.运行loadData文件，运行前最好先查看当前工作区内的文件夹是否创建和当前设定的路径是否一致，运行后将创建数据库文件
在数据库中，每个样本图片都将变成整数衡量特征储存在txt文件中，需要运行两次
##4.运行SVM(support vector machian),生成合适的分类器模型
##5.你已经可以打开ges_test.py文件来使用分类器了！ do it!
###------------------------------------------###
###----------一些参考----------------------##
##1.本次项目来源csdn论坛：https://blog.csdn.net/qq_41562704/article/details/88975569(一定要看！)
##2.关于傅里叶算子：a)https://www.cnblogs.com/edie0902/p/3658174.html (数学思想来源)
#b)https://github.com/alessandroferrari/elliptic-fourier-descriptors (椭圆傅里叶描述子的提取)
#c)https://github.com/timfeirg/Fourier-Descriptors (github大佬的提取代码)
##3. 关于SVM(support vector machian,支持向量机)的原理参考
#a)https://cuijiahua.com/blog/2017/11/ml_8_svm_1.html (原理来源)
#b)https://cuijiahua.com/blog/2017/11/ml_9_svm_2.html (sklearn的使用)
#c)https://blog.csdn.net/qysh123/article/details/80063447 (SVM参数的调整方法)
##4.SVM  bilibili视频参考
#a)https://www.bilibili.com/video/BV19t411a7j1?t=4218 (初步了解SVM，看完+消化大概2个小时)
#b)https://www.bilibili.com/video/BV1j4411H77t?t=683 (十几分钟初步认识科普视频，建议一定要看，如果你不想看上面那个74min的话)
####----74min那个视频代码下载：http://www.peixun.net/view/1281.html
###代码持续改进，后续还会结合实际添加应用模块，不断撸码中......