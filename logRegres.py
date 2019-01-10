import math
import numpy as np
import logRegres
def loadDataSet():
    dataMat=[];labalMat=[]
    fr=open('testSet.txt','rb')
    for line in fr.readlines():
        lineArr=line.strip().split()
        #strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) #读取X1,X2,并将X0设为1
        labalMat.append(int(lineArr[2]))
    return dataMat,labalMat
def sigmoid(inX):
    h=len(inX)
    returnInx=np.ones((h,1))
    for i in range(h):
        returnInx[i][0]= 1.0/(1+math.exp(-inX[i][0]))
    return returnInx
def sigmoid1(inX):
    if inX >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 / (1 + math.exp(-inX))
    else:
        return math.exp(inX) / (1 + math.exp(inX))

   # return np.longfloat(1.0/(1+math.exp(-inX)))

def gradAscent(dataMatIn,classLabels):
    #dataMatIn是一个2维numpy数组，每列代表不同的特征，每行代表每个训练样本
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose() #transpose转置 行向量转置为列向量
    m,n=np.shape(dataMatrix)
    alpha=0.001#向目标移动的步长
    maxCycles=500#迭代次数
    weights=np.ones((n,1)) #nx1 对于每一个特征值有一个weight
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights) #M*N N*1 --> M*1
        error=(labelMat-h) #error  M*1
        weights=weights+alpha*dataMatrix.transpose()*error # N*M * M*1 --> N*1
    return  weights
def plotBestFit(wei):   #调用梯度上升法时要调用weights.getA() 随机梯度上升法时直接调用weights即可
    import matplotlib.pyplot as plt
    weights=wei
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    n=np.shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n): #画点
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)           #创建一个步进为0.1的从-3.0至3.0的等差数列 画线
    y=(-weights[0]-weights[1]*x)/weights[2]#0是两个分类的分界处，我们设定0=w0x0+w1x1+w2x2 x0设为1 可求y值
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()
def stocGradAscent0(dataMatrix,classLabels):
    m,n=np.shape(dataMatrix)
    alpha=0.01
    weights=np.ones(n)
    for i in range(m):
        h=sigmoid1(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return  weights
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    """
        改进的随机梯度上升
        1、
            alpha 会随着迭代次数不断减小
        在降低alpha的函数中，每次减少1/(i+j)，
        其中j是迭代次数，i时样本点的下标
        当j远小于max（i）时 就不是严格下降的
        2、
            选取随机样本来更新回归系数
        减少周期性波动
    """
    import random
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m): # m为样品数
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex))) #随机在0到样品最大值之间取一个数
            h=sigmoid1(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
def classsifyVector(inX,weights):
    prob=sigmoid1(sum(inX*weights))
    if prob>0.5:return 1.0
    else: return 0.0
def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[];trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    errorcount=0.0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classsifyVector(lineArr,trainWeights))!=int(currLine[21]):
            errorcount+=1.0
    errorRate=(float(errorcount)/numTestVec)
    print('the error rate of this test is:%f'%errorRate)
    return  errorRate
def multiTest():
    #调用10次colicTest取平均
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iterations the average error rate is:%f"%(numTests,errorSum/float(numTests)))
