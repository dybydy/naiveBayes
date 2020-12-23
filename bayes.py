##############贝叶斯极大似然估计代码
###第一步：
###首先定义数据存储结构,节点代表X各个特征与y的条件概率构成的二维数组，链表代表X所有的特征维度
###用链表加二维数组节点的形式解决了x各个特征与y的条件概率构成的二维数组维度不一致的问题
class Node(object):
    def __init__(self,a=0,axis=0,cx=0,nums=0,next=None):
        self.a=a;#用来存储二维数组
        self.axis=axis;#用来存储x的特征向量是第几列
        self.cx=cx;#用来存储该特征向量集合中的不同的值
        self.nums=nums;#用来存储该特征向量集合中不同的值的总个数
        self.next=next;#用来连接下一个节点
    def __str(self):
        return("用来存储各个节点概率");
    __repr__="__str__";
    
###用来指示链表
class Linklist(object):
    def __init__(self):
        self.root=Node();
        self.tailnode=None;

###第二步：
###将数据从Excel导入python中，并对数据进行预处理
import numpy as np;
import pandas as pd;
df=pd.read_excel("D:\pythondata\\bayes.xls");#导入原始数据
df=pd.DataFrame(df);
x=df.values[:,:-1];#x代表X的特征向量
y=df.values[:,-1];#y代表对应的类别
n=x.shape[0];#代表样本的行数
m=x.shape[1];#代表样本的列数

###第三步：
###计算先验概率
c=[];#用来存储y集合中的不同的值
ynum=0;#用来存储y集合中不同值的个数
for i in range(n):
    if y[i] not in c:
        c.append(y[i]);
        ynum+=1;
c.sort();#对y类别的集合中不同的元素进行排序
t=[];#用来存储y=Ck的个数，即c[i]的值或者类别
pyck=[];#用来存储y=Ck的概率，即c[i]的值或类别的概率
for i in range(ynum):
    temp=0;#用来统计c[i]中元素的个数
    for j in range(n):
        if (y[j]==c[i]):
            temp+=1;
    t.append(temp);
    pyck.append((temp/n));
    
###第四步：
###计算各个条件概率的极大似然估计
root=Linklist();
for j in range(m):#依次构建各个节点
    #计算X的第j列特征向量不同的值的集合以及集合中各个元素对应的个数
    cx=[];#存储X的第j列的集合中不同的值
    xnums=0;#标记x的第j列的集合中不同元素对应的个数
    for i in range(n):
        if (x[i][j] not in cx):
            cx.append(x[i][j]);
            xnums+=1;
    cx.sort();
    #计算x的第j列的集合与y的集合形成的条件概率，最终结果用二维数组形式保存在各个节点中
    pxy=np.zeros((ynum,xnums));#构建二维数组，用来存储x的第j列的集合与y的集合中各个元素形成的条件概率
    for k in range(ynum):
        for i in range(xnums):
            nums=0;#用来计算I(Xi(j)=ajl,yi=Ck)
            for ii in range(n):
                if ((x[ii][j]==cx[i]) and (y[ii]==c[k])):
                    nums+=1;
            pxy[k][i]=nums/t[k];
    #print("本次循环中建立的二维数组是{}\n".format(pxy));
    node=Node();#用来构建当前节点
    node.a=pxy;
    node.cx=cx;
    node.nums=nums;
    if (root.tailnode==None):
        root.next=node;
        root.tailnode=node;
    else:
        root.tailnode.next=node;
        root.tailnode=node;
    root.tailnode.axis=j;

###第五步：
###对于给定的实例x,计算对应的y的各个类别的概率
#定义二分函数，方便快速查找元素的位置
def twobreak(a,x):
    low=0;
    high=len(a)-1;
    while (low<high):
        mid=(low+high)//2;
        if (a[mid]==x):
            return mid;
        elif (a[mid]>x):
            high=mid-1;
        else:
            low=mid+1;
    return ((low+high)//2);#若x不在a中，则返回x中位数的位置作为该元素的位置

maxrange=[];#用来存储实例对应y各个类别的概率的集合
a=[2,1];#用来表示实例
for i in range(ynum):
    percents=pyck[i];#用来存储实例对应y各个类别的概率
    print("\n",percents);
    node=root.next;#从第一个节点即X特征向量的第一列开始
    for j in range(m):
        wcx=twobreak(node.cx,a[j]);#定位a[i]元素在cx数组中的位置
        #print(wcx);
        percents*=pxy[i][wcx];#累乘求第i个类别概率
        node=node.next;#遍历各个节点
    #print("\n",percents)
    maxrange.append(percents);
    #print(maxrange);
    
###第六步：
###确定实例最大概率的类
k=maxrange.index(max(maxrange));
print("实例x的类别是:{}".format(c[k]));