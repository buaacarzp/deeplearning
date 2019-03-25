# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(peng.zhou)s
"""


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt #从matplotlib库中调用pyplot.py
import tf_utils
import time
from tensorflow.python.framework import ops
'''
TensorFlow的编码步骤： 
1. 创建Tensors (即变量) ，这些Tensors均未被执行  
2. 通过Tensors之间的运算操作，实现目标函数，比如代价函数 
3. Tensors初始化 
4. Session创建 
5. Session运行，该步骤是对目标函数的执行
'''

'''
使用tensorflow构建第一个神经网络
'''
x_train_root,y_train_root,x_test_root,y_test_root,classes=tf_utils.load_dataset()
x_train_re=x_train_root.reshape(x_train_root.shape[0],-1).T
x_test_re=x_test_root.reshape(x_test_root.shape[0],-1).T
#归一化数据
x_train=x_train_re/255
x_test=x_test_re/255

#转换为One_hot矩阵
y_train_oh=tf_utils.convert_to_one_hot(y_train_root,6)
y_test_oh=tf_utils.convert_to_one_hot(y_test_root,6)
'''
建立LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX网络
'''
#创建占位符，等到数据喂入session.run()
def create_placeholder(n_x,n_y):
    x=tf.placeholder(tf.float32,[n_x,None],name="x")
    y=tf.placeholder(tf.float32,[n_y,None],name="y")
    print("placeholder:",x,y)
    return x,y
def initialize_parameters():
    
    #初始化神经网络的参数
    '''
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]
        返回：带有w,b参数的字典形式
        '''
    tf.set_random_seed(1)#指定随机种子
    w1=tf.get_variable("w1",[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1=tf.get_variable("b1",[25,1],initializer=tf.zeros_initializer())
    w2=tf.get_variable("w2",[12,25],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2=tf.get_variable("b2",[12,1],initializer=tf.zeros_initializer())
    w3=tf.get_variable("w3",[6,12],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3=tf.get_variable("b3",[6,1],initializer=tf.zeros_initializer())
    parameters={"W1":w1,
                "b1":b1,
                "W2":w2,
                "b2":b2,
                "W3":w3,
                "b3":b3,
            }
    return parameters

#前向传播
def forward_propagation(train_x,parameters):
    '''
    input:w,b,train_x,train_y 
    output:最后一个Linear层的输出
    '''
    w1=parameters["W1"]
    b1=parameters["b1"]
    w2=parameters["W2"]
    b2=parameters["b2"]
    w3=parameters["W3"]
    b3=parameters["b3"]
    z1=tf.add(tf.matmul(w1,train_x),b1)
    a1=tf.nn.relu(z1)
    z2=tf.add(tf.matmul(w2,a1),b2)
    a2=tf.nn.relu(z2)
    z3=tf.add(tf.matmul(w3,a2),b3)
    print("z3.shape=",z3.shape)
    return z3

#计算成本
'''
tf.reduce_mean :用于计算平均值

reduce_mean(input_tensor,

                axis=None,

                keep_dims=False,

                name=None,

                reduction_indices=None)
第一个参数input_tensor： 输入的待降维的tensor;
第二个参数axis： 指定的轴，如果不指定，则计算所有元素的均值;
第三个参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，
设置为False，输出结果会降低维度;
第四个参数name： 操作的名称;第五个参数 reduction_indices：在以前版本中用来指定轴，已弃用;

tf.nn.sigmoid_cross_entropy_with_logits(logits=...,labels=...):计算成本
函数的InputJ就是Logits:这个logits需要经过sigmoid转化为a，之后再与标签值计算cost
tf.nn.softmax_cross_entropy_with_logits（logits=...,labels=...):计算成本
与sigmoid不同，它是经过softmax转化为a，得到相似概率值 然后与one_hot的值进行cost成本计算
    

'''
def compute_cost(z3,y):
    logits=tf.transpose(z3)#转置,可以用.T吗？
    labels=tf.transpose(y)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost
#反向传播，更新参数w,b
#反向传播是在计算成本函数后创建的一个“optimizer”优化器对象，运行tf.Session()时候，必须将此类
#与成本函数一起调用，当被调用的时候，它将使用选择色的方法和学习率对给定的成本进行优化
#optimizer=tf.train.GradientDescentOptimizer(learningrate=0.001).minimize(cost)
#这里不用上面的梯度下降来做，而是选用minibatch
#构建模型，把反向传播放在构建模型这块
def model(x_train,y_train,x_test,y_test,learning_rate=0.0001,num_epochs=200,minibatch_size=32,print_cost=True,is_plot=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed=3
    (n_x,m)=x_train.shape
    print("here is ",n_x,m)
    n_y=y_train.shape[0]
    print("n_y:",n_y)
    costs=[]
    #给x:训练集输入,Y:对应的标签值创建Placeholder
    x,Y=create_placeholder(n_x,n_y)#将输入与输出节点数量传入作为placeholder的size
    #初始化参数W,b
    parameters=initialize_parameters()
    #前向传播
    a3=forward_propagation(x,parameters)
    #计算损失函数
    print("a3's sieze=",a3.shape)
    cost=compute_cost(a3,Y)
    #反向传播，使用Adam优化
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #初始化所有的全局变量
    init=tf.global_variables_initializer()
    print("end?")
    #开始会话：创建session和迭代
    with tf.Session() as session:
        session.run(init)
        for epoch  in range(num_epochs):
            epoch_cost=0
            num_minibatches=int(m/minibatch_size)#minibatch的总数量
            seed=seed+1
            minibatches=tf_utils.random_mini_batches(x_train,y_train,minibatch_size,seed)
            for minibatche in minibatches:
                #选择一个minibatch
                (minibatch_x,minibatch_y)=minibatche
                #数据已经准备好了，开始运行session
                _,minibatch_cost=session.run([optimizer,cost],feed_dict={x:minibatch_x,Y:minibatch_y})
                #minibatch如何在一次迭代中计算误差？
                epoch_cost=epoch_cost+minibatch_cost/num_minibatches
            if epoch %5==0:
                costs.append(epoch_cost)
                if print_cost and epoch %100 ==0:
                    print("epch=",epoch,"epoch_cost=",epoch_cost)
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel("cost")
            plt.xlabel("iteration")
            plt.title("learing_rate="+str(learning_rate))
            plt.show()
            #保存学习后的参数
            parameters=session.run(parameters)
            print("参数已经保存到session")#?
            #计算当前的预测结果
            correct_prediction=tf.equal(tf.argmax(a3),tf.argmax(Y))
            #计算准确率
            accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
            print("训练集的准确率:",accuracy.eval({x:x_train,Y:y_train}))
            print("测试集的准确率:",accuracy.eval({x:x_test,Y:y_test}))   
            return parameters
#开始时间
start_time=time.clock()
#开始训练
parameters=model(x_train,y_train_oh,x_test,y_test_oh)
#结束时间
end_time=time.clock()
#计算时差
print("cpu的执行时间:",end_time-start_time,"(s)")


            
    
    
    
    
    

    
   
    
    
    

