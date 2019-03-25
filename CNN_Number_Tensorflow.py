# -*- coding: utf-8 -*-


#import math
import numpy as np

#import h5py
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops
#import tf_utils
import cnn_utils
#import cnn_utils
X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = cnn_utils.load_dataset()
#index=6
#print("Y_train_orig is:\n",Y_train_orig.shape)#(1, 1080)
#print("Y_train_orig is:\n",Y_train_orig)
#plt.imshow(X_train_orig[index])
#print("y=",np.squeeze(Y_train_orig[:,index]))
X_train=X_train_orig/255.
X_test=X_test_orig/255.
Y_train=cnn_utils.convert_to_one_hot(Y_train_orig,6).T
Y_test=cnn_utils.convert_to_one_hot(Y_test_orig,6).T
#创建占位符，详见https://blog.csdn.net/u013733326/article/details/80086090解释
def create_placeholder(n_H0,n_W0,n_C0,n_y):
    x=tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0])
    y=tf.placeholder(tf.float32,[None,n_y])
    return x,y
'''
x,y=create_placeholder(64,64,3,6)
print(x,"\n",y)
'''
#2.初始化参数
def init_parameters():
    tf.set_random_seed(1)
    w1=tf.get_variable("w1",[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    w2=tf.get_variable("w2",[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters={"w1":w1,"w2":w2}
    return parameters
#print(init_parameters())
#测试
'''
tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters=init_parameters()
    init=tf.global_variables_initializer()
    sess_test.run(init)
    print("w1:",parameters["w1"])
    print("w1=",parameters["w1"].eval()[1,1,1])
    print("w2=",parameters["w2"].eval()[1,1,1])
    sess_test.close()
'''
    
#前向传播
def forward_propgration(input_x,parameters):
    '''
    输入：
    1.训练集数据
    2.滤波器权重
    3.padding大小
    输出：
    FC后的Z3值
    '''

    w1=parameters['w1']
    w2=parameters['w2']  
    z1=tf.nn.conv2d(input_x,w1,strides=[1,1,1,1],padding="SAME")
    a1=tf.nn.relu(z1)
    pool_a1=tf.nn.max_pool(a1,ksize=[1,8,8,1],strides=[1,8,8,1],padding="SAME")
    z2=tf.nn.conv2d(pool_a1,w2,strides=[1,1,1,1],padding="SAME" )
    a2=tf.nn.relu(z2)
    pool_a2=tf.nn.max_pool(a2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")
    #一维化上一层的输出
    fc=tf.contrib.layers.flatten(pool_a2)
    z3=tf.contrib.layers.fully_connected(fc,6,activation_fn=None)#FC权重多少?,没有权重
    return z3
#测试一下
    '''
tf.reset_default_graph()
np.random.seed(1)
with tf.Session() as session_test:
    X,Y=create_placeholder(64,64,3,6)
    parameters=init_parameters()
    #init=tf.global_variables_initializer()
    z3=forward_propgration(X,parameters)
    init=tf.global_variables_initializer()
    session_test.run(init)
    a=session_test.run(z3,{X:np.random.randn(2,64,64,3),Y:np.random.randn(2,6)})
    print("forward result is:\n"+str(a))
    session_test.close()
#正向传播的每个过程后的大小
    '''
def compute_cost(z3,Y):
    '''
    输入:前向传播的Z3和标签值
    输出:损失值
    '''
    cost=tf.nn.softmax_cross_entropy_with_logits(logits=z3,labels=Y)
    mean_cost=tf.reduce_mean(cost)
    return mean_cost
    
'''    
#test
tf.reset_default_graph()
np.random.seed(1)
with tf.Session() as session_test:
    X,Y=create_placeholder(64,64,3,6)
    parameters=init_parameters()
    z3=forward_propgration(X,parameters)
    mean_cost=compute_cost(z3,Y)
    init=tf.global_variables_initializer()
    session_test.run(init)
    a=session_test.run(mean_cost,{X:np.random.randn(4,64,64,3),Y:np.random.randn(4,6)})
    print("mean_cost is:\n",a)
    session_test.close()
'''
#构建模型
'''
模型是将上述所有的整理在一起
1.创建占位符
2.初始化参数
3.前向传播
4.反向传播与优化器
5.创建Session()运行模型

def create_placeholder(n_H0,n_W0,n_C0,n_y):
    x=tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0])
    y=tf.placeholder(tf.float32,[None,n_y])
    return x,y

'''
def model(train_x,train_y,test_x,test_y,learning_rate=0.0009,num_epoch=100,minibatch_size=64,print_cost=True,plot_cost=True):
    
    '''
    input:
    1.输入层，标签层，滤波器参数
    def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    '''
    ops.reset_default_graph()
    tf.set_random_seed(1)
    cost_list=[]
    X,Y=create_placeholder(train_x.shape[1],train_x.shape[2],train_x.shape[3],train_x.shape[0])
    print(X,Y)
    parameters=init_parameters()
    print("1")
    z3=forward_propgration(train_x,parameters)#break
    print("2")
    mean_cost=compute_cost(z3,Y)
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_cost)
    init=tf.global_variables_initializer()
    with tf.Session() as session_model:
        session_model.run(init)
        
        #由于设置了epoch，所以在反向传播的时候会不断的更新参数
        for epoch in  range(num_epoch):
            minibatch_cost=0
            num_minibatches=int(train_x[0].shpae/minibatch_size)
            batch_data=cnn_utils.random_mini_batches(train_x,train_y,minibatch_size=64,seed=0)
            for num_mini in batch_data:
               (x,y)=num_mini
               #最小化这个数据块的成本 
               _,zpone=session_model.run([optimizer,mean_cost],feed_dict={X:x,Y:y})              
               #累加一个样本中minibatch的成本和
               minibatch_cost+=zpone/num_minibatches
            if print_cost:               
                if epoch %5==0:                    
                    print("当前是第",epoch,"个epoch","成本为：",minibatch_cost)
            if epoch %1==0:
                cost_list.append(minibatch_cost)
        if plot_cost:
            plt.plot(np.squeeze(cost_list))
            plt.xlabel("epoch")
            plt.ylabel("cost")
            plt.title("learing rate =",learning_rate)
            plt.show()
        #预测数据
        predict_op=tf.arg_max(z3,1)
        corrent_prediction=tf.equal(predict_op,tf.arg_max(Y,1))
        #计算准确度
        accuracy=tf.reduce_mean(tf.cast(corrent_prediction,"float"))
        print("corrent_accyracy=",accuracy)
        train_accuracy=accuracy.eval({X:train_x,Y:train_y})
        test_accuracy=accuracy.eval({X:test_x,Y:test_y})
        print("训练集准确度:",train_accuracy)
        print("测试集准确度:",test_accuracy)
        return(train_accuracy,test_accuracy,parameters)
#启动模型
start_time=time.clock()
_,_,parameters=model(X_train,Y_train,X_test,Y_test,num_epoch=150)
end_time=time.clock()            
        
    