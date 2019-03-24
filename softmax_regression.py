#!/usr/bin/env python
# coding: utf-8

# In[3]:


#coding:urf-8


# In[6]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[7]:


# 创建x, x是一个占位符，代表识别的图片


# In[10]:


x = tf.placeholder(tf.float32,[None,784])


# In[14]:


# w是softmax的参数，讲一个784维的输入转换为一个10维的输出
# tensorflow 中变量的参数用tf.Variable表示
# b是有一个Softmax模型的参数，一般叫做“偏置项”（bias）


# In[17]:


w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# In[19]:


# y是模型输出
# y_是实际的图像标签，以占位符表示


# In[22]:


y = tf.nn.softmax(tf.matmul(x,w)+b)
y_ = tf.placeholder(tf.float32, [None, 10])


# In[24]:


cross_entropy =     tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))


# In[25]:


# 梯度下降优化参数w和b


# In[27]:


train_step = tf.train.GradientDescentOptimizer(0.01).    minimize(cross_entropy)


# In[31]:


# 创建一个session。只有在session中运算才会执行


# In[33]:


sess = tf.InteractiveSession()
# 运行之前初试化变量，分配内存
tf.global_variables_initializer().run()


# In[35]:


for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})


# In[45]:


correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[47]:


print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:                                  mnist.test.labels}))

