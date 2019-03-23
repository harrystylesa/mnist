#!/usr/bin/env python
# coding: utf-8

# In[12]:


#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[13]:


for i in range(20):
    one_hot_label=mnist.train.labels[i,:]
    label = np.argmax(one_hot_label)
    print('mnist_train_%d.jpg label:%d'%(i,label))

