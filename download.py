#!/usr/bin/env python
# coding: utf-8

# In[3]:


#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[10]:


print(mnist.train.images.shape)
print(mnist.train.labels.shape)


# In[11]:


print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)


# In[13]:


print(mnist.test.images.shape)
print(mnist.test.labels.shape)


# In[14]:


print(mnist.train.images[0,:])

