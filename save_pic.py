#!/usr/bin/env python
# coding: utf-8

# In[4]:


#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os


# ### 读取MNIST数据集。如果不存在会下载

# In[5]:


mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)


# ### 把原始图片保存在MNIST_data/raw/文件夹下
# ### 如果没有这个文件夹会自动创建

# In[6]:


save_dir='MNIST_data/raw'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)


# ### 保存前20张图片

# In[7]:


for i in range(20):
    image_array = mnist.train.images[i,:]
    image_array = image_array.reshape(28,28)
    filename=save_dir+'mnist_train_%d.jpg'%i
    scipy.misc.toimage(image_array,cmin=0.0,cmax=1.0).save(filename)


# In[8]:


print(mnist.train.labels[0,:])

