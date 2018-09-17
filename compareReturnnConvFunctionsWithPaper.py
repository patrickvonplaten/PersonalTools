import tensorflow as tf 
import numpy as np 
import ipdb

channels = 6 
timeInstances = 20
filterSize = 4
numFilter = 1
batchSize = 1

x_shape = [batchSize, timeInstances, channels, 1]  
f_shape = [filterSize, channels, 1, numFilter]

x_val = np.random.random_sample(x_shape).astype(np.float32)
f_val = np.random.random_sample(f_shape).astype(np.float32)

x = tf.constant(x_val, name="x", dtype=tf.float32)
f = tf.constant(f_val, name="f", dtype=tf.float32)
output = tf.nn.convolution(input=x, filter=f, padding="VALID")
ipdb.set_trace()

with tf.Session() as sess:
    y = sess.run(output)

yTest = np.zeros((channels, timeInstances - filterSize + 1), dtype=np.float32) 

for i in range(channels): 
    for j in range(timeInstances - filterSize + 1):
        for k in range(filterSize):
            yTest[i][j] += f_val[k][i][numFilter-1] * x_val[batchSize-1][j + k][i]

y = y.reshape(17)
yCompare = np.sum(yTest, axis=0)

print('Are equal:', np.allclose(y,yCompare))





