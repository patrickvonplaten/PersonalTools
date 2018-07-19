#!/usr/bin/env python3 

from tfSi6Proc.basics.ml.returnn.network import ReturnnTfNetworkWrapper
import tensorflow as tf
import sys

configFile = sys.argv[1]
modelFile = sys.argv[2]

sess = tf.Session()
tfNetwork = ReturnnTfNetworkWrapper(configFile, modelFile, sess).getTFNetwork()
tfNetwork.print_network_info()

sess.close()
print "==========FINISH=========="
