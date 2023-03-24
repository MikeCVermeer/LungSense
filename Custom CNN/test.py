from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# print the tensorflow version
import tensorflow as tf
print(tf.__version__ + '\n\n')

tf.sysconfig.get_build_info() 
