from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import *

class ValueNet(Model):
    def __init__(self,num_atoms = 51,droprate = 0.5):
        super(ValueNet,self).__init__(name='valnet')   

        self.trunk = Sequential([
            LSTM(256),
            Dense(32,activation='relu'),
            Dense(32,activation='relu'),
        ])
        self.valuehead = Sequential([
            Dense(64,activation='relu'),
            Dense(64,activation='relu'),
            Dense(num_atoms,activation='softmax')
        ])
        self._droprate = droprate
    def __call__(self,x0,is_training=False):
        x0 = tf.cast(x0,dtype=tf.float32)

        merge = self.trunk(x0)
        if is_training:
            merge = tf.nn.dropout(merge,self._droprate)
        val = self.valuehead(merge)
        return val 
