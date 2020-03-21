from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from train import CatDQNTrainer

from stock_env import StockEnvBasic

from helpers.general import get_file_names


import numpy as np

import tensorflow as tf
from helpers.tf_helper import *

from tensorflow.keras.optimizers import Adam
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from cparse import config

#=========================================================================

c = config()
"""
    Main - collect data, load model, train, save model checkpoints
    
    CatDQNTrainer: Class for train iterations
    buffer: replay buffer to store collected data
        buffer is turned into dataset for training
"""

#====================================COLLECT=====================================
def collection_run(c,the_buffer,date_strs,cdict):
    """Fills replay buffer with experience from date_strs
    """
    n_trades = 0
    for date_str in date_strs:
        print(date_str)
        contract_str = cdict[date_str]
        the_env=StockEnvBasic(date_str,contract_str, **c.stock_env)
        the_env = TFPyEnvironment(the_env)
        time_step = the_env.reset()

        while not tf.reduce_all(time_step.is_last()):
            action = 1
            next_ts = the_env.step(action)
            pol_step = policy_step.PolicyStep(action)
            traj = trajectory.from_transition(time_step, pol_step, next_ts)
            the_buffer.add_batch(traj)
            time_step = next_ts
    return the_buffer




#====================================TRAIN======================================
#TF function decorator speeds train
@tf.function
def train_catdqn(the_experience,the_weights,the_catdqn):
    loss_info = the_catdqn._train(the_experience,the_weights)
    return loss_info

if __name__=="__main__":
    value_net,checkpoint=load_model_checkpoint(c)
    optimizer = Adam(**c.optimizer_vars)
    observation_spec = get_env_specs(c)[0]
    #Create Trainer
    catdqn_trainer = CatDQNTrainer(
                    optimizer=optimizer,
                    mod_net=value_net,
                    observation_spec = observation_spec,
                    **c.trainer_vars)

    checkpoint = tf.train.Checkpoint(module = catdqn_trainer._mod_net)
    with tf.device('/CPU:0'):
        cdict,date_strs = get_file_names(c.file_train,use_random=True)
        #Create Buffer
        new_buffer = get_tf_buffers(c=c,max_length = (c.default_env['time_end']-c.default_env['time_start_trade'])*len(date_strs))
        
        #Collect
        new_buffer= collection_run(
                            c,
                            new_buffer,
                            date_strs,
                            cdict)

        dataset = new_buffer.as_dataset(num_parallel_calls=2, sample_batch_size=c.batch_size, num_steps=c.step_size).prefetch(2).repeat(-1)

    #Convert buffer for train
    with tf.device('/GPU:0'):
        
        iterator = iter(dataset)

        #Train
        for n_step in range(c.n_trains):
            experience,_ = next(iterator) #if tf_buffer use experience,_

            weights = tf.ones((c.batch_size,1),dtype= tf.float32)

            loss_info = train_catdqn(experience,weights,catdqn_trainer)
            
            if (n_step+1) % c.save_interval==0:
                print(n_step+1,"Loss:",loss_info.loss.numpy()) 
                checkpoint.save(c.model_dir+'ver')
    checkpoint.save(c.model_dir+'ver')
    print("Complete")
