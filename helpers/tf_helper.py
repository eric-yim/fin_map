from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
#from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.trajectories import trajectory, policy_step
#from tf_agents.trajectories.time_step import TimeStep
from stock_env import StockEnvBasic

from model import ValueNet

def load_model_checkpoint(c):#returns the model at given chkpoint

    dir_name = tf.train.latest_checkpoint(c.model_dir)
    #if ver_name =='None':
    #    check_or_make_dir(dir_name)
        
    #else:
    #    dir_name = os.path.join(dir_name,ver_name)
    dummy_env= TFPyEnvironment(StockEnvBasic(**c.default_env))
    time_step = dummy_env.reset()

    temp = ValueNet(**c.model_vars)
    #initialize model
    temp(time_step.observation)
    checkpoint2 = tf.train.Checkpoint(module=temp)
    status=checkpoint2.restore(dir_name)
    return temp,checkpoint2

def get_env_specs(c):
    dummy_env= TFPyEnvironment(StockEnvBasic(**c.default_env))
    return dummy_env.observation_spec(),dummy_env.action_spec()

def get_tf_buffers(c,max_length=270):
    obs_spec,ac_spec = get_env_specs(c)
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = policy_step.PolicyStep(ac_spec)
    trajectory_spec = trajectory.from_transition(
        time_step_spec, action_spec , time_step_spec)
    the_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=trajectory_spec,
        batch_size=1,
        max_length=max_length)
    return the_replay_buffer