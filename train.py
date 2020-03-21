from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import gin
from tf_agents.agents import tf_agent
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import tensor_normalizer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from helpers.catdqn import cat_discounted_return
from tensorflow.keras import losses
CatDQNLossInfo = collections.namedtuple('CatDQNLossInfo', (
    'value_estimation_loss',
))

@gin.configurable
class CatDQNTrainer():
    """CatDQN"""
    def __init__(self,
                   optimizer=None,
                   mod_net=None,
                   observation_spec=None,
                   discount_factor=0.95,
                   num_epochs = 15,
                   normalize_rewards=False,
                   reward_norm_clipping=10.0,
                   gradient_clipping=None,
                   name=None):
        tf.Module.__init__(self, name=name)
        self._optimizer = optimizer
        self._mod_net = mod_net
        self._discount_factor = discount_factor
        self._num_epochs = num_epochs
        self._reward_norm_clipping = reward_norm_clipping
        self._gradient_clipping = gradient_clipping or 0.0
        self._observation_spec=observation_spec
        self._reward_normalizer = None
        if normalize_rewards:
          self._reward_normalizer = tensor_normalizer.StreamingTensorNormalizer(
              tensor_spec.TensorSpec([], tf.float32), scope='normalize_reward')
        
        super(CatDQNTrainer, self).__init__()
    def _train(self, experience,weights):
        # Get individual tensors from transitions.
        (time_steps, policy_steps_,next_time_steps) = trajectory.to_transition(experience)
        #observations = time_steps.observation
        actions = policy_steps_.action

        rewards = next_time_steps.reward
        print(rewards)
        discounts = next_time_steps.discount
        if self._reward_normalizer:
            rewards = self._reward_normalizer.normalize(
                rewards, 
                center_mean=False, clip_value=self._reward_norm_clipping)


        value_preds = self.double_batch_pred(self._mod_net,experience.observation,is_training=True)
        #print("VPRED",value_preds.shape,value_preds_2.shape)
    
        returns = self.compute_return(next_time_steps,value_preds)
        value_estimation_losses = []

        loss_info = None
        # For each epoch, create its own train op that depends on the previous one.
        for i_epoch in range(self._num_epochs):
          with tf.name_scope('epoch_%d' % i_epoch):


            # Build one epoch train op.
            with tf.GradientTape() as tape:
              loss_info = self.get_epoch_loss(
                time_steps,returns,weights)#action_distribution_parameters

            variables_to_train = self._mod_net.trainable_weights
            grads = tape.gradient(loss_info.loss, variables_to_train)
            # Tuple is used for py3, where zip is a generator producing values once.
            grads_and_vars = tuple(zip(grads, variables_to_train))
            if self._gradient_clipping > 0:
              grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars, self._gradient_clipping)




            self._optimizer.apply_gradients(
              grads_and_vars)#, global_step=self.train_step_counter)

            value_estimation_losses.append(loss_info.extra.value_estimation_loss)

        loss_info = tf.nest.map_structure(tf.identity, loss_info)
        return loss_info

    def compute_return(self, next_time_steps,value_preds):
        """Compute the Monte Carlo return
        Args:
          next_time_steps: batched tensor of TimeStep tuples after action is taken.
          value_preds: Batched value prediction tensor. Should have one more entry
            in time index than time_steps, with the final value corresponding to the
            value prediction of the final state.
        Returns:
          tuple of (return, normalized_advantage), both are batched tensors.
        """
        discounts = next_time_steps.discount * tf.constant(
            self._discount_factor, dtype=tf.float32)

        rewards = next_time_steps.reward


        # Normalize rewards if self._reward_normalizer is defined.
        if self._reward_normalizer:
            rewards = self._reward_normalizer.normalize(
                rewards, center_mean=False, clip_value=self._reward_norm_clipping)


        # Make discount 0.0 at end of each episode to restart cumulative sum
        #   end of each episode.
        episode_mask = common.get_episode_mask(next_time_steps)
        discounts *= episode_mask


        # Compute Monte Carlo returns.
        final_vpreds = value_preds[:,-1,:]
        returns = cat_discounted_return(rewards,discounts,final_vpreds)

        return returns
   
    def get_epoch_loss(self, time_steps, returns, weights):
        # Call all loss functions and add all loss values.
        value_estimation_loss = self.value_estimation_loss(time_steps, returns,weights)
        total_loss = value_estimation_loss
        return tf_agent.LossInfo(total_loss,
          CatDQNLossInfo(value_estimation_loss=value_estimation_loss))

    def value_estimation_loss(self,time_steps,returns,weights):
        """Computes the value estimation loss for actor-critic training.
        All tensors should have a single batch dimension.
        Args:
          time_steps: A batch of timesteps.
          returns: Per-timestep returns for value function to predict. (Should come
            from TD-lambda computation.)
          weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.  Includes a mask for invalid timesteps.
          debug_summaries: True if debug summaries should be created.
        Returns:
          value_estimation_loss: A scalar value_estimation_loss loss.
        """
        observation = time_steps.observation
        value_preds = self.double_batch_pred(self._mod_net,observation,is_training=True)

        value_estimation_error = losses.kullback_leibler_divergence(returns,value_preds)
        value_estimation_error *= weights
        value_estimation_loss = tf.reduce_mean(input_tensor=value_estimation_error)
        return value_estimation_loss
    def double_batch_pred(self,the_model,all_inputs,is_training=False):
        specs = self._observation_spec

        outer_dims = nest_utils.get_outer_array_shape(all_inputs, specs)
        all_inputs,_ = nest_utils.flatten_multi_batched_nested_tensors(all_inputs, specs)
        vals= the_model(all_inputs,is_training=is_training)
        vals = tf.reshape(vals,(*outer_dims,-1))
        return vals