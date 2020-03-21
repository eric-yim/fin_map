import tensorflow as tf
#TODO: target_support is currently fixed at tf.linspace(-10.0,10.0,51)

def project_distribution(supports, weights, target_support):
    """Projects a batch of (support, weights) onto target_support. (from tf agents cat dqn)
    """
    target_support_deltas = target_support[1:] - target_support[:-1]
    delta_z = target_support_deltas[0]
    v_min, v_max = target_support[0], target_support[-1]
    batch_size = tf.shape(supports)[0]
    num_dims = tf.shape(target_support)[0]
    clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]

    tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])

    reshaped_target_support = tf.tile(target_support[:, None], [batch_size, 1])
    reshaped_target_support = tf.reshape(reshaped_target_support,
                                         [batch_size, num_dims, 1])

    numerator = tf.abs(tiled_support - reshaped_target_support)
    quotient = 1 - (numerator / delta_z)

    clipped_quotient = tf.clip_by_value(quotient, 0, 1)

    weights = weights[:, None, :]

    inner_prod = clipped_quotient * weights

    projection = tf.reduce_sum(inner_prod, 3)
    projection = tf.reshape(projection, [batch_size, num_dims])
    return projection

def discounted_return_fn(accumulated_weights, reward_discount):
    #hardcoded target_support
    target_support = tf.linspace(-10.0,10.0,51)#TODO: allow user to input target_support
    
    reward, discount = reward_discount
    batch_size = reward.shape[0]
    num_dims = target_support.shape[0]
    tiled = tf.tile(target_support,[batch_size])
    tiled = tf.reshape(tiled,[batch_size,num_dims])
    supports = tiled * discount + reward
    proj = project_distribution(supports, accumulated_weights, target_support)
    return proj

def cat_discounted_return(rewards,discounts,final_preds):
    rewards = tf.expand_dims(tf.transpose(rewards),-1)
    discounts = tf.expand_dims(tf.transpose(discounts),-1)
    returns = tf.scan(
        fn=discounted_return_fn,
        elems=(rewards, discounts),
        reverse=True,
        initializer=final_preds,
        back_prop=False)
    returns = tf.transpose(returns,[1,0,2])
    return returns

def scalar_returns(returns):
    #hardcoded target_support
    target_support = tf.linspace(-10.0,10.0,51)#TODO: allow user to input target_support
    return tf.reduce_sum(tf.multiply(returns,target_support),axis=-1)