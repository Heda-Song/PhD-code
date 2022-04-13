import tensorflow as tf

def dropblock(inp, keep_rate, block_size):
    """ dropblock on featuremaps.
    https://arxiv.org/abs/1810.12890
    """ 
    inp_shape = inp.shape.as_list() # [batch,w,h,c]
    centre_size = inp_shape[-2] - (block_size // 2) * 2
    alpha = (1-keep_rate) / (block_size**2) * (inp_shape[-2]**2) / ((inp_shape[-2] - block_size + 1)**2)
    distrib = tf.distributions.Bernoulli(probs=alpha)
    centre_mask = distrib.sample([centre_size,centre_size]) 
    pad_size = tf.cast(block_size // 2, tf.int32)
    pad_mask = tf.pad(centre_mask, paddings=[[pad_size,pad_size],[pad_size,pad_size]], mode='CONSTANT') # [w,h]
    pad_mask = tf.expand_dims(tf.expand_dims(pad_mask, 0), -1)
    mask = tf.nn.max_pool(pad_mask, [1,block_size,block_size,1], [1,1,1,1], padding='SAME') # [w,h]
    mask_final = tf.cast(1 - mask, tf.float32)
    mask = tf.tile(mask_final, [inp_shape[0],1,1,inp_shape[-1]])
    drop_inp = inp * mask
    scale = tf.cast(inp_shape[1] * inp_shape[2], tf.float32) / tf.reduce_sum(mask_final[0,:,:,0])
    scaled_drop_inp = tf.cond(tf.less(tf.reduce_sum(mask_final[0,:,:,0]), 0.00001), lambda:inp, lambda:scale*drop_inp)

    return scaled_drop_inp

