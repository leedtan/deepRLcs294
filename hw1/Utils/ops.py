
import tensorflow as tf
import numpy as np

eps = 1e-8

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                
                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')
                    
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def lrelu(x):
        return tf.maximum(x, 1e-2*x)
    
def noise(shape, noise_level):
    return tf.random_normal(shape, mean=0, stddev = noise_level)


def map2idx(liker, liked, ratings, unique_users):

    # dict mapping the id to an index
    user_map = dict(zip(unique_users, range(len(unique_users))))
    inverse_user_map = dict(zip(range(len(unique_users)), unique_users))

    pairs = []
    for u1, u2, r in zip(liker, liked, ratings):
        pairs.append((user_map[u1], user_map[u2], r))

    return np.array(pairs), user_map, inverse_user_map

#Normalize user outgoing ratings as proposed in ModelDesign.txt
def normalize_outgoing_likes(rating_events, unique_users, user_map):
    for u in unique_users:
        u_idx = user_map[u]
        giver_occurances = rating_events[:,0] == u_idx
        if np.sum(giver_occurances) > 0:
            rating_events[giver_occurances, 2] -= np.mean(rating_events[giver_occurances, 2])
            rating_events[giver_occurances, 2] /= (np.std(rating_events[giver_occurances, 2]) + eps)
    return rating_events


def split_all_train_val(obs, act, split=.1):

    shuffle  = np.random.permutation(obs.shape[0])
    partition = int(np.floor(obs.shape[0] * (1-split)))

    train_idx = shuffle[:partition]
    val_idx = shuffle[partition:]
    
    obs_trn = obs[train_idx,:]
    obs_val = obs[val_idx,:]
    
    act_trn = act[train_idx,:]
    act_val = act[val_idx,:]

    expert_train = {'observations':obs_trn,
                    'actions':act_trn}

    expert_val = {'observations':obs_val,
                    'actions':act_val}

    return expert_train, expert_val







