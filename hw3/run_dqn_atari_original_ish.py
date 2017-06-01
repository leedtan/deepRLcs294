import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from Utils import ops
import dqn
from dqn_utils import *
from atari_wrappers import *



def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            bn0 = ops.batch_norm(name='bn0')
            bn1 = ops.batch_norm(name='bn1')
            bn2 = ops.batch_norm(name='bn2')
            # original architecture
            out1 = ops.lrelu(bn0(layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4)))
            out2 = ops.lrelu(bn1(layers.convolution2d(out1, num_outputs=64, kernel_size=4, stride=2)))
            out3 = ops.lrelu(bn2(layers.convolution2d(out2, num_outputs=64, kernel_size=3, stride=1)))
            
            test_out1 = ops.lrelu(bn0(layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4), train=False))
            test_out2 = ops.lrelu(bn1(layers.convolution2d(test_out1, num_outputs=64, kernel_size=4, stride=2), train=False))
            test_out3 = ops.lrelu(bn2(layers.convolution2d(test_out2, num_outputs=64, kernel_size=3, stride=1), train=False))
            
        out4 = tf.concat((layers.flatten(out2), layers.flatten(test_out3)),1)
        
        test_out4 = tf.concat((layers.flatten(test_out2), layers.flatten(test_out3)),1)
        with tf.variable_scope("action_value"):
            bn3 = ops.batch_norm(name='bn3')
            out5 = tf.concat((out4, ops.lrelu(bn3(layers.fully_connected(out4, num_outputs=512)))),1)
            out6 = layers.fully_connected(out5, num_outputs=num_actions, activation_fn=None)
            
            test_out5 = tf.concat((test_out4, ops.lrelu(bn3(layers.fully_connected(out4, num_outputs=512), train=False))),1)
            test_out6 = layers.fully_connected(test_out5, num_outputs=num_actions, activation_fn=None)

        return out6, test_out6

def atari_learn(env,
                session,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    session = get_session()
    atari_learn(env, session, num_timesteps=task.max_timesteps)

if __name__ == "__main__":
    main()
