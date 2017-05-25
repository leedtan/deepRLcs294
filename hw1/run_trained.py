#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
	python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
			--num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

from Utils import ops
from Model import Model
import os
import sys
import time


def main():
	
	
	class Logger(object):
	    def __init__(self, filename="last_run_output.txt"):
	        self.terminal = sys.stdout
	        self.log = open(filename, "a")
	
	    def write(self, message):
	        self.terminal.write(message)
	        self.log.write(message)
	        self.flush()
	        
	    def flush(self):
	        self.log.flush()
	        
	sys.stdout = Logger("logs/" + str(os.path.basename(sys.argv[0])) +
	                    str(time.time()) + ".txt")
	
	print(sys.version)
	
	
	returns, expert_data = pickle.load( open( "objs.pickle", "rb" ) )
	
	obs, act = expert_data['observations'], expert_data['actions']
	
	expert_train, expert_val = ops.split_all_train_val(obs, act)
	
	size_obs = obs.shape[1]
	size_act = act.shape[1]
	
	model = Model(size_obs, size_act)
	model.restore()
	
	
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=20,
						help='Number of expert roll outs')
	args = parser.parse_args()

	print('loading and building expert policy')
	policy_fn = load_policy.load_policy(args.expert_policy_file)
	print('loaded and built')

	with tf.Session():
		tf_util.initialize()

		import gym
		env = gym.make(args.envname)
		max_steps = args.max_timesteps or env.spec.timestep_limit

		returns = []
		observations = []
		actions = []
		for i in range(args.num_rollouts):
			print('iter', i)
			obs = env.reset()
			done = False
			totalr = 0.
			steps = 0
			while not done:
				expert_action = policy_fn(obs[None,:])
				predicted_actions = model.get_yhat(np.array([obs]))
				observations.append(obs)
				actions.append(expert_action.flatten())
				obs, r, done, _ = env.step(predicted_actions)
				totalr += r
				steps += 1
				if args.render:
					env.render()
				if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
				if steps >= max_steps:
					break
			returns.append(totalr)

		print('returns', returns)
		print('mean return', np.mean(returns))
		print('std of return', np.std(returns))

		expert_data = {'observations': np.array(observations),
					   'actions': np.array(actions)}
		with open('objs.pickle', 'wb') as f:
				pickle.dump([returns, expert_data], f)
	
	'''
	with open('objs.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
		returns, expert_data = pickle.load(f)
	'''

if __name__ == '__main__':
	main()
