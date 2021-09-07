#!/usr/bin/env python3
import tensorflow as tf
import os
import numpy as np
import math
from Parameter import Param
param = Param()

# # Robot Control Param
# LAYER1_SIZE = 400
# LAYER2_SIZE = 300

# DRL-EMS-HEV Param
LAYER1_SIZE = param.layer1_size
LAYER2_SIZE = param.layer2_size
LAYER3_SIZE = param.layer3_size

LEARNING_RATE = param.learning_rate
TAU = param.TAU

model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model', 'critic')


class CriticNetwork:
	def __init__(self,sess,state_dim,action_dim):
		self.time_step = 0
		self.sess = sess
		# create q network
		self.state_input, self.action_input, self.q_value_output,\
		self.net = self.create_q_network(state_dim,action_dim)

		# create target q network (the same structure with q network)
		self.target_state_input, self.target_action_input, self.target_q_value_output,\
		self.target_update = self.create_target_q_network(state_dim,action_dim,self.net)

		self.create_training_method()
		self.sess.run(tf.initialize_all_variables())
			
		self.update_target()
		self.load_network()

	def create_training_method(self):
		self.y_input = tf.placeholder("float",[None,1])
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output))
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
		self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

	def create_q_network(self,state_dim,action_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE
		# # DRL-EMS-HEV structure
		layer3_size = LAYER3_SIZE

		state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])

		# # Robot Control structure
		# W1 = self.variable([state_dim,layer1_size],state_dim)
		# b1 = self.variable([layer1_size],state_dim)
		# W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
		# W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
		# b2 = self.variable([layer2_size],layer1_size+action_dim)
		# W3 = tf.Variable(tf.random_uniform([layer2_size,1],-0.003,0.003))
		# b3 = tf.Variable(tf.random_uniform([1],-0.003,0.003))

		# layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		# layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
		# q_value_output = tf.matmul(layer2,W3) + b3

		# # Robot Control Return
		# return state_input,action_input,q_value_output,[W1,b1,W2,W2_action,b2,W3,b3]

		# # DRL-EMS-HEV structure
		W1_s = tf.get_variable('w1_s', [state_dim, layer1_size])
		W1_a = tf.get_variable('w1_a', [action_dim, layer1_size])
		b1 = tf.get_variable('b1', [1, layer1_size])
		W2 = tf.get_variable('w2', [layer1_size, layer2_size])
		b2 = tf.get_variable('b2', [1, layer2_size])
		W3 = tf.get_variable('w3', [layer2_size, layer3_size])
		b3 = tf.get_variable('b3', [1, layer3_size])
		W4 = tf.get_variable('w4', [layer3_size, 1])
		b4 = tf.get_variable('b4', [1, layer3_size])
		net1 = tf.nn.relu(tf.matmul(state_input, W1_s) + tf.matmul(action_input, W1_a) + b1)
		net2 = tf.nn.relu(tf.matmul(net1, W2) + b2)
		net3 = tf.nn.relu(tf.matmul(net2, W3) + b3)
		q_value_output = tf.matmul(net3,W4) + b4

		# DRL-EMS-HEV Return
		return state_input,action_input,q_value_output,[W1_s,W1_a,b1,W2,b2,W3,b3,W4,b4]

	def create_target_q_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])

		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		# # Robot Control structure
		# layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		# layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
		# q_value_output = tf.matmul(layer2,target_net[5]) + target_net[6]

		# DRL-EMS-HEV structure
		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + tf.matmul(action_input, target_net[1]) + target_net[2])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[3]) + target_net[4])
		layer3 = tf.nn.relu(tf.matmul(layer2,target_net[5]) + target_net[6])
		q_value_output = tf.matmul(layer3,target_net[7]) + target_net[8]

		return state_input,action_input,q_value_output,target_update

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,y_batch,state_batch,action_batch):
		self.time_step += 1
		self.sess.run(self.optimizer,feed_dict={self.y_input:y_batch, self.state_input:state_batch, self.action_input:action_batch})

	def gradients(self,state_batch,action_batch):
		return self.sess.run(self.action_gradients,feed_dict={self.state_input:state_batch, self.action_input:action_batch})[0]

	def target_q(self,state_batch,action_batch):
		return self.sess.run(self.target_q_value_output,feed_dict={self.target_state_input:state_batch, self.target_action_input:action_batch})

	def q_value(self,state_batch,action_batch):
		return self.sess.run(self.q_value_output,feed_dict={self.state_input:state_batch, self.action_input:action_batch})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(model_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded critic network")
		else:
			print("Could not find old network weights")

	def save_network(self,time_step):
		print('save critic-network...',time_step)
		self.saver.save(self.sess, model_dir + 'critic-network', global_step=time_step)
