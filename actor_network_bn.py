#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import os
import numpy as np
import math
from Parameter import Param
param = Param()

# Hyper Parameters
# # Robot Contorl Network Param
# LAYER1_SIZE = 400
# LAYER2_SIZE = 300
# DRL-EMS-HEV Network Param
LAYER1_SIZE = param.layer1_size
LAYER2_SIZE = param.layer2_size
LAYER3_SIZE = param.layer3_size
LEARNING_RATE = param.learning_rate
TAU = param.TAU

model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model', 'actor')
# print("============")
# print(model_dir)


class ActorNetwork:
	def __init__(self,sess,state_dim,action_dim):
		self.time_step = 0
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		# create actor network
		self.state_input,self.action_output,self.net,self.is_training = self.create_network(state_dim, action_dim)

		# create target actor network
		self.target_state_input,self.target_action_output,self.target_update,self.target_is_training = self.create_target_network(state_dim,action_dim,self.net)

		# define training rules
		self.create_training_method()
		self.sess.run(tf.initialize_all_variables())

		self.update_target()
		self.load_network()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

	def create_network(self,state_dim, action_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE
		# DRL-EMS-HEV Param
		layer3_size = LAYER3_SIZE

		state_input = tf.placeholder("float",[None,state_dim])
		is_training = tf.placeholder(tf.bool)

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size)
		b2 = self.variable([layer2_size],layer1_size)

		# # # Robot Contorl structure
		# W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-0.003, 0.003))
		# b3 = tf.Variable(tf.random_uniform([action_dim],-0.003,0.003))

		# DRL-EMS-HEV structure
		W3 = self.variable([layer2_size,layer3_size],layer2_size)
		b3 = self.variable([layer3_size],layer2_size)
		W4 = tf.Variable(tf.random_uniform([layer3_size,action_dim],-0.003, 0.003))
		b4 = tf.Variable(tf.random_uniform([action_dim],-0.003,0.003))

		layer1 = tf.matmul(state_input,W1) + b1
		layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='batch_norm_1',activation=tf.nn.relu)
		layer2 = tf.matmul(layer1_bn,W2) + b2
		layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='batch_norm_2',activation=tf.nn.relu)

		# # Robot Control structure
		# action = tf.matmul(layer2_bn, W3) + b3

		# DRL-EMS-HEV structure
		layer3 = tf.matmul(layer2_bn,W3) + b3
		layer3_bn = self.batch_norm_layer(layer3,training_phase=is_training,scope_bn='batch_norm_3',activation=tf.nn.relu)
		action = tf.matmul(layer3_bn, W4) + b4

		action_eng = self.batch_norm_layer(action[:, None, 0],training_phase=is_training,scope_bn='action_eng',activation=tf.sigmoid)
		# action_gen = self.batch_norm_layer(action[:, None, 1],training_phase=is_training,scope_bn='action_gen',activation=tf.tanh)
		# action_mot = self.batch_norm_layer(action[:, None, 2],training_phase=is_training,scope_bn='action_mot',activation=tf.tanh)
		action_gen = self.batch_norm_layer(action[:, None, 1],training_phase=is_training,scope_bn='action_gen',activation=tf.sigmoid)
		action_gen = 2*action_gen -1
		action_mot = self.batch_norm_layer(action[:, None, 2],training_phase=is_training,scope_bn='action_mot',activation=tf.sigmoid)
		action_mot = 2*action_mot -1
		
		# # Robot Control Structure
		# action_eng = tf.sigmoid(action[:, None, 0])
		# action_gen = tf.tanh(action[:, None, 1])
		# action_mot = tf.tanh(action[:, None, 2])
		action = tf.concat([action_eng, action_gen, action_mot], axis=-1)

		# return state_input, action, [W1,b1,W2,b2,W3,b3], is_training

		return state_input, action, [W1,b1,W2,b2,W3,b3,W4,b4], is_training

	def create_target_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		is_training = tf.placeholder(tf.bool)
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.matmul(state_input,target_net[0]) + target_net[1]
		layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='target_batch_norm_1',activation=tf.nn.relu)
		layer2 = tf.matmul(layer1_bn,target_net[2]) + target_net[3]
		layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='target_batch_norm_2',activation=tf.nn.relu)

		# # Robot Control structure
		# action = tf.matmul(layer2_bn, target_net[4]) + target_net[5]

		# DRL-EMS-HEV Structure
		layer3 = tf.matmul(layer2_bn,target_net[4]) + target_net[5]
		layer3_bn = self.batch_norm_layer(layer3,training_phase=is_training,scope_bn='target_batch_norm_3',activation=tf.nn.relu)
		action = tf.matmul(layer3_bn, target_net[6]) + target_net[7]

		action_eng = self.batch_norm_layer(action[:, None, 0], training_phase=is_training, scope_bn='target_action_eng', activation=tf.sigmoid)
		# action_gen = self.batch_norm_layer(action[:, None, 1], training_phase=is_training, scope_bn='target_action_gen', activation=tf.tanh)
		# action_mot = self.batch_norm_layer(action[:, None, 2], training_phase=is_training, scope_bn='target_action_mot', activation=tf.tanh)	
		action_gen = self.batch_norm_layer(action[:, None, 1],training_phase=is_training,scope_bn='target_action_gen',activation=tf.sigmoid)
		action_gen = 2*action_gen -1
		action_mot = self.batch_norm_layer(action[:, None, 2],training_phase=is_training,scope_bn='target_action_mot',activation=tf.sigmoid)
		action_mot = 2*action_mot -1
		
		# # Robot Control Sturcture
		# action_eng = tf.sigmoid(action[:, None, 0])
		# action_gen = tf.tanh(action[:, None, 1])
		# action_mot = tf.tanh(action[:, None, 2])
		# action = tf.concat([action_eng, action_gen, action_mot], axis=-1)

		return state_input, action, target_update, is_training

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,q_gradient_batch,state_batch):
		self.time_step += 1
		self.sess.run(self.optimizer,feed_dict={self.q_gradient_input:q_gradient_batch, self.state_input:state_batch, self.is_training: True})

	def actions(self,state_batch):
		return self.sess.run(self.action_output,feed_dict={self.state_input:state_batch, self.is_training: True})

	def action(self,state):
		return self.sess.run(self.action_output,feed_dict={self.state_input:[state], self.is_training: False})[0]

	def target_actions(self,state_batch):
		return self.sess.run(self.target_action_output,feed_dict={self.target_state_input: state_batch, self.target_is_training: True})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

	def batch_norm_layer(self,x,training_phase,scope_bn,activation=None):
		return tf.cond(training_phase, 
		lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True, updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
		lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True, updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))

	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(model_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded actoe network")
		else:
			print("Could not find old network weights")

	def save_network(self,time_step):
		print('save actor-network...',time_step)
		self.saver.save(self.sess, model_dir + 'actor-network', global_step=time_step)

