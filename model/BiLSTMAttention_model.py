# -*- coding: utf-8 -*-

# Created by csw on 2019/8/19.

import tensorflow as tf
import tensorflow.contrib.slim as slim


class BiLSTMAttention(object):
	def __init__(self, args):
		self.args = args

		self.input = tf.placeholder(tf.int64, [None, args.sequence_Length], name='input')
		self.label = tf.placeholder(tf.int64, [None], name='label')
		self.embeddingPosition = tf.placeholder(tf.float32, [None, args.sequence_Length, args.sequence_Length],
												name="embeddingPosition")
		self.KeepProb = tf.placeholder(tf.float32, name="KeepProb")
		self.global_step = tf.Variable(0, trainable=False)

		with tf.name_scope('embedding'):
			self.W = tf.Variable(tf.random_uniform([args.vocab_size, args.embedding_size], -1.0, 1.0), name='W')
			self.embedding_words = tf.nn.embedding_lookup(self.W, self.input)

		output, prediction = self.forward()

		with tf.name_scope('L2_norm'):
			tv = tf.trainable_variables(scope='Bi-LSTM')
			self.l2_norm = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
			self.l2_loss = args.l2_lambda * self.l2_norm
			tf.summary.scalar('l2_loss', self.l2_loss)

		with tf.name_scope('loss'):
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
																  labels=self.label)
			self.loss = tf.reduce_mean(loss)
			self.losses = self.loss + self.l2_loss
			tf.summary.scalar('loss', self.loss)
			tf.summary.scalar('losses', self.losses)

		with tf.name_scope('optimizer'):
			lr = tf.train.exponential_decay(args.lr, self.global_step,
											5000, 0.96, staircase=True)
			self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.losses,
																 global_step=self.global_step)
			tf.summary.scalar('lr', lr)

		with tf.name_scope('accuracy'):
			correctPred = tf.equal(prediction, self.label)
			self.accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
			tf.summary.scalar('accuracy', self.accuracy)
		self.merged = tf.summary.merge_all()

	def forward(self):
		with tf.variable_scope('Bi-LSTM'):
			for idx, hidddensize in enumerate(list(map(int, self.args.hiddensizes.split(',')))):
				with tf.variable_scope('Bi-LSTM' + str(idx)):
					lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidddensize,
																					   state_is_tuple=True))
					lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hidddensize,
																					   state_is_tuple=True))
					outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
																 self.embedding_words,
																 dtype=tf.float32,
																 scope='bi-lstm'+str(idx))
					self.embedding_words = tf.concat(outputs, 2)
			outputs = tf.split(self.embedding_words, 2, -1)

			with tf.variable_scope('Attention'):
				H = outputs[0] + outputs[1]
				_outputs, _hiddensize = self.attention(H, self.args.attention_size)

			with tf.variable_scope('output'):
				_outputs = tf.reshape(_outputs, [-1, _hiddensize])          # 不加这句会报错
				source = slim.fully_connected(_outputs, self.args.class_num, activation_fn=None)
				prediction = tf.argmax(tf.nn.softmax(source), 1)
			return source, prediction

	def attention(self, inputs, attention_size):
		hiddensize = list(map(int, self.args.hiddensizes.split(',')))[-1]

		# Trainable parameters
		w_omega = tf.Variable(tf.random_normal([hiddensize, attention_size], stddev=0.1))
		b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
		u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

		with tf.name_scope('v'):
			# Applying fully connected layer with non-linear activation to each of the B*T timestamps;
			#  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
			v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

		# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
		vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
		alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

		# Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
		output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
		return output, hiddensize
