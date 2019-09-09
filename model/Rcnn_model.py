# -*- coding: utf-8 -*-

# Created by csw on 2019/8/20.

import tensorflow as tf


class Rcnn(object):
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
			self.embedding_words_ = self.embedding_words

		output, prediction = self.forward()

		with tf.name_scope('L2_norm'):
			tv = tf.trainable_variables(scope='Rcnn')
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
		with tf.variable_scope('Rcnn'):
			for idx, hidden_size in enumerate(list(map(int, self.args.hidden_sizes.split()))):
				with tf.variable_scope('Bi-LSTM' + str(idx)):
					lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
						tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
						output_keep_prob=self.KeepProb)

					lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
						tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
						output_keep_prob=self.KeepProb)
					output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
																self.embedding_words,
																dtype=tf.float32,
																scope='bilstm' + str(idx))
					self.embedding_words = tf.concat(output, 2)

			fw_output, bw_output = tf.split(self.embedding_words, 2, -1)

			with tf.name_scope("context"):
				shape = [tf.shape(fw_output)[0], 1, tf.shape(fw_output)[2]]
				context_left = tf.concat([tf.zeros(shape), fw_output[:, :-1]], axis=1, name="context_left")
				context_right = tf.concat([bw_output[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

			with tf.name_scope('wordRepresentation'):
				word_representation = tf.concat([context_left, self.embedding_words_, context_right], axis=2)

			with tf.variable_scope('text_representation'):
				text_representation = tf.layers.dense(word_representation, self.args.output_size,
													  activation=tf.tanh)
			_output = tf.reduce_mean(text_representation, axis=1)

			with tf.variable_scope('output'):
				source = tf.layers.dense(_output, self.args.class_num)
				prediction = tf.argmax(tf.nn.softmax(source), 1)
			return source, prediction
