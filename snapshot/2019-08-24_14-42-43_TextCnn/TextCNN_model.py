# -*- coding: utf-8 -*-

# Created by csw on 2019/8/18.

import tensorflow as tf
import tensorflow.contrib.slim as slim


class TextCNN(object):
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
			self.embedding_words_Expand = tf.expand_dims(self.embedding_words, -1)

		output, prediction = self.text_cnn()

		with tf.name_scope('L2_norm'):
			tv = tf.trainable_variables(scope='cnn')
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

	def text_cnn(self):
		pooledOutputs = []
		with tf.variable_scope('cnn'):
			for i, filterSize in enumerate(list(map(int, self.args.filterSizes.split(',')))):
				with tf.variable_scope("conv-maxpool-%s" % filterSize):
					conv = slim.conv2d(self.embedding_words_Expand, self.args.numFilters,
									   [filterSize, self.args.embedding_size],
									   padding='VALID')
					pooled = slim.max_pool2d(conv, [self.args.sequence_Length - filterSize + 1, 1], stride=1)
					pooledOutputs.append(pooled)

			# 池化后的维度不变，按照最后的维度channel来concat
			_hPool = tf.concat(pooledOutputs, 3)
			_hPoolFlat = tf.layers.flatten(_hPool)
			# dropout
			with tf.name_scope("dropout"):
				_hDrop = tf.nn.dropout(_hPoolFlat, self.KeepProb)

			with tf.variable_scope('output'):
				source = slim.fully_connected(_hDrop, self.args.class_num, activation_fn=None)
				prediction = tf.argmax(tf.nn.softmax(source), 1)
			return source, prediction
