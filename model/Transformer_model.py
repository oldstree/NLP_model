# -*- coding: utf-8 -*-

# Created by csw on 2019/8/19.

import tensorflow as tf
import tensorflow.contrib.slim as slim


class Transformer(object):
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
			self.embed_pos_words = tf.concat([self.embedding_words, self.embeddingPosition], -1)

		output, prediction = self.forward()

		with tf.name_scope('L2_norm'):
			tv = tf.trainable_variables(scope='transformer')
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
		with tf.variable_scope('transformer'):
			multiHeadAtt = self._multiheadAttention(rawKeys=self.input, queries=self.embed_pos_words,
													keys=self.embed_pos_words)
			embeddedWords = self._feedForward(multiHeadAtt,
											  [self.args.filters,
											   self.args.embedding_size + self.args.sequence_Length])

			outputs = tf.reshape(embeddedWords, [-1, self.args.sequence_Length * (
					self.args.embedding_size + self.args.sequence_Length)])

		with tf.name_scope('dropout'):
			_hDrop = tf.nn.dropout(outputs, keep_prob=self.KeepProb)

		with tf.variable_scope('output'):
			source = slim.fully_connected(_hDrop, self.args.class_num, activation_fn=None)
			prediction = tf.argmax(tf.nn.softmax(source), 1)
		return source, prediction

	def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope="MultiHeadAttention"):
		numHeads = self.args.numHeads

		if numUnits is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
			numUnits = queries.get_shape().as_list()[-1]

		# tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
		# 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
		# Q, K, V的维度都是[batch_size, sequence_length, embedding_size]
		with tf.variable_scope(scope):
			Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
			K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
			V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

		# 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
		# Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
		Q_ = tf.concat(tf.split(Q, numHeads, axis=-1), axis=0)
		K_ = tf.concat(tf.split(K, numHeads, axis=-1), axis=0)
		V_ = tf.concat(tf.split(V, numHeads, axis=-1), axis=0)

		# 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
		similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

		# 对计算的点积进行缩放处理，除以向量长度的根号值
		scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)

		# 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
		# 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
		# 之前，我们需要将其mask为0。虽然在queries中也存在这样的填充词，但原则上模型的结果之和输入有关，而且在self-Attention中
		# queryies = keys，因此只要一方为0，计算出的权重就为0。
		# 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

		# 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
		keyMasks = tf.tile(rawKeys, [numHeads, 1])

		# 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
		keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

		# tf.ones_like生成元素全为1，维度和scaledSimilary相同的tensor, 然后得到负无穷大的值
		paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))

		# tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
		# 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
		maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings,
								  scaledSimilary)  # 维度[batch_size * numHeads, queries_len, key_len]

		# 在计算当前的词时，只考虑上文，不考虑下文，出现在Transformer Decoder中。在文本分类时，可以只用Transformer Encoder。
		# Decoder是生成模型，主要用在语言生成中
		if causality:
			diagVals = tf.ones_like(maskedSimilary[0, :, :])  # [queries_len, keys_len]
			tril = tf.contrib.linalg.LinearOperatorTriL(diagVals).to_dense()  # [queries_len, keys_len]
			masks = tf.tile(tf.expand_dims(tril, 0),
							[tf.shape(maskedSimilary)[0], 1, 1])  # [batch_size * numHeads, queries_len, keys_len]

			paddings = tf.ones_like(masks) * (-2 ** (32 + 1))
			maskedSimilary = tf.where(tf.equal(masks, 0), paddings,
									  maskedSimilary)  # [batch_size * numHeads, queries_len, keys_len]

		# 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len]
		weights = tf.nn.softmax(maskedSimilary)

		# 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
		outputs = tf.matmul(weights, V_)

		# 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
		outputs = tf.concat(tf.split(outputs, numHeads, axis=0), axis=2)

		outputs = tf.nn.dropout(outputs, keep_prob=0.9)

		# 对每个subLayers建立残差连接，即H(x) = F(x) + x
		outputs += queries
		# normalization 层
		outputs = self._layerNormalization(outputs)
		return outputs

	def _feedForward(self, inputs, filters, scope="feedForward"):
		# 在这里的前向传播采用卷积神经网络

		# 内层
		with tf.variable_scope(scope + '1'):
			params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
					  "activation": tf.nn.relu, "use_bias": True}
			outputs = tf.layers.conv1d(**params)

		# 外层
		with tf.variable_scope(scope + '2'):
			params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
					  "activation": None, "use_bias": True}

			# 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
			# 维度[batch_size, sequence_length, embedding_size]
			outputs = tf.layers.conv1d(**params)

		# 残差连接
		outputs += inputs

		# 归一化处理
		outputs = self._layerNormalization(outputs)

		return outputs

	def _layerNormalization(self, inputs, scope="layerNorm"):
		with tf.name_scope(scope):
			# LayerNorm层和BN层有所不同
			epsilon = 1e-8
			inputsShape = inputs.get_shape()  # [batch_size, sequence_length, embedding_size]
			paramsShape = inputsShape[-1:]
			# LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
			# mean, variance的维度都是[batch_size, sequence_len, 1]
			mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
			beta = tf.Variable(tf.zeros(paramsShape))
			gamma = tf.Variable(tf.ones(paramsShape))
			normalized = (inputs - mean) / ((variance + epsilon) ** .5)
			outputs = gamma * normalized + beta
		return outputs
