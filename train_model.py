# -*- coding: utf-8 -*-

# Created by csw on 2019/8/18.

import os
import shutil
import argparse
import datetime

import numpy as np
import tensorflow as tf
from model import TextCNN_model
from model import Transformer_model
from model import BiLSTMAttention_model
from model import Rcnn_model


def get_Batch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    numBatches = len(x) // batchSize
    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="int64")
        yield batchX, batchY


def fixedPositionEmbedding(batchSize, sequenceLen):
    embeddedPosition = []
    for batch in range(batchSize):
        x = []
        for step in range(sequenceLen):
            a = np.zeros(sequenceLen)
            a[step] = 1
            x.append(a)
        embeddedPosition.append(x)

    return np.array(embeddedPosition, dtype="float32")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mnist test")
    # learning
    parser.add_argument('-lr', type=float, default=0.0005, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size for training [default: 64]')

    # result
    parser.add_argument('-result_dir', type=str, default='./snapshot', help='where to save the snapshot')

    # dataset
    parser.add_argument('-dataset_path', type=str, default='./data/Index_data', help='the path of dataset')

    # model
    parser.add_argument('-l2_lambda', type=float, default=0.1)
    parser.add_argument('-model_class', type=str, default='TextCnn',
                        help='selection of model types [default: TextCnn, Transformer, BiLSTMAttention, Rcnn]')
    parser.add_argument('-sequence_Length', type=int, default=200, help='the length of sentence')
    parser.add_argument('-vocab_size', type=int, default=31983, help='the size of vocabulary')
    parser.add_argument('-embedding_size', type=int, default=128, help='the size of embedding')
    # model-cnn
    parser.add_argument('-numFilters', type=int, default=128, help='the number of convolution2d filter')
    parser.add_argument('-filterSizes', type=str, default='2,3,4,5', help='the size of convolution2d filter')
    # model-Transformer
    parser.add_argument('-filters', type=int, default=128, help='number of convolution1d filter')
    parser.add_argument('-numHeads', type=int, default=8, help='number of heads for Attention')
    # model-BiLSTMAttention
    parser.add_argument('-hiddensizes', type=str, default='256, 128')
    parser.add_argument('-attention_size', type=int, default=128)
    # model-Rcnn
    parser.add_argument('-hidden_sizes', type=str, default='256')
    parser.add_argument('-output_size', type=int, default=128)

    # option
    args = parser.parse_args()
    args.class_num = 2

    # load the dataset
    trainReviews = np.load(os.path.join(args.dataset_path, 'trainReviews.npy'))
    trainLabels = np.load(os.path.join(args.dataset_path, 'trainLabels.npy'))
    evalReviews = np.load(os.path.join(args.dataset_path, 'evalReviews.npy'))
    evalLabels = np.load(os.path.join(args.dataset_path, 'evalLabels.npy'))

    mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_dir = os.path.join(args.result_dir, mulu + '_' + args.model_class)
    os.makedirs(result_dir)

    # save parameters
    with open(os.path.join(result_dir, 'Parameters.txt'), "a") as file:
        file.write("the parameters of this run:\n")
        for attr, value in sorted(args.__dict__.items()):
            file.write("\t{}={}\n".format(attr.upper(), value))

    with tf.Graph().as_default():
        if args.model_class == 'TextCnn':
            model = TextCNN_model.TextCNN(args)
            shutil.copy("./model/TextCNN_model.py", result_dir + "/TextCNN_model.py")
        elif args.model_class == 'Transformer':
            model = Transformer_model.Transformer(args)
            shutil.copy("./model/Transformer_model.py", result_dir + "/Transformer_model.py")
        elif args.model_class == 'BiLSTMAttention':
            model = BiLSTMAttention_model.BiLSTMAttention(args)
            shutil.copy("./model/BiLSTMAttention_model.py", result_dir + '/BiLSTMAttention_model.py')
        elif args.model_class == 'Rcnn':
            model = Rcnn_model.Rcnn(args)
            shutil.copy("./model/Rcnn_model.py", result_dir + '/Rcnn_model.py')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            train_summary_writer = tf.summary.FileWriter(result_dir + '/train', sess.graph)
            dev_summary_writer = tf.summary.FileWriter(result_dir + '/dev')

            for step in range(args.epochs + 1):
                for data, label in get_Batch(trainReviews, trainLabels, args.batch_size):
                    batch_size = len(data)
                    embedPosition = fixedPositionEmbedding(batch_size, args.sequence_Length)
                    feed_dict = {model.input: data,
                                 model.label: label,
                                 model.embeddingPosition: embedPosition,
                                 model.KeepProb: 0.5}
                    _, t_losses, t_accuracy, train_summary = sess.run([model.optimizer,
                                                                       model.l2_norm,
                                                                       model.accuracy,
                                                                       model.merged],
                                                                      feed_dict=feed_dict)
                    train_summary_writer.add_summary(train_summary, step)
                print('step:{:>3d}, t_losses:{:.4f}, t_accuracy:{:.4f}'.format(step, t_losses, t_accuracy))

                if step % 10 == 0:
                    batch_size = len(evalReviews[:100])
                    embedPosition = fixedPositionEmbedding(batch_size, args.sequence_Length)
                    feed_dict = {model.input: evalReviews[:100],
                                 model.label: evalLabels[:100],
                                 model.embeddingPosition: embedPosition,
                                 model.KeepProb: 1.0}
                    d_losses, d_accuracy, dev_summary = sess.run([model.losses,
                                                                  model.accuracy,
                                                                  model.merged],
                                                                 feed_dict=feed_dict)
                    dev_summary_writer.add_summary(dev_summary)
                    print('step:{:>3d}, d_losses:{:.4f}, d_accuracy:{:.4f}'.format(step, d_losses, d_accuracy))
            train_summary_writer.close()
