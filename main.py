# -*- coding: utf-8 -*-

# Created by csw on 2019/8/20.

import os

print("train model:")
os.system('CUDA_VISIBLE_DEVICES=0 python -u train_model.py -model_class TextCnn \
                                                           > ./log/log1.txt &')

os.system('CUDA_VISIBLE_DEVICES=0 python -u train_model.py -model_class Transformer \
                                                           > ./log/log2.txt &')

os.system('CUDA_VISIBLE_DEVICES=1 python -u train_model.py -model_class BiLSTMAttention \
                                                           > ./log/log3.txt &')

os.system('CUDA_VISIBLE_DEVICES=1 python -u train_model.py -model_class Rcnn \
                                                           > ./log/log4.txt &')
