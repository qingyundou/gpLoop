#!/usr/bin/env python3

import os
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data import PersianLexicon
from model import Encoder, Decoder
from config import DataConfig, ModelConfig, TestConfig

import configparser
import fastwer
from tqdm import tqdm
from pathlib import Path

import json
import random


if __name__ == '__main__':
    # get word
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', type=str, default='پایتون')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--config', type=str, default='default.ini')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    # exp_name = Path(args.config).stem
    # ep_max = int(config['TRAIN']['Ep_max'])
    # ep = config['TEST']['Ep']

    ds = PersianLexicon(
            DataConfig.graphemes_path,
            DataConfig.phonemes_path,
            DataConfig.lexicon_path
        )

    lexicon_test, lexicon_valid, lexicon_train = [], [], []
    step = 20
    # import pdb; pdb.set_trace()
    for i in range(0, len(ds.lexicon)):
        g, p = ds.lexicon[i]
        # if i%10000==0: print(i, g, p)

        if i%step==0:
            lexicon_valid.append([g, p])
        else:
            lexicon_train.append([g, p])

    random.shuffle(lexicon_train)
    lexicon_train, lexicon_test = lexicon_train[len(lexicon_valid):], lexicon_train[:len(lexicon_valid)]

    with open(DataConfig.lexicon_path_test, "w") as outfile:
        json.dump(lexicon_test, outfile)

    with open(DataConfig.lexicon_path_valid, "w") as outfile:
        json.dump(lexicon_valid, outfile)

    with open(DataConfig.lexicon_path_train, "w") as outfile:
        json.dump(lexicon_train, outfile)

    with open(DataConfig.lexicon_path_train) as fd:
        lexicon_train = json.load(fd)
    with open(DataConfig.lexicon_path_valid) as fd:
        lexicon_valid = json.load(fd)
    with open(DataConfig.lexicon_path_test) as fd:
        lexicon_test = json.load(fd)

    print('train', 'valid', 'test')
    print(len(lexicon_train), len(lexicon_valid), len(lexicon_test))
    print(lexicon_train[:20])
    print(lexicon_valid[:5])
    print(lexicon_test[:5])
    # for i in range(0, 40):
    #     g, p = ds.lexicon[i]
    #     if [g,p] in lexicon_valid:
    #         print('valid', i, g, p)
    #     elif [g,p] in lexicon_test:
    #         print('test', i, g, p)
    #     else:
    #         assert [g,p] in lexicon_train




