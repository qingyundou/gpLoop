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

from test_g2p import G2P_eval
from test_p2g_v2 import P2G_eval


def load_model(model_path, model):
    model.load_state_dict(torch.load(
        model_path,
        map_location=lambda storage,
        loc: storage
    ))
    model.to(TestConfig.device)
    model.eval()
    return model


confusions_fr = {'HH': '', 'CH': 'SH', 'IH':'IY', 'UW':'AA'}
confusions_de = {'W': 'V', 'CH': 'SH'}
confusions_jp = {'R': 'L', 'V': 'B'}
# confusions_cn = {'N': 'L', 'L': 'N'}
confusions_cn = {}

class AccentAdder(object):
    def __init__(self, g2p, p2g):
        self.g2p = g2p
        self.p2g = p2g

        self.ds = PersianLexicon(
            DataConfig.graphemes_path,
            DataConfig.phonemes_path,
            DataConfig.lexicon_path
        )

        self.dict = {g:p for g,p in self.ds.lexicon}

    def hack_sentence(self, sentence, accent=''):
        sentence_accent = [ self.add_accent_g(w, accent=accent) for w in sentence.split() ]
        return ' '.join(sentence_accent)

    def add_accent_g(self, g, accent=''):
        if accent=='':
            return g

        g = g.upper()    
        if g in self.dict:
            p = self.dict[g]
            p = '.'.join(p.split())
        else:
            p = self.g2p(g)
            p = '.'.join(p[:-1])

        p_accent, hacked = self.add_accent_p(p, accent)
        if not hacked:
            return g.lower()
        g_accent = self.p2g(p_accent)
        g_accent = ''.join(g_accent[:-1])
        return g_accent

    def add_accent_p(self, p, accent=''):
        hacked = False
        if accent=='':
            return p, hacked
        if accent=='fr':
            confusions = confusions_fr
        elif accent=='de':
            confusions = confusions_de
        elif accent=='jp':
            confusions = confusions_jp
        elif accent=='cn':
            confusions = confusions_cn

        p = p.split('.')
        for i, ph in enumerate(p):
            if ph in confusions:
                p[i] = confusions[ph]
                hacked = True
        p = [ ph for ph in p if ph!='']

        if accent=='cn':
            if p[-1] in ['T', 'D', 'K', 'G']:
                p.append('AH')
                hacked = True
            if p[-1] in ['B', 'P', 'M', 'F']:
                p.append('UH')
                hacked = True
        return '.'.join(p), hacked

    def eval(self, ds):
        step = 5000
        g_lst, p_lst = [g for g,p in ds.lexicon], [p for g,p in ds.lexicon]
        g_hyp_lst, p_hyp_lst = [], []
        for i in range(0, len(ds.lexicon)):
            g, p = ds.lexicon[i]
            g_accent = self.add_accent_g(g, accent=accent)
            p_accent = self.g2p(g_accent)
            p_accent = ' '.join(p_accent[:-1])
            g_hyp_lst.append(g_accent)
            p_hyp_lst.append(p_accent)
            if i%step==0:
                print('original', i, g, p)
                print('accented', g_accent, p_accent)

        error_rate_g2p2g = fastwer.score(g_hyp_lst, g_lst, char_level=True)
        print(f'best valid ckpt, {eval_set} set grapheme error_rate: {error_rate_g2p2g}, {len(ds.lexicon)} words')

        error_rate_g2p2g2p = fastwer.score(p_hyp_lst, p_lst, char_level=False)
        print(f'best valid ckpt, {eval_set} set phoneme error_rate: {error_rate_g2p2g2p}, {len(ds.lexicon)} words')


def add_accent_g(g, g2p, p2g, accent=''):
    p = g2p(g)
    p = '.'.join(p[:-1])
    # import pdb; pdb.set_trace()
    p_accent = add_accent_p(p, accent)
    g_accent = p2g(p_accent)
    return ''.join(g_accent[:-1])

def add_accent_p(p, accent=''):
    if not accent:
        return p
    if accent=='fr':
        confusions = confusions_fr
    elif accent=='de':
        confusions = confusions_de
    elif accent=='jp':
        confusions = confusions_jp

    p = p.split('.')
    for i, ph in enumerate(p):
        if ph in confusions:
            p[i] = confusions[ph]
    p = [ ph for ph in p if ph!='']

    if accent=='cn':
        if p[-1] in ['T', 'D', 'K', 'G']:
            p.append('AH')
        if p[-1] in ['B', 'P', 'M', 'F']:
            p.append('UH')
    return '.'.join(p)


if __name__ == '__main__':
    # get word
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', type=str, default='پایتون')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--config', type=str, default='default.ini')
    parser.add_argument('--set', type=str, default='test')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    # exp_name = config['GENERAL']['Exp_name']
    exp_name = Path(args.config).stem
    ep_max = int(config['TRAIN']['Ep_max'])
    ep = config['TEST']['Ep']
    ep_g2p = config['ACCENT']['Ep_g2p']
    ep_p2g = config['ACCENT']['Ep_p2g']
    eval_set = args.set

    eval_set = 'valid'
    accent = ''

    if eval_set == 'train':
        lexicon_path = DataConfig.lexicon_path
        step = 20
    elif eval_set == 'valid':
        lexicon_path = DataConfig.lexicon_path_valid
        step = 1
    elif eval_set == 'test':
        lexicon_path = DataConfig.lexicon_path_test
        step = 1

    ds = PersianLexicon(
        DataConfig.graphemes_path,
        DataConfig.phonemes_path,
        lexicon_path
    )

    encoder_model_path = f'models/g2p/{DataConfig.language}/{exp_name}/encoder_{ep_g2p}.pth'
    decoder_model_path = f'models/g2p/{DataConfig.language}/{exp_name}/decoder_{ep_g2p}.pth'
    g2p = G2P_eval(encoder_model_path, decoder_model_path)
    
    encoder_model_path = f'models/p2g/{DataConfig.language}/{exp_name}/encoder_{ep_p2g}.pth'
    decoder_model_path = f'models/p2g/{DataConfig.language}/{exp_name}/decoder_{ep_p2g}.pth'
    p2g = P2G_eval(encoder_model_path, decoder_model_path)

    acc_adder = AccentAdder(g2p, p2g)
    data = {
    'fr': ['The hero stopped the looting at the chip factory'],
    'de':['Why are you so worried about the chadder cheese'],
    'jp':['The rules are all about vegetables'],
    'cn':['The lookbook is full of bright colors']
    }
    for accent, sentences in data.items():
        print(accent)
        s = 'What are the huge vegetables in the research lab'
        s_accent = acc_adder.hack_sentence(s, accent=accent)
        print(f'{s}\n{s_accent}')

        for s in sentences:
            s_accent = acc_adder.hack_sentence(s, accent=accent)
            print(f'{s}\n{s_accent}')

    # print(len(acc_adder.dict), acc_adder.dict['MOLAND'])
    import pdb; pdb.set_trace()

    acc_adder.eval(ds)


    # words = ['HERO', 'CHIP', 'SHIP', 'WHAT', 'VOLT', 'LOOT', 'LOT', 'SLIM', 'SLEEP']
    # for w in words:
    #     print(w, g2p(w))


    # step = 1000
    # g_lst, p_lst = [], []
    # g_hyp_lst, p_hyp_lst = [g for g,p in ds.lexicon], [p for g,p in ds.lexicon]
    # for i in range(0, len(ds.lexicon)):
    #     g, p = ds.lexicon[i]
    #     g_accent = add_accent_g(g, g2p, p2g, accent=accent)
    #     p_accent = g2p(g_accent)
    #     p_accent = ' '.join(p_accent[:-1])
    #     g_lst.append(g_accent)
    #     p_lst.append(p_accent)
    #     if i%step==0:
    #         print(i, g, p)
    #         print(g_accent, p_accent)

    # error_rate = fastwer.score(g_hyp_lst, g_lst, char_level=True)
    # print(f'best valid ckpt, {eval_set} set grapheme error_rate: {error_rate}, {len(ds.lexicon)} words')

    # error_rate = fastwer.score(p_hyp_lst, p_lst, char_level=False)
    # print(f'best valid ckpt, {eval_set} set phoneme error_rate: {error_rate}, {len(ds.lexicon)} words')
    
