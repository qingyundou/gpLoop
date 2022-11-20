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
import re

from test_g2p import G2P_eval
from test_p2g import P2G_eval


def load_model(model_path, model):
    model.load_state_dict(torch.load(
        model_path,
        map_location=lambda storage,
        loc: storage
    ))
    model.to(TestConfig.device)
    model.eval()
    return model


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
        # sentence = sentence.split()
        # sentence_accent = [ self.add_accent_g(w, accent=accent) for w in sentence ]

        sentence_accent = re.findall(r"[\w']+|[.,!?;()-]", sentence)
        for i, w in enumerate(sentence_accent):
            if ord('a')<=ord(w[0])<=ord('z') or ord('A')<=ord(w[0])<=ord('z'):
                sentence_accent[i] = self.add_accent_g(w, accent=accent)
        
        sentence_accent = ' '.join(sentence_accent)
        sentence_accent = re.sub(r'\s+([?,:;.!")-])', r'\1', sentence_accent)
        sentence_accent = re.sub(r'([(-])\s+', r'\1', sentence_accent)

        return sentence_accent

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
        # p = self.g2p(g)
        # p = '.'.join(p[:-1])

        p_accent, info_dict = self.add_accent_p(p, accent)
        if not info_dict['hacked']:
            return g.lower()
        g_accent = self.p2g(p_accent)
        g_accent = ''.join(g_accent[:-1])

        if accent=='fr':
            g_accent = self.postproc_fr(g_accent, info_dict)
        # if accent=='de': print(p, p_accent)
        return g_accent

    def postproc_fr(self, g, info_dict):
        if info_dict['hacked_ch_sh']:
            g = g.replace('SCH', 'SH')
        return g



    def add_accent_p(self, p, accent=''):
        if accent=='':
            return p, {'hacked': False}
        if accent=='fr':
            return self.add_accent_p_fr(p)
        elif accent=='de':
            return self.add_accent_p_de(p)
        elif accent=='jp':
            return self.add_accent_p_jp(p)
        elif accent=='cn':
            return self.add_accent_p_cn(p)

    def add_accent_p_fr(self, p):
        info_dict = {'hacked': False, 'hacked_ch_sh': False}
        # confusions = {'HH': '', 'CH': 'SH', 'IH':'IY', 'UW':'AA'}
        confusions = {'HH': '', 'CH': 'SH'}

        hacked_ch_sh = False
        p = p.split('.')
        for i, ph in enumerate(p):
            if ph in confusions:
                p[i] = confusions[ph]
                info_dict['hacked'] = True
                if ph=='CH': info_dict['hacked_ch_sh'] = True
        p = [ ph for ph in p if ph!='']

        return '.'.join(p), info_dict

    def add_accent_p_de(self, p):
        info_dict = {'hacked': False}
        confusions = {'W': 'V', 'CH': 'SH'}

        p = p.split('.')
        for i, ph in enumerate(p):
            if ph in confusions:
                p[i] = confusions[ph]
                info_dict['hacked'] = True
                if i>0 and p[i]=='V' and p[i-1]=='HH':
                    p[i-1] = ''
        p = [ ph for ph in p if ph!='']

        return '.'.join(p), info_dict

    def add_accent_p_jp(self, p):
        info_dict = {'hacked': False}
        confusions = {'R': 'L', 'V': 'B'}

        p = p.split('.')
        for i, ph in enumerate(p):
            if ph in confusions:
                p[i] = confusions[ph]
                info_dict['hacked'] = True
        p = [ ph for ph in p if ph!='']

        return '.'.join(p), info_dict

    def add_accent_p_cn(self, p):
        info_dict = {'hacked': False}
        confusions = {} # {'N': 'L', 'L': 'N'}

        p = p.split('.')
        for i, ph in enumerate(p):
            if ph in confusions:
                p[i] = confusions[ph]
                info_dict['hacked'] = True
        p = [ ph for ph in p if ph!='']

        if p[-1] in ['T', 'D', 'K', 'G']:
            p.append('AH')
            info_dict['hacked'] = True
        if p[-1] in ['B', 'P', 'M', 'F']:
            p.append('UH')
            info_dict['hacked'] = True
        return '.'.join(p), info_dict

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


class AccentAdder_Context(AccentAdder):
    def add_accent_g(self, g_raw, accent=''):
        if accent=='':
            return g

        g = g_raw.upper()    
        if g in self.dict:
            p = self.dict[g]
            p = p.split()
        else:
            p = self.g2p(g)
            p = p[:-1]

        if accent=='fr':
            return self.add_accent_pg_fr(p, g_raw)
        elif accent=='de':
            return self.add_accent_pg_de(p, g_raw)
        elif accent=='jp':
            return self.add_accent_pg_jp(p, g_raw)
        elif accent=='cn':
            return self.add_accent_pg_cn(p, g_raw)


    def add_accent_pg_fr(self, p, g_raw):
        info_dict = {'hacked': False, 'hacked_ch_sh': False}
        # confusions = {'HH': '', 'IH':'IY', 'UW':'AA'}
        confusions = {'HH': ''}

        highfreq_words = {'HAD':'ADD', 'WHO': 'WOO', 'WHOSE': 'WOOS'}

        g = g_raw.upper()
        if g in highfreq_words: return highfreq_words[g]

        if 'CH' in g:
            confusions['CH'] = 'SH'

        if (p[0]=='HH' and p[1]=='W') or (p[0]=='HH' and g[:2]=='WH'):
            del p[0]

        for i, ph in enumerate(p):
            if ph in confusions:
                p[i] = confusions[ph]
                info_dict['hacked'] = True
                if ph=='CH': info_dict['hacked_ch_sh'] = True
        p = [ ph for ph in p if ph!='']
        p_accent = '.'.join(p)

        if not info_dict['hacked']:
            return g_raw

        g_accent = self.p2g(p_accent)
        g_accent = ''.join(g_accent[:-1])

        if info_dict['hacked_ch_sh']:
            g_accent = g_accent.replace('SCH', 'SH')

        # if g=='WHO':
        #     import pdb; pdb.set_trace()

        return g_accent


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

    # eps = [10, 15, 20, 25, 30, 35, 40, 46, 48, 55]
    # eps = range(10, 30, 3)
    eps = [20]
    for i in eps:
        ep_p2g = f'e{i:02d}'
        print(ep_p2g)
    
        encoder_model_path = f'models/p2g/{DataConfig.language}/{exp_name}/encoder_{ep_p2g}.pth'
        decoder_model_path = f'models/p2g/{DataConfig.language}/{exp_name}/decoder_{ep_p2g}.pth'
        p2g = P2G_eval(encoder_model_path, decoder_model_path)

        acc_adder = AccentAdder_Context(g2p, p2g)

        sentences_path = '/home/qd212/models/WaveRNN/test_sentences/sentences_espnet_250_proc.txt'
        with open(sentences_path, 'r') as f:
            sentences = f.readlines()
        
        accent = 'fr'
        cnt = 0
        s_accent_list = []
        for s in sentences:
            cnt += 1
            # if cnt>3: break
            s_accent = acc_adder.hack_sentence(s, accent=accent)
            s_accent = s_accent + '\n'
            # print(f'{s}\n{s_accent}')
            s_accent_list.append(s_accent)

        sentences_accent_path = sentences_path.replace('.txt', f'_{accent}.txt')
        with open(sentences_accent_path, 'w') as f:
            f.writelines(s_accent_list)

        # data = {
        # 'fr': ['hero humour have house', 'chadder cheese rich reach research search', 'loot food mood goose', 'hit lit liquid slim'],
        # 'de':['why what when where ware', 'chadder cheese rich reach research search']
        # }

        data = {
        'fr':['The humble hero had cheddar cheese for lunch', 'Who what when where why one future nature lunch'],
        # 'de':['Why are rich people so worried about what to ware and when'],
        # 'jp':['The rules are all about vegetables'],
        # 'cn':['The lookbook is full of bright colors']
        }
        for accent, sentences in data.items():
            # print(accent)
            # s = 'What are the huge vegetables in the research lab'
            # s_accent = acc_adder.hack_sentence(s, accent=accent)
            # print(f'{s}\n{s_accent}')

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
    
