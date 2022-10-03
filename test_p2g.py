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


def load_model(model_path, model):
    model.load_state_dict(torch.load(
        model_path,
        map_location=lambda storage,
        loc: storage
    ))
    model.to(TestConfig.device)
    model.eval()
    return model


class P2G_eval(object):
    def __init__(self, encoder_model_path, decoder_model_path, device=TestConfig.device):
        # data
        self.ds = PersianLexicon(
            DataConfig.graphemes_path,
            DataConfig.phonemes_path,
            DataConfig.lexicon_path
        )

        # model
        self.encoder_model = Encoder(
            ModelConfig.phonemes_size,
            ModelConfig.hidden_size
        )
        load_model(encoder_model_path, self.encoder_model)

        self.decoder_model = Decoder(
            ModelConfig.graphemes_size,
            ModelConfig.hidden_size
        )
        load_model(decoder_model_path, self.decoder_model)

        # device
        self.device = device
        self.encoder_model = self.encoder_model.to(self.device)
        self.decoder_model = self.decoder_model.to(self.device)


    def __call__(self, word, visualize=False, g=''):
        word = word.split('.')
        x = [0] + [self.ds.p2idx[ch] for ch in word] + [1]
        x = torch.tensor(x).long().unsqueeze(1).to(self.device)
        with torch.no_grad():
            enc = self.encoder_model(x)

        phonemes, att_weights = [], []
        x = torch.zeros(1, 1).long().to(self.device)
        hidden = torch.ones(
            1,
            1,
            ModelConfig.hidden_size
        ).to(self.device)
        t = 0
        t_max = 1.2*len(g) if g else 2.2*len(word)
        while t < t_max:
            with torch.no_grad():
                out, hidden, att_weight = self.decoder_model(
                    x,
                    enc,
                    hidden
                )

            att_weights.append(att_weight.detach().cpu())
            max_index = out[0, 0].argmax()
            x = max_index.unsqueeze(0).unsqueeze(0)
            t += 1

            phonemes.append(self.ds.idx2g[max_index.item()])
            if max_index.item() == 1:
                break

        if visualize:
            att_weights = torch.cat(att_weights).squeeze(1).numpy().T
            y, x = att_weights.shape
            plt.imshow(att_weights, cmap='gray')
            plt.yticks(range(y), ['<sos>'] + list(word) + ['<eos>'])
            plt.xticks(range(x), phonemes)
            plt.savefig(f'attention/{DataConfig.language}/{word}.png')

        return phonemes


    def eval(self, ds, step=100, verbose=False):
        g_lst, g_hyp_lst = [], []
        for i in range(0, len(ds.lexicon), step):
            if i%5000==0: print(i)
            g, p = ds.lexicon[i]
            p = '.'.join(p.split())
            g_hyp = self(p, g=g)
            g_hyp = ''.join(g_hyp[:-1])
            # print(g, p, p_hyp)
            # import pdb; pdb.set_trace()
            g_lst.append(g)
            g_hyp_lst.append(g_hyp)

        error_rate = fastwer.score(g_hyp_lst, g_lst, char_level=True)
        # print(f'WER = {error_rate}, out of {len(ds.lexicon)}/{step} words')
        print('ref', g_lst[::1000])
        print('hyp', g_hyp_lst[::1000])
        if verbose:
            return error_rate, g_hyp_lst, g_lst
        else:
            return error_rate


class P2G_train(P2G_eval):
    def __init__(self, encoder_model, decoder_model, device=TestConfig.device):
        # data
        self.ds = PersianLexicon(
            DataConfig.graphemes_path,
            DataConfig.phonemes_path,
            DataConfig.lexicon_path
        )

        # model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

        # device
        self.device = device
        self.encoder_model = self.encoder_model.to(self.device)
        self.decoder_model = self.decoder_model.to(self.device)


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
    eval_set = args.set

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

    encoder_model_path = f'models/p2g/{DataConfig.language}/{exp_name}/encoder_best.pth'
    decoder_model_path = f'models/p2g/{DataConfig.language}/{exp_name}/decoder_best.pth'

    p2g = P2G_eval(encoder_model_path, decoder_model_path)
    result = p2g(args.word, args.visualize)
    print('.'.join(result))
    error_rate = p2g.eval(ds, step=step)
    print(f'best valid ckpt, {eval_set} error_rate: {error_rate}, {len(ds)}/{step} words')

    
