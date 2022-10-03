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


class G2P(object):
    def __init__(self):
        # data
        self.ds = PersianLexicon(
            DataConfig.graphemes_path,
            DataConfig.phonemes_path,
            DataConfig.lexicon_path
        )

        # model
        self.encoder_model = Encoder(
            ModelConfig.graphemes_size,
            ModelConfig.hidden_size
        )
        load_model(TestConfig.encoder_model_path, self.encoder_model)

        self.decoder_model = Decoder(
            ModelConfig.phonemes_size,
            ModelConfig.hidden_size
        )
        load_model(TestConfig.decoder_model_path, self.decoder_model)

    def __call__(self, word, visualize=False):
        x = [0] + [self.ds.g2idx[ch] for ch in word] + [1]
        x = torch.tensor(x).long().unsqueeze(1)
        with torch.no_grad():
            enc = self.encoder_model(x)

        phonemes, att_weights = [], []
        x = torch.zeros(1, 1).long().to(TestConfig.device)
        hidden = torch.ones(
            1,
            1,
            ModelConfig.hidden_size
        ).to(TestConfig.device)
        t = 0
        while True:
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

            phonemes.append(self.ds.idx2p[max_index.item()])
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

class G2P_eval(G2P):
    def __init__(self, encoder_model_path, decoder_model_path, device=TestConfig.device):
        # data
        self.ds = PersianLexicon(
            DataConfig.graphemes_path,
            DataConfig.phonemes_path,
            DataConfig.lexicon_path
        )

        # model
        self.encoder_model = Encoder(
            ModelConfig.graphemes_size,
            ModelConfig.hidden_size
        )
        load_model(encoder_model_path, self.encoder_model)

        self.decoder_model = Decoder(
            ModelConfig.phonemes_size,
            ModelConfig.hidden_size
        )
        load_model(decoder_model_path, self.decoder_model)

        # device
        self.device = device
        self.encoder_model = self.encoder_model.to(self.device)
        self.decoder_model = self.decoder_model.to(self.device)


    def __call__(self, word, visualize=False, p=''):
        x = [0] + [self.ds.g2idx[ch] for ch in word] + [1]
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
        t_max = 1.2*len(p.split()) if p else 1.2*len(word)
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

            phonemes.append(self.ds.idx2p[max_index.item()])
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
        p_lst, p_hyp_lst = [], []
        for i in range(0, len(ds.lexicon), step):
            if i%5000==0: print(i)
            g, p = ds.lexicon[i]
            p_hyp = self(g, p=p)
            p_hyp = ' '.join(p_hyp[:-1])
            # print(g, p, p_hyp)
            # import pdb; pdb.set_trace()
            p_lst.append(p)
            p_hyp_lst.append(p_hyp)

        error_rate = fastwer.score(p_hyp_lst, p_lst, char_level=False)
        # print(f'WER = {error_rate}, out of {len(ds.lexicon)}/{step} words')
        print('ref', p_lst[::1000])
        print('hyp', p_hyp_lst[::1000])
        if verbose:
            return error_rate, p_hyp_lst, p_lst
        else:
            return error_rate


class G2P_train(G2P_eval):
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

    # eval_set = 'valid'

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

    encoder_model_path = f'models/g2p/{DataConfig.language}/{exp_name}/encoder_best.pth'
    decoder_model_path = f'models/g2p/{DataConfig.language}/{exp_name}/decoder_best.pth'

    g2p = G2P_eval(encoder_model_path, decoder_model_path)
    result = g2p(args.word, args.visualize)
    print('.'.join(result))
    error_rate = g2p.eval(ds, step=step)
    print(f'best valid ckpt, {eval_set} error_rate: {error_rate}, {len(g2p.ds)}/{step} words')

    # for ep in range(1, 3):
    #     if ep<10: ep = f'0{ep}'

    #     encoder_model_path = f'models/g2p/{DataConfig.language}/{exp_name}/encoder_e{ep}.pth'
    #     decoder_model_path = f'models/g2p/{DataConfig.language}/{exp_name}/decoder_e{ep}.pth'

    #     g2p = G2P()
    #     result = g2p(args.word, args.visualize)
    #     print('.'.join(result))

    #     step = 20
    #     g2p = G2P_train(g2p.encoder_model, g2p.decoder_model)
    #     error_rate = g2p.eval(ds, step=step)
    #     print(f'ep {ep}, error_rate: {error_rate}, {len(g2p.ds)}/{step} words')

    
