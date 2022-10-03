#!/usr/bin/env python3

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import PersianLexicon, collate_fn
from model import Encoder, Decoder
from config import DataConfig, ModelConfig, TrainConfig

import configparser
import argparse
from test_p2g_v2 import P2G_train
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='default.ini')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)
# exp_name = config['GENERAL']['Exp_name']
exp_name = Path(args.config).stem
ep_max = int(config['TRAIN']['Ep_max'])

# data prep
ds = PersianLexicon(
    DataConfig.graphemes_path,
    DataConfig.phonemes_path,
    DataConfig.lexicon_path_train
)
dl = DataLoader(
    ds,
    collate_fn=collate_fn,
    batch_size=TrainConfig.batch_size
)

# models
encoder_model = (
    Encoder(
        ModelConfig.phonemes_size,
        ModelConfig.hidden_size
    )
    .to(TrainConfig.device)
)
decoder_model = (
    Decoder(
        ModelConfig.graphemes_size,
        ModelConfig.hidden_size
    )
    .to(TrainConfig.device)
)

# log
log = SummaryWriter(TrainConfig.log_path)

# loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(
    list(encoder_model.parameters()) + list(decoder_model.parameters()),
    lr=TrainConfig.lr
)

# eval
ds_valid = PersianLexicon(
    DataConfig.graphemes_path,
    DataConfig.phonemes_path,
    DataConfig.lexicon_path_valid
)
ds_test = PersianLexicon(
    DataConfig.graphemes_path,
    DataConfig.phonemes_path,
    DataConfig.lexicon_path_test
)
error_rate_min = float('inf')
p2g = P2G_train(encoder_model, decoder_model, device=TrainConfig.device)

# training loop
counter = 0
for e in range(ep_max):
    print('-' * 20 + f'epoch: {e+1:02d}' + '-' * 20)
    # for g, p in tqdm(dl):
    for g, p in dl:
        g = g.to(TrainConfig.device)
        p = p.to(TrainConfig.device)
        # encode
        enc = encoder_model(p)

        # decoder
        T, N = g.size()
        outputs = []
        hidden = (
            torch.ones(
                1,
                N,
                ModelConfig.hidden_size
            )
            .to(TrainConfig.device)
        )
        for t in range(T - 1):
            out, hidden, _ = decoder_model(
                g[t:t+1],
                enc,
                hidden
            )
            outputs.append(out)
        outputs = torch.cat(outputs)

        # flat Time and Batch, calculate loss
        outputs = outputs.view((T-1) * N, -1)
        g = g[1:]  # trim first phoneme
        g = g.view(-1)
        loss = criterion(outputs, g)

        # updata weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.add_scalar('loss', loss.item(), counter)
        counter += 1

    # save model
    save_dir = f'models/p2g/{DataConfig.language}/{exp_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # torch.save(
    #     encoder_model.state_dict(),
    #     f'{save_dir}/encoder_e{e+1:02d}.pth'
    # )
    # torch.save(
    #     decoder_model.state_dict(),
    #     f'{save_dir}/decoder_e{e+1:02d}.pth'
    # )

    # eval model
    print('starting to eval')

    step = 1
    error_rate = p2g.eval(ds_valid, step=step)
    print(f'ep {e+1}, valid error_rate: {error_rate}, {len(ds_valid)}/{step} words')
    test_error_rate = p2g.eval(ds_test, step=step)
    print(f'ep {e+1}, test error_rate: {test_error_rate}, {len(ds_test)}/{step} words')
    if error_rate < error_rate_min:
        print(f'better than prev best valid error_rate: {error_rate_min}')
        error_rate_min = error_rate
        torch.save(
        encoder_model.state_dict(),
            f'{save_dir}/encoder_best.pth'
        )
        torch.save(
            decoder_model.state_dict(),
            f'{save_dir}/decoder_best.pth'
        )

        torch.save(
            encoder_model.state_dict(),
            f'{save_dir}/encoder_e{e+1:02d}.pth'
        )
        torch.save(
            decoder_model.state_dict(),
            f'{save_dir}/decoder_e{e+1:02d}.pth'
        )
