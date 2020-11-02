import torch
import pandas as pd
import numpy as np
import argparse
import time
import datetime
import os
import hiddenlayer as hl
import random

from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from DataLoader import *

class electra:
    def __init__(self, path):
        self.parser = argparse.ArgumentParser()

    def parse(self):
        self.parser.add_argument('--epoch', type=int, default=5)
        self.parser.add_argument('--batch', type=int, default=32)
        self.parser.add_argument('--duration', type=int, default=50)
        self.parser.add_argument('--lr', type=float, default=6e-6)
        self.parser.add_argument('--n_cpu', type=int, default=4)
        self.parser.add_argument('--eps', type=float, default=1e-8)
        self.opt = self.parser.parse_args()

        return

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))

        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.parse()
        print(self.opt)

        train_data = Data('.', 'train')
        test_data = Data('.', 'test')

        model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator',num_labels=2)
        model.device

        batch_size = self.opt.batch
        optimizer = AdamW(model.parameters(), lr=self.opt.lr, eps=self.opt.eps)
        epoch = self.opt.epoch
        t_data, t_sampler, v_data, v_sampler = train_data.split_data()
        train_data_loader = DataLoader(t_data, sampler=t_sampler, batch_size=self.opt.batch)
        validation_dataloader = DataLoader(v_data, sampler=v_sampler, batch_size=self.opt.batch)

        total_steps = len(train_data_loader) * self.opt.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)





if __name__ == '__main__':
    electra_model = electra('.')
    electra_model.train()