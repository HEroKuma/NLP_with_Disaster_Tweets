import torch
import pandas as pd
import argparse
import os
import hiddenlayer as hl

from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW


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

    def train(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.parse()
        print(self.opt)
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
        model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator',num_labels=2)
        model.device
        indices = tokenizer.batch_encode_plus(texts, max_length=64, )


if __name__ == '__main__':
    electra_model = electra('.')
    electra_model.train()