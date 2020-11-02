import torch
import pandas as pd
import numpy as np
import argparse
import time
import datetime
import os
import hiddenlayer as hl
import random
import tqdm

from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup
from DataLoader import *

class electra:
    def __init__(self, path):
        self.parser = argparse.ArgumentParser()
        self.model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator',num_labels=2)

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
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.parse()
        print(self.opt)

        train_data = Data('.', 'train')

        # model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator',num_labels=2)
        self.model.to(self.device)

        batch_size = self.opt.batch
        optimizer = AdamW(self.model.parameters(), lr=self.opt.lr, eps=self.opt.eps)
        t_data, t_sampler, v_data, v_sampler = train_data.split_data()
        self.train_dataloader = DataLoader(t_data, sampler=t_sampler, batch_size=self.opt.batch)
        self.validation_dataloader = DataLoader(v_data, sampler=v_sampler, batch_size=self.opt.batch)

        total_steps = len(self.train_dataloader) * self.opt.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        seed_val = 14
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        loss_values = []
        with tqdm.tqdm(total=self.opt.epoch, miniters=1, mininterval=0) as progress:
            for epoch_i in range(self.opt.epoch):
                print("")
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.opt.epoch))
                print('Training...')
                t0 = time.time()
                total_loss = 0
                elapsed = self.format_time(time.time() - t0)
                self.model.train()
                for step, batch in enumerate(self.train_dataloader):
                    b_input_ids = batch[0].to(self.device)
                    b_input_mask = batch[1].to(self.device)
                    b_labels = batch[2].to(self.device)
                    self.model.zero_grad()
                    outputs = self.model(b_input_ids, token_type_ids=None,
                                         attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs[0]
                    total_loss == loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    progress.set_description(
                        '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed)
                    )

                avg_train_loss = total_loss / len(self.train_dataloader)
                loss_values.append(avg_train_loss)
                print("")
                print("  Average training loss: {0:.2f}".format(avg_train_loss))
                print("  Training epoch took: {:}".format(self.format_time(time.time() - t0)))

        print("")
        print("Training complete!")

    def validation(self):
        t0 = time.time()
        self.model.eval()
        self.preds = []
        self.true = []
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in self.validation_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            self.preds.append(logits)
            self.true.append(label_ids)

            tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps == 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

    def metrics(self):
        flat_predictions = [item for sublist in self.preds for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in self.true for item in sublist]
        print(classification_report(flat_predictions, flat_true_labels))

        return

    def test(self):



if __name__ == '__main__':
    electra_model = electra('.')
    electra_model.train()
    electra_model.validation()
    electra_model.metrics()
