import os
import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from transformers import ElectraTokenizer, ElectraForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

class Data(Dataset):
    def __init__(self, root, mode):
        self.data = pd.read_csv(os.path.join(root, '{}.csv'.format(mode)))
        self.mode = mode
        self.data['text'] = self.data['text'].apply(self.preprocess)
        self.data = self.data[self.data['text'] != '']
        if self.mode == 'train':
            self.data = self.data[['text', 'target']]
            self.labels = self.data.target.values
        self.text = self.data.text.values


    def __getitem__(self, index):
        data = [self.text[index]]
        if self.mode == 'train':
            data.append(self.labels[index])
        return data

    def __len__(self):
        return len(self.file)

    def preprocess(self, text):
        text = text.lower()
        # remove hyperlinks
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
        text = re.sub(r'http?:\/\/.*[\r\n]*', '', text)
        # Replace &amp, &lt, &gt with &,<,> respectively
        text = text.replace(r'&amp;?', r'and')
        text = text.replace(r'&lt;', r'<')
        text = text.replace(r'&gt;', r'>')
        # remove mentions
        text = re.sub(r"(?:\@)\w+", '', text)
        # remove non ascii chars
        text = text.encode("ascii", errors="ignore").decode()
        # remove some puncts (except . ! ?)
        text = re.sub(r'[:"#$%&\*+,-/:;<=>@\\^_`{|}~]+', '', text)
        text = re.sub(r'[!]+', '!', text)
        text = re.sub(r'[?]+', '?', text)
        text = re.sub(r'[.]+', '.', text)
        text = re.sub(r"'", "", text)
        text = re.sub(r"\(", "", text)
        text = re.sub(r"\)", "", text)

        text = " ".join(text.split())

        return text

    def split_data(self):
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
        indices = tokenizer.batch_encode_plus(self.text, max_length=64, add_special_tokens=True,
                                              return_attention_mask=True,
                                              pad_to_max_length=True, truncation=True)
        input_ids = indices["input_ids"]
        attention_masks = indices["attention_mask"]
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                            self.labels,
                                                                                            random_state=87,
                                                                                            test_size=.2)
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, self.labels,
                                                               random_state=87, test_size=.2)
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        validation_labels = torch.tensor(validation_labels, dtype=torch.long)
        train_masks = torch.tensor(train_masks, dtype=torch.long)
        validation_masks = torch.tensor(validation_masks, dtype=torch.long)

if __name__ == '__main__':
    a = Data('.', 'test')
    print(a.data)