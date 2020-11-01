import os
import pandas as pd
import re
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, root, mode):
        self.data = pd.read_csv(os.path.join(root, '{}.csv'.format(mode)))
        self.data['text'] = self.data['text'].apply(self.preprocess)
        if mode == 'train':
            self.data = self.data[self.data['text'] != '']
            self.data = self.data[['text', 'target']]

    def __getitem__(self, index):
        return self.data[index]

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

if __name__ == '__main__':
    a = Data('.', 'test')
    print(a.data)