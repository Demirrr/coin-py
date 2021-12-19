import torch


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, df, seq_length=2, even_interval=1, labeller=None):
        self.data = df.values
        self.even_interval = even_interval
        self.seq_length = seq_length
        self.labeller = labeller

        self.length = len(self.data) - self.seq_length  # - self.even_interval
        self.compute_stats()

    def compute_stats(self):
        labels = []
        for i in range(len(self)):
            labels.append(self[i][1].tolist())

        labels = torch.Tensor(labels)
        print('Distribution over labels:',labels.mean(dim=0))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:(idx + self.seq_length)]
        y = self.data[idx + self.seq_length:(idx + self.seq_length + self.even_interval)]
        x, y = self.labeller(x, y)
        return torch.Tensor(x), torch.Tensor(y)

class PredictPriceDataset(torch.utils.data.Dataset):
    def __init__(self, df, seq_length=12,length_next_seq=1):
        self.data = df.values
        self.seq_length = seq_length
        self.length_next_seq=length_next_seq
        self.length = len(self.data) - self.seq_length-  self.length_next_seq

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:(idx + self.seq_length)]
        y = self.data[idx + self.seq_length:(idx + self.seq_length+self.length_next_seq)]
        return torch.Tensor(x), torch.Tensor(y)
