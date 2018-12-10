import numpy as np
from skiimage import transform
from torch.utils.data import Dataset, DataLoader

class GeezDigitsDataset(Dataset):
    '''Fashion MNIST Dataset'''
    def __init__(self, X,Y, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file
            transform (callable): Optional transform to apply to sample
        """
        self.X = X.reshape(-1, 1, 28, 28)#.astype(float);
        self.Y = Y;
        self.transform = transform;

    def __len__(self):
        return len(self.X,self.Y);

    def __getitem__(self, idx):
        item = self.X[idx];
        label = self.Y[idx];

        if self.transform:
            item = self.transform(item);

        return (item, label);
