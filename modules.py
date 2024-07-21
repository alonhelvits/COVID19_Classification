import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert('L')
        label = self.dataframe.iloc[idx]['finding']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
