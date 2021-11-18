import os
import pandas as pd

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

class PawpularityDataset(Dataset):
    def __init__(self, imgs, labels, transform):
        self.imgs = imgs
        self.labels = labels 
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = os.path.join('train', self.imgs[index] + '.jpg')
        img = Image.open(img_path)
        label = self.labels[index] / 100.0

        if self.transform:
            img = self.transform(img)
        
        return img, label

class PawpularityDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        annotations = pd.read_csv('train.csv')
        imgs = annotations["Id"].to_numpy()
        labels = annotations["Pawpularity"].to_numpy()
        self.img_train, self.img_val, self.label_train, self.label_val = train_test_split(imgs, labels, train_size=0.8)

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def train_dataloader(self):
        dataset = PawpularityDataset(self.img_train, self.label_train, self.train_transform)
        return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        dataset = PawpularityDataset(self.img_val, self.label_val, self.val_transform)
        return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)