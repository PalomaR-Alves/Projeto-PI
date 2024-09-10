import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import cv2
from os import path
from PIL import Image
from data_utils import load_csv_data
from image_utils import pipeline1


class HouseNumberDataset(Dataset):
    def __init__(self, dataframe, root_dir, pipeline = None, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.pipeline = pipeline
        self.transform = transform
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = path.join(self.root_dir, self.dataframe.iloc[idx]['file_name'])
        image = cv2.imread(img_name)
        
        label = int(self.dataframe.iloc[idx]['label']) # casting str to int 

        if self.pipeline:
            image = self.pipeline.process(image)

        if self.transform:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
        
        return image, torch.tensor(label)

# Example usage
# Transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load the data
train_data = load_csv_data()
train_dataset = HouseNumberDataset(dataframe=train_data, root_dir='data/train', pipeline=pipeline1, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load test the data
test_data = load_csv_data()
test_dataset = HouseNumberDataset(dataframe=test_data, root_dir='data/test', pipeline=pipeline1, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)