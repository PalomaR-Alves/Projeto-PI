import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import pandas as pd
import os
from PIL import Image
from data_utils import load_csv_data


# Define a simple CNN architecture
class HouseNumberCNN(nn.Module):
    def __init__(self):
        super(HouseNumberCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Output layers (4 digits, 0-9 classification for each digit)
        self.digit1 = nn.Linear(256, 10)  # Thousand place
        self.digit2 = nn.Linear(256, 10)  # Hundred place
        self.digit3 = nn.Linear(256, 10)  # Tens place
        self.digit4 = nn.Linear(256, 10)  # Ones place
        
    def forward(self, x):
        # Convolutional layers with ReLU and Max Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layers for each digit
        out_digit1 = self.digit1(x)
        out_digit2 = self.digit2(x)
        out_digit3 = self.digit3(x)
        out_digit4 = self.digit4(x)
        
        return out_digit1, out_digit2, out_digit3, out_digit4

# Define the dataset class
class HouseNumberDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['file_name'])
        image = Image.open(img_name).convert("RGB")
        
        label = self.dataframe.iloc[idx]['label']

        # Check if label is an integer, if so, convert it to a list
        if isinstance(label, int):
            label = [label]

        # Ensure label is a list with 4 digits (pad with zeros if necessary)
        label = list(map(int, label)) + [0] * (4 - len(label))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)

# Example usage
# Transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load the data
data = load_csv_data()
dataset = HouseNumberDataset(dataframe=data, root_dir='data/train', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate model, loss function, and optimizer
model = HouseNumberCNN()
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for each digit classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Example for 10 epochs
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = 0
        
        # Calculate loss for each digit separately
        for i in range(4):
            loss += criterion(outputs[i], labels[:, i])
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

# After training, use the model for prediction on new images
