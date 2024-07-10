import torch
import torch.nn as nn
from torch.utils.data import  Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import models
from tools import *
from utils.serialization import load_checkpoint

"""
Datasets:test
input:
    path: images/
    mode: pre/test/train
"""
class VisualDataset(Dataset):
    def __init__(self, path, mode = 'pre'):
        self.path = path
        self.mode  = mode
        self.dataset = None
        self.transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if self.mode == 'pre':
            self.dataset = os.listdir(self.path)
        if self.mode == 'test':
            self.dataset = os.listdir(self.path)
        if self.mode == 'train':
            self.dataset = os.listdir(self.path)
            

        
    def __getitem__(self, index):
        if self.mode == 'test' or self.mode == 'train':
            img = Image.open(self.path  + self.dataset[index])
            img = self.transformations(img)
            label = int(self.img[index].split('_')[-1])
            return (img, label, index)
        if self.mode == 'pre':
            img = Image.open(self.path  + self.dataset[index])
            img = self.transformations(img)
            pre_img_name = self.dataset[index]
            return (img, pre_img_name,index)
        
    def __len__(self):
        return len(self.dataset)
        

"""
predict:
input:
    model: model
    data: dataloader
    device: device
output:
    result: list    
"""
def predict(model, data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    result = {}
    model.eval()
    with torch.no_grad():
        for i, (inputs, name, index) in enumerate(data):
            #print(name)
            inputs = inputs.to(device)
            feature, output = model(inputs)
      
            _, predicted = torch.max(output, 1)
            for ind, j in enumerate(name):
                result[j] = predicted[ind].item()+1
    return result




"""
load_model:
input:
    path: path
output:
    model: model
"""
def load_model(path):
    checkpoint = torch.load(path)
    network = models.create('resnet50', num_features=2048, num_classes=3)
    network.cuda()
    network.load_state_dict(checkpoint['state_dict'], strict=False)
    return network


"""
测试案例
"""
def test():
    print("begin tiles")
    # cut_wsi_into_tiles('./test/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png','./images')
    print("begin predict")
    model = load_model('./param/epoch_8_model_best.pth.tar')
    dataset = VisualDataset('./images/', mode='pre')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    result = predict(model, dataloader)
    
    # Convert result dictionary to list of tuples (image_name, predicted_label)
    result_list = [(name, predicted) for name, predicted in result.items()]

    # Save result_list to CSV
    df = pd.DataFrame(result_list, columns=['Image', 'Predicted_Label'])
    df.to_csv('result.csv', index=False)
    
    print("end predict")


