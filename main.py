
from flask import Flask, jsonify, request, make_response, url_for
from flask_cors import CORS
from multiprocessing import Pool
# import re
from run import *
from datasets import *
import time
import pandas as pd
import os
import numpy as np
#import cv2
import matplotlib.pyplot as plt
import shutil
import csv
from werkzeug.utils import secure_filename
from torchvision import transforms
from torch.utils.data import  Dataset
import pandas as pd
import processed
import train
import threading
import pandas as pd
import numpy as np
import zipfile
import os 
from PIL import Image
import torch
from tqdm import tqdm
import pymysql


OPENSLIDE_PATH = r'D:\openslide\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm
app = Flask(__name__)
CORS(app)
app.config["JSON_AS_ASCII"] = False

# # read config
config = read_config()
#
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
current_directory_2 = os.path.dirname(os.path.dirname( current_directory))
current_iteration = 20
BASE_PATH= "../data_new" # change for your one address
liter ="4"
data_name = "test"
connection_path = "bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3"
BASE_PATH_2 = "./Data/save_data_" + liter
connection_path_2 = "Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3"
data_name_2 = "test2"
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


def getNum(filename):
    files_in_folder = os.listdir(filename)
    ans = len(files_in_folder)
    return ans








def process_images_in_directory(image_path, output_base_dir, slice_size=224):
    image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    
    # 获取当前文件夹中已有子文件夹的个数，用于确定新文件夹的名称
    subfolders = [f.path for f in os.scandir(output_base_dir) if f.is_dir()]
    folder_count = len(subfolders)
    
    for idx, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        # 打开图片
        with Image.open(os.path.join(image_path, image_file)) as image:
            # 新文件夹的名称
            new_folder_name = f"image_{folder_count + idx}"
            # 新文件夹的完整路径
            new_folder_path = os.path.join(output_base_dir, new_folder_name)
            
            # 如果新文件夹不存在，则创建
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            
            # 计算切片的行数和列数
            rows = image.size[1] // slice_size
            cols = image.size[0] // slice_size
            
            # 逐个切片并保存
            for row in range(rows):
                for col in range(cols):
                    left = col * slice_size
                    upper = row * slice_size
                    right = left + slice_size
                    lower = upper + slice_size

                    # 裁剪图片
                    image_slice = image.crop((left, upper, right, lower))
                    
                    # 保存切片图片
                    slice_name = f"x_{row}_y_{col}.png"
                    image_slice.save(os.path.join(new_folder_path, slice_name))
    
    print("Processing completed.")


def add_Access(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'GET, POST, OPTIONS')
    response.headers.add("X-Powered-By", ' 3.2.1')
    response.headers.add("Content-Type", "application/json;charset=utf-8")
    response.headers.add('Access-Control-Allow-Methods',
                         'DNT,X-Mx-ReqToken,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization')
 
    return response

def normalize_df(df,key=[]):
    smooth=1e-5
    for k in key:
        k_l=np.array(df[k].to_list())
        k_max=np.max(k_l)
        k_min=np.min(k_l[k_l!=0])
        df[k]=(k_l-k_min)/(k_max-k_min+smooth)
    return df


@app.route("/change", methods=["GET"]) 
def first_run_2():
    global current_iteration
    global config
    global logger
    model = None
    dataLoader = None
    try:
        if current_iteration == 0:
            print(0)
        else:
            source_file = '../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/new.csv'
            target_folder = './Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/'
            if not os.path.exists(os.path.join(target_folder, 'new.csv')):
                  shutil.copy(source_file, target_folder)
           
            print("loading data" )
            if data_name_2=="test2":
                path = os.path.join(BASE_PATH_2,connection_path_2)
                
            # elif data_name=="hubmap":
            #     path = os.path.join(BASE_PATH,"save_data/")               
            sample_data = pd.read_csv(os.path.join(path, 'sample_data.csv')).to_dict()
            epoch_Data = pd.read_csv(os.path.join(path, 'epoch_Data.csv')).to_dict()
            #WSI_Data=normalize_df(pd.read_csv(os.path.join(path, 'WSI_Data.csv')),key=['bmm','spr']).to_dict()
            WSI_Data=pd.read_csv(os.path.join(path, 'WSI_Data.csv')).to_dict()
            New_data=pd.read_csv(os.path.join(path, 'new.csv')).to_dict()
            confusion_Data = pd.read_csv(os.path.join(
                path, 'confusion.csv')).to_dict() if data_name=="test" else None
            bk_data = pd.read_csv(os.path.join(
                path, 'bk_data.csv')).to_dict() if data_name=="test" else None

            response = make_response(jsonify({
                'load_status': 200,
                'dataset': data_name,
                'iteration': current_iteration,
                'sample_data': sample_data,
                'epoch_Data': epoch_Data,
                'WSI_Data': WSI_Data,
                'New_data':New_data,
                'confusion_Data': confusion_Data,
                'bk_data': bk_data,
            }))
    except Exception as e:
        print("[Exception]:",e)
        response = make_response(jsonify({"load_status": 500}))
    
    return add_Access(response)

def load_model(path):
    checkpoint = torch.load(path)
    network = models.create('resnet50', num_features=2048, num_classes=3)
    network.cuda()
    network.load_state_dict(checkpoint['state_dict'], strict=False)
    return network

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

@app.route("/change_2", methods=["GET"]) 
def first_run_3():
    global current_iteration
    global config
    global logger
    model = None
    dataLoader = None
    try:
        if current_iteration == 0:
            print(0)
        else:
            print("loading data" )
            if data_name_2=="test2":
                path = os.path.join(BASE_PATH,connection_path)
                
            # elif data_name=="hubmap":
            #     path = os.path.join(BASE_PATH,"save_data/")               
            sample_data = pd.read_csv(os.path.join(path, 'sample_data.csv')).to_dict()
            epoch_Data = pd.read_csv(os.path.join(path, 'epoch_Data.csv')).to_dict()
            #WSI_Data=normalize_df(pd.read_csv(os.path.join(path, 'WSI_Data.csv')),key=['bmm','spr']).to_dict()
            WSI_Data=pd.read_csv(os.path.join(path, 'WSI_Data.csv')).to_dict()
            New_data=pd.read_csv(os.path.join(path, 'new.csv')).to_dict()
            
            
            confusion_Data = pd.read_csv(os.path.join(
                path, 'confusion.csv')).to_dict() if data_name=="test" else None
            bk_data = pd.read_csv(os.path.join(
                path, 'bk_data.csv')).to_dict() if data_name=="test" else None

            response = make_response(jsonify({
                'load_status': 200,
                'dataset': data_name,
                'iteration': current_iteration,
                'sample_data': sample_data,
                'epoch_Data': epoch_Data,
                'WSI_Data': WSI_Data,
                'New_data':New_data,
                'confusion_Data': confusion_Data,
                'bk_data': bk_data,
            }))
    except Exception as e:
        print("[Exception]:",e)
        response = make_response(jsonify({"load_status": 500}))
    
    return add_Access(response)

def load_model(path):
    checkpoint = torch.load(path)
    network = models.create('resnet50', num_features=2048, num_classes=3)
    network.cuda()
    network.load_state_dict(checkpoint['state_dict'], strict=False)
    return network

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




@app.route("/init", methods=["GET"])
def first_run():
    global current_iteration
    global config
    global logger
    model = None
    dataLoader = None
    try:
        if current_iteration == 0:
            print(0)
        else:
            print("loading data" )
            if data_name=="test":
                path = os.path.join(BASE_PATH,connection_path)
                
            # elif data_name=="hubmap":
            #     path = os.path.join(BASE_PATH,"save_data/")               
            sample_data = pd.read_csv(os.path.join(path, 'sample_data.csv')).to_dict()
            epoch_Data = pd.read_csv(os.path.join(path, 'epoch_Data.csv')).to_dict()
            #WSI_Data=normalize_df(pd.read_csv(os.path.join(path, 'WSI_Data.csv')),key=['bmm','spr']).to_dict()
            WSI_Data=pd.read_csv(os.path.join(path, 'WSI_Data.csv')).to_dict()
            New_data=pd.read_csv(os.path.join(path, 'new.csv')).to_dict()
            confusion_Data = pd.read_csv(os.path.join(
                path, 'confusion.csv')).to_dict() if data_name=="test" else None
            bk_data = pd.read_csv(os.path.join(
                path, 'bk_data.csv')).to_dict() if data_name=="test" else None

            response = make_response(jsonify({
                'load_status': 200,
                'dataset': data_name,
                'iteration': current_iteration,
                'sample_data': sample_data,
                'epoch_Data': epoch_Data,
                'WSI_Data': WSI_Data,
                'New_data':New_data,
                'confusion_Data': confusion_Data,
                'bk_data': bk_data,
            }))
    except Exception as e:
        print("[Exception]:",e)
        response = make_response(jsonify({"load_status": 500}))
    
    return add_Access(response)

def load_model(path):
    checkpoint = torch.load(path)
    network = models.create('resnet50', num_features=2048, num_classes=3)
    network.cuda()
    network.load_state_dict(checkpoint['state_dict'], strict=False)
    return network

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

def write_to_csv(patch_id, img_id, selected_label, origin_label):
    print("you have reached here !!!!!!11")
    output_file_path = './Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/new.csv'
    fieldnames = ['patch_id', 'img_id', 'selected_label', 'origin_label']
    
    # 写入新的CSV文件或追加到现有文件中
    with open(output_file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果文件为空，写入表头
        if os.stat(output_file_path).st_size == 0:
            writer.writeheader()
        
        # 写入新的行
        writer.writerow({
            'patch_id': patch_id,
            'img_id': img_id,
            'selected_label': selected_label,
            'origin_label':  origin_label
        })
def write_to_csv2(patch_id, img_id, selected_label, origin_label2):
    output_file_path = '../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/new.csv'
    fieldnames = ['patch_id', 'img_id', 'selected_label', 'origin_label']
    
    # 写入新的CSV文件或追加到现有文件中
    with open(output_file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果文件为空，写入表头
        if os.stat(output_file_path).st_size == 0:
            writer.writeheader()
        
        # 写入新的行
        writer.writerow({
            'patch_id': patch_id,
            'img_id': img_id,
            'selected_label': selected_label,
            'origin_label':  origin_label2
        })



@app.route("/last2",methods = ['POST'])
def last2():
    data = request.json
    img_id = data.get('img_id')
    patch_id = data.get('patch_id')
    print(img_id)
    print(patch_id)

    with open('./Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/sample_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # 读取所有记录
   
    

    # 对匹配的记录进行修改
    for row in rows:
        if row['file_name'] == str(patch_id) and row['img_id'] == str(img_id):
            # 根据selected_label修改class列的值
            row['class'] = 1
            row['is_labeled'] = 1
            # 修改file_name列的值
            # 解析原始的file_name

    # 写入到临时文件
    output_file_path = '../temp/sample_data_modified2.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 确保输出文件夹存在
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    destination_path = './Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/sample_data.csv'
    shutil.copy(output_file_path, destination_path)
    return jsonify({'message': 'CSV file saved successfully'})


@app.route("/last2_2",methods = ['POST'])
def last2_2():
    data = request.json
    img_id = data.get('img_id')
    patch_id = data.get('patch_id')
    print(img_id)
    print(patch_id)

    with open('../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/sample_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # 读取所有记录
   
    

    # 对匹配的记录进行修改
    for row in rows:
        if row['file_name'] == str(patch_id) and row['img_id'] == str(img_id):
            # 根据selected_label修改class列的值
            row['class'] = 1
            row['is_labeled'] = 1
            # 修改file_name列的值
            # 解析原始的file_name

    # 写入到临时文件
    output_file_path = '../temp/sample_data_modified2_2.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 确保输出文件夹存在
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    destination_path = '../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/sample_data.csv'
    shutil.copy(output_file_path, destination_path)
    return jsonify({'message': 'CSV file saved successfully'})

@app.route("/last3",methods = ['POST'])
def last3():
    data = request.json
    img_id = data.get('img_id')
    patch_id = data.get('patch_id')
    print(img_id)
    print(patch_id)

    with open('./Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/sample_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # 读取所有记录
   
    

    # 对匹配的记录进行修改
    for row in rows:
        if row['file_name'] == str(patch_id) and row['img_id'] == str(img_id):
            # 根据selected_label修改class列的值
            row['class'] = 2
            row['is_labeled'] = 1
            # 修改file_name列的值
            # 解析原始的file_name

    # 写入到临时文件
    output_file_path = '../temp/sample_data_modified2.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 确保输出文件夹存在
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    destination_path = './Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/sample_data.csv'
    shutil.copy(output_file_path, destination_path)
    return jsonify({'message': 'CSV file saved successfully'})


@app.route("/last3_2",methods = ['POST'])
def last3_2():
    data = request.json
    img_id = data.get('img_id')
    patch_id = data.get('patch_id')
    print(img_id)
    print(patch_id)

    with open('../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/sample_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # 读取所有记录
   
    

    # 对匹配的记录进行修改
    for row in rows:
        if row['file_name'] == str(patch_id) and row['img_id'] == str(img_id):
            # 根据selected_label修改class列的值
            row['class'] = 2
            row['is_labeled'] = 1
            # 修改file_name列的值
            # 解析原始的file_name

    # 写入到临时文件
    output_file_path = '../temp/sample_data_modified2_2.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 确保输出文件夹存在
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    destination_path = '../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/sample_data.csv'
    shutil.copy(output_file_path, destination_path)
    return jsonify({'message': 'CSV file saved successfully'})


@app.route("/last4",methods = ['POST'])
def last4():
    data = request.json
    img_id = data.get('img_id')
    patch_id = data.get('patch_id')
    print(img_id)
    print(patch_id)

    with open('./Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/sample_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # 读取所有记录
   
    

    # 对匹配的记录进行修改
    for row in rows:
        if row['file_name'] == str(patch_id) and row['img_id'] == str(img_id):
            # 根据selected_label修改class列的值
            row['class'] = 3
            row['is_labeled'] = 1
            # 修改file_name列的值
            # 解析原始的file_name

    # 写入到临时文件
    output_file_path = '../temp/sample_data_modified2.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 确保输出文件夹存在
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    destination_path = './Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/sample_data.csv'
    shutil.copy(output_file_path, destination_path)
    return jsonify({'message': 'CSV file saved successfully'})


@app.route("/last4_2",methods = ['POST'])
def last4_2():
    data = request.json
    img_id = data.get('img_id')
    patch_id = data.get('patch_id')
    print(img_id)
    print(patch_id)

    with open('../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/sample_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # 读取所有记录
   
    

    # 对匹配的记录进行修改
    for row in rows:
        if row['file_name'] == str(patch_id) and row['img_id'] == str(img_id):
            # 根据selected_label修改class列的值
            row['class'] = 3
            row['is_labeled'] = 1
            # 修改file_name列的值
            # 解析原始的file_name

    # 写入到临时文件
    output_file_path = '../temp/sample_data_modified2_2.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 确保输出文件夹存在
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    destination_path = '../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/sample_data.csv'
    shutil.copy(output_file_path, destination_path)
    return jsonify({'message': 'CSV file saved successfully'})

  
@app.route("/last",methods = ['POST'])
def last():
    data = request.json
    selected_label = data.get('selectedLabel')
    path = data.get('path')
    path2 = data.get('path2')
    print(path)
    print(path2)
    with open('./Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/sample_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # 读取所有记录
    for row in rows:

        if row['patch_id'] == str(path) and row['img_id'] == str(path2):
            print("原来的class的类别是",row['class'])
            if  row['class'] == '1':
                origin_label = "no cancer"
            elif row['class'] == '2':
                origin_label = "cancer"
            elif row['class'] == '3':
                origin_label = "high cancer"
    write_to_csv(path, path2, selected_label, origin_label)
    #target 获得原来的标签
    # 读取CSV文件并查找匹配的记录，并进行修改
    

    # 对匹配的记录进行修改
    for row in rows:
        if row['patch_id'] == str(path) and row['img_id'] == str(path2):
            print("原来的class的类别是",row['class'])
            # 根据selected_label修改class列的值
            if selected_label == 'no cancer':
                row['class'] = '1'
                row['noise'] =  0
            elif selected_label == 'cancer':
                row['class'] = '2'
                row['noise'] =  0
            elif selected_label == 'high cancer':
                row['class'] = '3'
                row['noise'] =  0
            
            # 修改file_name列的值
            # 解析原始的file_name
            file_name_parts = row['file_name'].split('_')
            # 修改class部分
            file_name_parts[-1] = row['class']
            # 重新组合成新的file_name
            new_file_name = '_'.join(file_name_parts) + ".png"
            row['file_name'] = new_file_name

    # 写入到临时文件
    output_file_path = '../temp/sample_data_modified.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 确保输出文件夹存在
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    destination_path = './Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/sample_data.csv'
    shutil.copy(output_file_path, destination_path)





    
    return jsonify({'message': 'CSV file saved successfully'})



@app.route("/last_2",methods = ['POST'])
def last_2():
    data = request.json
    selected_label = data.get('selectedLabel')
    path = data.get('path')
    path2 = data.get('path2')
    with open('../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/sample_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # 读取所有记录
    for row in rows:
        if row['patch_id'] == str(path) and row['img_id'] == str(path2):
            print("原来的class的类别是",row['class'])
            if  row['class'] == '1':
                origin_label2 = "no cancer"
            elif row['class'] == '2':
                origin_label2 = "cancer"
            elif row['class'] == '3':
                origin_label2 = "high cancer"
    write_to_csv2(path, path2, selected_label, origin_label2)
    #target 获得原来的标签
    # 读取CSV文件并查找匹配的记录，并进行修改
    

    # 对匹配的记录进行修改
    for row in rows:
        if row['patch_id'] == str(path) and row['img_id'] == str(path2):
            print("原来的class的类别是",row['class'])
            # 根据selected_label修改class列的值
            if selected_label == 'no cancer':
                row['class'] = '1'
                row['noise'] =  0
            elif selected_label == 'cancer':
                row['class'] = '2'
                row['noise'] =  0
            elif selected_label == 'high cancer':
                row['class'] = '3'
                row['noise'] =  0
            
            # 修改file_name列的值
            # 解析原始的file_name
            file_name_parts = row['file_name'].split('_')
            # 修改class部分
            file_name_parts[-1] = row['class']
            # 重新组合成新的file_name
            new_file_name = '_'.join(file_name_parts) + ".png"
            row['file_name'] = new_file_name

    # 写入到临时文件
    output_file_path = '../temp/sample_data_modified_2.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 确保输出文件夹存在
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    destination_path = '../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/sample_data.csv'
    shutil.copy(output_file_path, destination_path)





    
    return jsonify({'message': 'CSV file saved successfully'})


progress = 0
print_load = None

terminate_thread = False


@app.route("/loading_2", methods=["GET"]) 
def get_progress():
        return jsonify({'progress': progress, 'print_load':print_load, 'trainingComplete': progress >= 100})

def send_signal():
    global progress
    global terminate_thread
    while not terminate_thread:
        progress += 0.5
        time.sleep(25)  # 每隔 5 秒发送一次信号
        if terminate_thread:
            break
        
# 创建并启动发送信号的线程
signal_thread = threading.Thread(target=send_signal)

@app.route("/mysql",methods = ['POST'])
def mysql():
    try:
        conn = pymysql.connect(
         host='localhost',  # 主机名
         port=3306,         # 端口号，MySQL默认为3306
         user='root',       # 用户名
         password='123',    # 密码
         database='tvcg2'   # 数据库名称
        )
        cursor = conn.cursor()
        patient_name = request.form.get('patientName')
        patient_id = request.form.get('patientID')
        sql = "INSERT INTO patient (就诊号, 报告号) VALUES (%s, %s)"
        values = (patient_name, patient_id)  #wait to be seen 
        cursor.execute(sql, values)
        conn.commit()
       
    except Exception as e:
        print("[Exception]:", e)
        return jsonify({'load_status': 500}), 500
    return jsonify({'load_status': 200}), 200

@app.route("/mysql2",methods = ['GET'])
def mysql2():
    try:
        conn = pymysql.connect(
         host='localhost',  # 主机名
         port=3306,         # 端口号，MySQL默认为3306
         user='root',       # 用户名
         password='123',    # 密码
         database='tvcg2'   # 数据库名称
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patient")
        result = cursor.fetchall()
        data = [{'patient_id': row[0], 'patient_name': row[1],'patient_key' : row[2]} for row in result]

        # 返回 JSON 格式的数据给前端
    except Exception as e:
        print("[Exception]:", e)
        return jsonify({'load_status': 500}), 500
    return jsonify({'load_status': 200, 'data': data})


@app.route("/mysql3",methods = ['POST'])
def mysql3():
    try:
        conn = pymysql.connect(
         host='localhost',  # 主机名
         port=3306,         # 端口号，MySQL默认为3306
         user='root',       # 用户名
         password='123',    # 密码
         database='tvcg2'   # 数据库名称
        )
        ID = request.json.get('ID')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patient WHERE ID = %s", (ID))
        conn.commit()

        # 返回 JSON 格式的数据给前端
    except Exception as e:
        print("[Exception]:", e)
        return jsonify({'load_status': 500}), 500
    return jsonify({'load_status': 200}), 200

@app.route("/load",methods = ['POST'])
def load():
    global progress
    global print_load
    global terminate_thread
    try:
        progress = 0
        print_load = "开始训练 !!!!!!!!!!!!!!!!!!!!!!"
        file = request.files['file']
        filename = file.filename
        print("here we got ", filename)
        WSI_path = './Wsis'
        WSI_new_path = '../../../Visual/public/data/WSIs_new'
       
        if len(os.listdir(WSI_path)) > 0:
            print_load = "正在清理缓存"
            progress = 2
            print_load = "清理WSI缓存!!!!!!!!!!!"
    # 清空目录
            for file in os.listdir(WSI_path):
              file_path = os.path.join(WSI_path, file)
              if os.path.isfile(file_path):
                 os.remove(file_path)
              elif os.path.isdir(file_path):
                 shutil.rmtree(file_path)
      
        if len(os.listdir( WSI_new_path)) > 0:
    # 清空目录
            progress = 4
            print_load = "清理新WSI缓存!!!!!!!!!!!"
            for file in os.listdir(WSI_new_path):
              file_path = os.path.join(WSI_new_path, file)
              if os.path.isfile(file_path):
                 os.remove(file_path)
              elif os.path.isdir(file_path):
                 shutil.rmtree(file_path)
      
        if len(os.listdir( '../../../Visual/public/data/init_data_image_new')) > 0:
            progress = 6
            print_load = "清理新image缓存!!!!!!!!!!!"
    # 清空目录
            for file in os.listdir('../../../Visual/public/data/init_data_image_new'):
              file_path = os.path.join('../../../Visual/public/data/init_data_image_new', file)
              if os.path.isfile(file_path):
                 os.remove(file_path)
              elif os.path.isdir(file_path):
                 shutil.rmtree(file_path)
     
        if len(os.listdir( './test3')) > 0:
            progress = 8
            print_load = "清理压缩包缓存!!!!!!!"
    # 清空目录
            for file in os.listdir( './test3'):
              file_path = os.path.join( './test3', file)
              if os.path.isfile(file_path):
                 os.remove(file_path)
              elif os.path.isdir(file_path):
                 shutil.rmtree(file_path)
        print("5")
     
        if len(os.listdir( './save/init_data')) > 0:
            progress = 10
            print_load = "清理init_data缓存!!!!!!!"
    # 清空目录
            for file in os.listdir('./save/init_data'):
              file_path = os.path.join('./save/init_data', file)
              if os.path.isfile(file_path):
                 os.remove(file_path)
              elif os.path.isdir(file_path):
                 shutil.rmtree(file_path)
        print("6")
    
        if len(os.listdir( './save/init_data_image')) > 0:
            progress = 11
            print_load = "清理init_data_image缓存!!!!!!!"
    # 清空目录
            for file in os.listdir('./save/init_data_image'):
              file_path = os.path.join('./save/init_data_image', file)
              if os.path.isfile(file_path):
                 os.remove(file_path)
              elif os.path.isdir(file_path):
                 shutil.rmtree(file_path)
   
        if len(os.listdir( './Data')) > 0:
            progress = 12
            print_load = "清理Data缓存!!!!!!!"
            for file in os.listdir('./Data'):
              file_path = os.path.join('./Data', file)
              if os.path.isfile(file_path):
                 os.remove(file_path)
              elif os.path.isdir(file_path):
                 shutil.rmtree(file_path)
        print_load = "正在开始解压数据，请稍后"
        file_path = "./" + filename # 前端上传的zip路径

        print_load = "开始解压数据集压缩包!!!!!!!!!!!!!!!!"
        upzip_files = processed.unzip_data(file_path, WSI_path, WSI_new_path) # 解压
        
        image_path, mask_path = upzip_files
        output_base_dir = "../../../Visual/public/data/init_data_image_new"
        progress = 13
        print_load = "数据集压缩包解压完成,正在开始切片！！！！！！！！"
        
        process_images_in_directory(image_path, output_base_dir)
        progress = 16
        print_load = "WSI图片切片完成!!!!!!!!!!!!!"
        WSI_path = './Wsis'
        thumbnails_path = './save/init_data_image' # 切片保存数据——展示要的image_1、image_2文件夹那个
        output_path = './save/init_data' # 训练需要的文件夹
        progress = 20
        print_load = "切片完成，开始处理数据!!!!!!!!!!!!"
        progress = 25
        print_load = "开始将图像和蒙版切割成小图像，并将它们保存到相应的目录中!!!!!!!!!!!!!!!!"
        processed.process_data(image_path, mask_path,thumbnails_path, output_path)
        progress = 35
        print_load = "开始处理数据并将他们存储到npy文件夹中!!!!!!!!!!!!!!!!!!!!!!!"
        time.sleep(5)
        progress = 45
        print_load = "数据处理完成!!!!!!!!!!!!!"
        time.sleep(5)
        print_load = "下面开始训练!!!!!!!!!!!!"
        progress = 47
        print_load = "初始化模型参数!!!!!!!!!"
        time.sleep(2)
        progress = 50
        print_load = "初始化模型参数!!!!!!!!!"
        time.sleep(2)
        progress = 52
        print_load = '[INFO]:=================================================='
        print_load = '[INFO]:cuda_id:0'
        time.sleep(2)
        print_load = '[INFO]:multi_cuda:0'
        print_load = '[INFO]:seed:42'
        time.sleep(2)
        print_load = '[INFO]:model:resnet50'
        progress = 55
        time.sleep(2)
        print_load = '[INFO]:num_features:2048'
        time.sleep(2)
        print_load = '[INFO]:batch_size:12'
        print_load = '[INFO]:optim:SGD'
        time.sleep(2)
        print_load = '[INFO]:loss:Recall'
        print_load = '[INFO]:num_workers:4'
        time.sleep(2)
        print_load = '[INFO]:dataset:{"name": "breast_train", "path": "D:/TVCG2/FlaskServer/FlaskServer/BPAL/save/init_data", "size: 224", "tile_size: 1024", "num_class: 3"}'
        print_load = '[INFO]:num_classes:3'
        time.sleep(2)
        print_load = '[INFO]:num_classes_sub:3'
        progress = 57
        time.sleep(2)
        print_load = '[INFO]:noise_rate:0.3'
        print_load = '[INFO]:noise_method:nearly'
        print_load = '[INFO]:noise_range:all'
        print_load = '[INFO]:methods:Both'
        print_load = '[INFO]:pretrained_iteration:5'
        print_load = '[INFO]:max_iteration:1'
        print_load = '[INFO]:Kmeans_cluster:10'
        time.sleep(2)
        print_load = '[INFO]:Kmeans_Visual_cluster:8'
        print_load = '[INFO]:CC_cluster:3'
        progress = 59
        time.sleep(2)
        print_load = '[INFO]:PCA_components:256'
        print_load =  '[INFO]:Knn_act:0'
        print_load =  '[INFO]:Knn_number:1'
        print_load =  '[INFO]:grad_save:0'
        time.sleep(2)
        print_load =  '[INFO]:visual_method:tsne'
        print_load =  '[INFO]:save_data_dir:./Data'
        time.sleep(2)
        print_load =  '[INFO]:logger_path:./logs'
        print_load =  '[INFO]:save_param_dir:./param'
        print_load =  '[INFO]:=================================================='
        time.sleep(5)
        progress = 60
        print_load =  '!!!!!!!!!!!!!!正式开始训练，请耐心的等待！！！！！！！！！！！！！！'
        signal_thread.start()
        if(progress == 96):
            terminate_thread = True
        train.train_2('./config/config.yaml')
       
        progress = 100
        time.sleep(20)
        progress = 0
        print_load = None
    except Exception as e:
        print("[Exception]:", e)
        return jsonify({'load_status': 500}), 500
    return jsonify({'load_status': 200}), 200

@app.route("/delete", methods=['POST'])
def delete():
    data = request.json
    a = data.get('a')  # patch_id
    b = data.get('b')  # img_id
    
    # 读取 CSV 文件
    df = pd.read_csv('./Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/new.csv')
    print("you have reach here 1")
    # 根据条件删除记录
    df = df[(df['patch_id'] != b) | (df['img_id'] != a)]
    
    # 保存修改后的文件
    df.to_csv('./Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/new.csv', index=False)
    
    print("Deleted records where patch_id = {} and img_id = {}".format(b, a))

    
    with open('./Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/sample_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # 读取所有记录
        print(len(rows))
    for row in rows:
        if row['patch_id'] == str(b) and row['img_id'] == str(a):
            row['noise'] = 1
    # 写入到临时文件
    output_file_path = '../temp/sample_data_modified3.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 确保输出文件夹存在
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    destination_path = './Data/save_data_' + liter + '/Both_Recall_breast_train_batch_12_tileSize_1024_noise_0.3/sample_data.csv'
    shutil.copy(output_file_path, destination_path)


    
    
    return jsonify({'message': 'Successfully deleted'})




@app.route("/delete2", methods=['POST'])
def delete2():
    data = request.json
    a = data.get('a')  # patch_id
    b = data.get('b')  # img_id
    
    # 读取 CSV 文件
    df = pd.read_csv('../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/new.csv')
    print("you have reach here 1")
    # 根据条件删除记录
    df = df[(df['patch_id'] != b) | (df['img_id'] != a)]
    
    # 保存修改后的文件
    df.to_csv('../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/new.csv', index=False)
    
    print("Deleted records where patch_id = {} and img_id = {}".format(b, a))

    
    with open('../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/sample_data.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # 读取所有记录
        print(len(rows))
    for row in rows:
        if row['patch_id'] == str(b) and row['img_id'] == str(a):
            row['noise'] = 1
    # 写入到临时文件
    output_file_path = '../temp/sample_data_modified3_2.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 确保输出文件夹存在
    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    destination_path = '../data_new/bmm+spr_Recall_breast_train_batch_48_tileSize_1024_noise_0.3/sample_data.csv'
    shutil.copy(output_file_path, destination_path)

    return jsonify({'message': 'Successfully deleted'})

if __name__ == '__main__':
    app.run(use_reloader=False,debug=True)
