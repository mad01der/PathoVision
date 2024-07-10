import os

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
# 指定文件夹路径
def process_images_in_directory(image_path, output_base_dir, slice_size=224):
    # 获取当前文件夹中所有图片的文件名列表
    image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    
    # 获取当前文件夹中已有子文件夹的个数，用于确定新文件夹的名称
    subfolders = [f.path for f in os.scandir(output_base_dir) if f.is_dir()]
    folder_count = len(subfolders)
    
    for image_file in tqdm(image_files, desc="Processing images"):
        # 打开图片
        with Image.open(os.path.join(image_path, image_file)) as image:
            # 新文件夹的名称
            new_folder_name = f"image_{folder_count}"
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
file_path = "D:/TVCG2/FlaskServer/FlaskServer/BPAL/test1.zip" # 前端上传的zip路径
WSI_path = './WSis' # 解压之后的路径
      
upzip_files = processed.unzip_data(file_path, WSI_path) # 解压
      
image_path, mask_path = upzip_files
output_base_dir = "D:/TVCG2/Visual/public/data/init_data_image_new"
thumbnails_path = 'D:/TVCG2/FlaskServer/FlaskServer/BPAL/save/init_data_image' # 切片保存数据——展示要的image_1、image_2文件夹那个
output_path = 'D:/TVCG2/FlaskServer/FlaskServer/BPAL/save/init_data' # 训练需要的文件夹
processed.process_data(image_path, mask_path,thumbnails_path, output_path)