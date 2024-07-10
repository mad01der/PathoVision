import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
# from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm


"""
cut_wsi_into_tiles : function to cut wsi into tiles
    input : 
        image (.png/.jpg)
    output : images of tiles (png)
"""
def cut_wsi_into_tiles(image, output_dir, slice_size=224):
    # 打开图片和对应的mask
    image = Image.open(image)
    # 计算切片的行数和列数
    rows = image.size[1] // slice_size
    cols = image.size[0] // slice_size
    images = []
    # 逐个切片并保存
    for row in tqdm(range(rows)):
        for col in range(cols):
            # 计算切片的区域
            left = col * slice_size
            upper = row * slice_size
            right = left + slice_size
            lower = upper + slice_size

            # 裁剪图片和对应的mask
            image_slice = image.crop((left, upper, right, lower))
    
            # 保存切片图片和对应的mask
            slice_name = f'x_{row}_y_{col}.png'
            image_slice.save(os.path.join(output_dir, slice_name))
            images.append(slice_name)
    return images


"""
cut_tif_wsi_into_tiles : function to cut wsi into tiles
    input : 
        wsi_img (.tiff/.tif)
    output : images of tiles (png)
"""

def cut_tif_wsi_into_tiles(wsi_img, output_dir, slice_size=224):
    # 打开WSI图像
    wsi_slide = OpenSlide(wsi_img)
    
    images = []
    
    # 计算切片的行数和列数
    rows = wsi_slide.level_dimensions[0][1] // slice_size
    cols = wsi_slide.level_dimensions[0][0] // slice_size
    
    # 逐个切片并保存
    for row in tqdm(range(rows)):
        for col in range(cols):
            # 计算切片的区域
            left = col * slice_size
            upper = row * slice_size
            right = min(left + slice_size, wsi_slide.level_dimensions[0][0])
            lower = min(upper + slice_size, wsi_slide.level_dimensions[0][1])
            # 最大的level开始切分
            level = 0
            # 读取切片图像数据
            image_data = wsi_slide.read_region((left, upper), level, (right-left, lower-upper))

            # 最大的level开始切分
            # 保存切片图像
            slice_name = f'x_{row}_y_{col}.png'
            image_data.save(os.path.join(output_dir, slice_name))
            images.append(slice_name)
    
    # 关闭WSI图像
    wsi_slide.close()
    
    return images
# if __name__ == "__main__":
#     # 测试cut_wsi_into_tiles
#     # wsi_img ='./test1.tiff'
#     # output_dir = 'thumbnails/images1'
#     # images = cut_wsi_into_tiles(wsi_img, output_dir, slice_size=224, overlap=0)
#     # print(len(images))

#     # 测试cut_other_wsi_into_tiles
#     image = "./test2.png"
#     output_dir = 'thumbnails/images'
#     images = cut_other_wsi_into_tiles(image, output_dir, slice_size=224, overlap=0)
#     print(len(images))