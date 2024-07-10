import pandas as pd
import numpy as np
import zipfile
import os 
from PIL import Image
import torch
from tqdm import tqdm
from torchvision import transforms
import shutil

"""
unzip_data
input:
    - file_path: str
    - WSI_path: str
output:
    - upzip_files: list
"""
def unzip_data(file_path,WSI_path,WSI_path_2):
    # 解压压缩包
    with zipfile.ZipFile(file_path, 'r') as z:
        z.extractall()
    if not os.path.exists(WSI_path):
        os.makedirs(WSI_path)
    if not os.path.exists(WSI_path_2):
        os.makedirs(WSI_path_2)
    upzip_files = []
    for root, dirs, files in os.walk('./'):
        if (root.endswith('images') ) and len(root)>8:
            # 把 \\换成/
            #root = root.replace('\\', '/')
            # 改名为image_1.png
            files = sorted(files)
            for i, file in enumerate(files):
                os.rename(os.path.join(root, file), os.path.join(root, f'{i+1}.png'))
                # 复制到WSI_path
                shutil.copy(os.path.join(root, f'{i+1}.png'), os.path.join(WSI_path, f'{i+1}.png'))
                shutil.copy(os.path.join(root, f'{i+1}.png'), os.path.join(WSI_path_2, f'{i}.png'))



            upzip_files.append(root)
        if (root.endswith('masks') ) and len(root)>8:
            # 把 \\换成/
            #root = root.replace('\\', '/')
            # 改名为image_1.png
            upzip_files.append(root)
            files = sorted(files)
            for i, file in enumerate(files):
                os.rename(os.path.join(root, file), os.path.join(root, f'{i+1}.png'))

    return upzip_files




"""
thumbnails_deal: slice images and masks into smaller images and save them to corresponding directories
input:
    - image_path: str
    - mask_path: str
    - output_path: str
    - slice_size : int
output:
    - None
"""
def thumbnails_deal(image_path, mask_path, output_path, slice_size):

    # Function to save images and masks to corresponding directories
    def save_images_and_masks(image_slice, mask_slice, output_folder, filename):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        if not os.path.exists(output_folder.replace('images', 'masks')):
            os.makedirs(output_folder.replace('images', 'masks'), exist_ok=True)     
        image_slice.save(os.path.join(output_folder, filename))
        mask_slice.save(os.path.join(output_folder.replace('images', 'masks'), filename))

    image_files = sorted(os.listdir(image_path))
    mask_files = sorted(os.listdir(mask_path))
    # print(image_files, mask_files)
    if len(image_files) != len(mask_files):
        print("Error: The number of images in 'image' folder is different from the number of images in 'mask' folder.")
        exit()

    labels_dict = {1: '1', 2: '2', 3: '3'}
    label1 = 0
    label2 = 0
    label3 = 0
    for i, (image_file, mask_file) in tqdm(enumerate(zip(image_files, mask_files))):
        image_path2 = os.path.join(image_path, image_file)
        mask_path2 = os.path.join(mask_path, mask_file)
        image = Image.open(image_path2)
        mask = Image.open(mask_path2)

        if image.size != mask.size:
            print(f"Error: Image size doesn't match mask size for image {image_file}. Skipping this image.")
            continue

        rows = image.size[1] // slice_size
        cols = image.size[0] // slice_size

        for row in range(rows):
            for col in range(cols):
                left = col * slice_size
                upper = row * slice_size
                right = left + slice_size
                lower = upper + slice_size

                image_slice = image.crop((left, upper, right, lower))
                mask_slice = mask.crop((left, upper, right, lower))

                # Calculate label using label_deal function
                label = get_label(np.array(mask_slice))
             
                if label == 1:
                    label1+=1
                elif label == 2:
                    label2+=1
                else:
                    label3+=3

                # Save image and mask slices with labels
                slice_name = f'x_{row}_y_{col}_{i}_{labels_dict[label]}.png'
                save_images_and_masks(image_slice, mask_slice, os.path.join(output_path, f'image_{i+1}'), slice_name)
    print(f"Label 1: {label1}, Label 2: {label2}, Label 3: {label3}")

def get_label(mask):
    # 计算像素值为1的个数
    cancer_act_rate = [0.0, 0.15, 0.65, 1.0]
    interval=0.05
    num_1 = np.sum(mask == 1)
    area_ratio = num_1 / (224*224)
    if area_ratio > cancer_act_rate[0] and area_ratio < cancer_act_rate[1]:
        label = 1  # no cancer
    elif area_ratio > cancer_act_rate[1]+interval and area_ratio < cancer_act_rate[2]:
        label = 2  # cancer
    elif area_ratio > cancer_act_rate[2]+interval and area_ratio < cancer_act_rate[3]:
        label = 3  # more cancer
    else :
        label = 1 # 无效label
    return label

"""
npy_data_init_percent : process data and save them to npy files
input:
    - img_path: str
    - save_path: str
    - label_rate: float
    - test_rate: float
    - size: int
    - img_num: int
output:
    - None
"""
def npy_data_init_percent(
        img_path,  # /init_data_image
        save_path,  # /init_data
        label_rate=0.5,
        test_rate=0.2,
        size=224,
        img_num=0,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # imglen是wsi图的个数
    train_dataset = []
    labeled_set = []
    unlabeled_set = []
    test_dataset = []
    files = []
    ts_files=[]


    transformations = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    global_index = 0
    for imgid in tqdm(range(img_num)):
        files_path = []
        path = os.path.join(img_path, 'image_'+str(imgid+1))

        for file_name in os.listdir(path):
            files_path.append(file_name)

        patch_num = len(files_path)
        #
        rand_init = torch.randperm(patch_num)
        labeled_index = rand_init[:int(label_rate*patch_num)]
        unlabeled_index = rand_init[int(
            label_rate*patch_num): int((1-test_rate)*patch_num)]
        
        
        # labeled and unlabeled
        for patch_index in rand_init[:int((1-test_rate)*patch_num)]:
            files.append((imgid, patch_index, files_path[patch_index]))
            image = Image.open(os.path.join(path, files_path[patch_index]))
            new_data = Image.fromarray(np.uint8(image)).convert('RGB')
            img = transformations(new_data)
            label = int(files_path[patch_index][-5])
            train_dataset.append(
                (img, label, patch_index, imgid, global_index))
            if patch_index in labeled_index:
                labeled_set.append(
                    (img, label, patch_index, imgid, global_index))

            elif patch_index in unlabeled_index:
                unlabeled_set.append(
                    (img, label, patch_index, imgid, global_index))

            global_index += 1
        
        
        # test
        for patch_index in rand_init[int((1-test_rate)*patch_num):]:
            ts_files.append((imgid, patch_index, files_path[patch_index]))
            image = Image.open(os.path.join(path, files_path[patch_index]))
            new_data = Image.fromarray(np.uint8(image)).convert('RGB')
            img = transformations(new_data)
            label = int(files_path[patch_index][-5])
            test_dataset.append((img, label, patch_index, imgid, 0))

    train_dataset = np.array(train_dataset)
    labeled_set = np.array(labeled_set)
    unlabeled_set = np.array(unlabeled_set)
    test_dataset = np.array(test_dataset)
    
    print(len(train_dataset), len(labeled_set), len(
        unlabeled_set), len(test_dataset), len(files))

    #
    np.save(os.path.join(save_path, "patch_file.npy"), np.array(files))
    np.save(os.path.join(save_path, "ts_patch_file.npy"), np.array(ts_files))

    np.save(os.path.join(save_path, "train_dataset.npy"), train_dataset)
    np.save(os.path.join(save_path, "init_labeled_set.npy"), labeled_set)
    np.save(os.path.join(save_path, "init_unlabeled_set.npy"), unlabeled_set)
    np.save(os.path.join(save_path, "test_dataset.npy"), test_dataset)

"""
npy_data_init_bk : process data and save them to npy files
input:
    - img_path: str
    - save_path: str
    - size: int
    - img_num: int
output:
    - None
"""
def npy_data_init_bk(
        img_path,  # /init_data_image
        save_path,  # /init_data
        size=224,
        img_num=0,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # imglen是wsi图的个数
    bk_files=[]
    bk_data=[]

    transformations = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    global_index = 0
    for imgid in tqdm(range(img_num)):
        bk_files_path=[]
        path = os.path.join(img_path, 'image_'+str(imgid+1))

        for file_name in os.listdir(path):
            bk_files_path.append(file_name)
        #print(imgid,len(bk_files_path))
        # bk
        for patch_index in range(len(bk_files_path)):
            bk_files.append((imgid, patch_index, bk_files_path[patch_index]))
            image = Image.open(os.path.join(path, bk_files_path[patch_index]))
            new_data = Image.fromarray(np.uint8(image)).convert('RGB')
            img = transformations(new_data)
            label = int(bk_files_path[patch_index][-5])
            bk_data.append(
                (img, label, patch_index, imgid, global_index))
            global_index +=1
    
    bk_data = np.array(bk_data)
    bk_files = np.array(bk_files)

    print(len(bk_data),len(bk_files))
    np.save(os.path.join(save_path, "bk_dataset.npy"), bk_data)
    np.save(os.path.join(save_path, "bk_patch_file.npy"), bk_files)

"""
process_data: process data
input: 
    - image_path: str
    - mask_path: str
    - thumbnails_path: str
    - output_path: str
output:
    - None
"""
def process_data(image_path, mask_path, thumbnails_path,output_path):
  
    # slice images and masks into smaller images and save them to corresponding directories
    print('-'*10,'begin to slice images and masks into smaller images and save them to corresponding directories','-'*10)
    
    thumbnails_deal(image_path, mask_path, thumbnails_path, 224)
    save_img_path = output_path
    img_path = thumbnails_path
    num = len(os.listdir(image_path))
    # process data and save them to npy files

    
    print('-'*10,'begin to process data and save them to npy files','-'*10)
    npy_data_init_percent(img_path=img_path,save_path=save_img_path,img_num=num,label_rate=0.7,test_rate=0.2)
    npy_data_init_bk(img_path=img_path,save_path=save_img_path,img_num=num)
    # 
    print('-'*10,'data processing is completed','-'*10)


 

def test_process_data():
    # 解压数据
    file_path = './test2.zip' # 前端上传的zip路径
    WSI_path = './WSis' # 解压之后的路径
    WSI_path_2 = 'D:/TVCG2/Visual/public/data/WSIs'
    upzip_files = unzip_data(file_path, WSI_path) # 解压
    upzip_files = unzip_data(file_path, WSI_path_2)
    # print(upzip_files)
    # 处理数据
    image_path, mask_path = upzip_files
    WSI_path = './WSis'
    thumbnails_path = '../save/init_data_image' # 切片保存数据——展示要的image_1、image_2文件夹那个
    output_path = '../save/init_data' # 训练需要的文件夹
    process_data(image_path, mask_path,thumbnails_path, output_path)

# if __name__ == "__main__":
#     test_process_data()