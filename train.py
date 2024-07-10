import gc
from SPR.spr import spr
from SPR.spr_main import spr_main
from utils.log_config import MyLogger
import os
import torch
from torch.autograd import Variable
from utils.model_utils import *
from utils.serialization import *
from utils.make_dir import make_dir
import numpy as np
import models
import os.path as osp
from torch.cuda.amp import autocast, GradScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from datasets import HDF5Dataset
import pandas as pd
import yaml
from datasets import *
from sklearn.preprocessing import MinMaxScaler
from CAM import *
import cv2
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap
import pynvml
from loss.recall_loss import RecallLoss
from BMM.bmm_main import *
from spr_and_bmm import *


"""
train:
input: 
    - config_path: str, path to the config file
    - 
output:
"""
def train_2(config_path):

    config = read_config(config_path)
    # init logger
    logger = init_logger(config)
    # init dataset
    peso_data = PesoTrain(config, logger)
    peso_data.init_data()
    # ===========================================
    logger.info("Loading cuda....")
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_id']
    logger.info("Start!!!")
    # ===========================================
    print("!!!!!!!!!!!!!!!开始正式数据获取训练!!!!!!!!!!!!!!!")
    # create model
    param_path = './param/BPAL-BCSS.pth.tar'
    model, scaler = init_backbone(config, logger, param_path)
    # train
    train_backbone(model, scaler, peso_data, 0, config, logger)
    for e in range(1, config['max_iteration']+1):
        auc, acc, cm,pre,epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
            train_backbone(
            model, scaler, peso_data, e, config, logger)
        active_learning_train_epoch(
            config, logger, model, scaler, e, peso_data, auc, acc, cm,pre,epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)

# 测试案例
def test_2():
    train_2('./config/config.yaml')
    
# if __name__ == '__main__':
#    test()