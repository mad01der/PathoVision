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
import datetime
#
# CONFIG_PATH = "/home/chenzhongming/TVCG_wsi/config/config.yaml"
CONFIG_PATH = "./config/config.yaml"


def read_config(config_path):
    fs = open(config_path, encoding="UTF-8")
    datas = yaml.load(fs,Loader=yaml.FullLoader)
    return datas


def init_logger(config):
    name = config['dataset']['name']+"_batch_"+str(config["batch_size"])+"_tileSize_"+str(
        config['dataset']["tile_size"])+"_noise_"+str(config["noise_rate"])
    logger = MyLogger(file_name=name, config=config)
    logger.info("="*50)
    logger.info("Init Model Config...")
    logger.info("="*50)
    for k, v in config.items():
        logger.info("%s:%s" % (k, v))
    logger.info("="*50)

    return logger

def init_backbone(config, logger,param_path):
    model = models.create(
        config['model'], num_features=config["num_features"], num_classes=config["dataset"]["num_class"])
    if config['multi_cuda']:
        # model=torch.nn.DataParallel(model)
        logger.warning("multi cuda test")
    model = model.cuda()
    if param_path:
        checkpoint = load_checkpoint(param_path, logger)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    scaler = GradScaler()
    if torch.cuda.is_available():
        logger.info("cuda is available")
    else:
        logger.warning(" is not available!")

    return model, scaler


def train_backbone(network, scaler, dataset, epoch, config, logger):
    if config['optim'] == "SGD":
        optimizer = torch.optim.SGD(
            network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(
            network.parameters(), lr=0.001, weight_decay=5e-4)
        
    if config['loss'] == "CE":
        criterion = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=-1).cuda()
        
    elif config['loss'] == "Recall":
        criterion = RecallLoss(n_classes = 3)

    #
    labeled_loader = torch.utils.data.DataLoader(dataset=Preprocessor(dataset.labeled_set),
                                                 batch_size=config["batch_size"],
                                                 num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=Preprocessor(dataset.test_dataset),
                                              batch_size=config["batch_size"],
                                              num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    bk_loader= torch.utils.data.DataLoader(dataset=Preprocessor(dataset.bk_dataset),
                                              batch_size=config["batch_size"],
                                              num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(dataset=Preprocessor(dataset.train_dataset),
                                                  batch_size=config["batch_size"],
                                                  num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    #
                        
    # next train
    best_auc = 0
    best_acc = 0
    best_cm ,prediction= None,None
    dynamic_epoch = config["pretrained_iteration"]
    for e in range(dynamic_epoch):
        loss = []
        # train models
        network.train()
        for i, (images, labels, _, _, global_idx,_) in enumerate(labeled_loader):
            images = Variable(images).cuda()
            labels = labels.long().cuda()
            optimizer.zero_grad()
            with autocast():
                feature, logits = network(images)
                loss_1 = criterion(logits, labels)
                loss_1 = loss_1.mean()

                loss.append(loss_1.item())

            scaler.scale(loss_1).backward()
            scaler.step(optimizer)
            scaler.update()
            
        
        with torch.no_grad():
            auc_ = evaluate_auc(test_loader, network)
            acc_, cm = evaluate_acc(test_loader, network)
        
        # wandb
        # wandb.log({f"epoch{epoch}_iter_train_loss": np.mean(loss),f"epoch{epoch}_iter_test_auc": auc_,f"epoch{epoch}_iter_test_acc": acc_,f"epoch{epoch}_iter_test_qkw": qkw})
        # logger
        logger.info("[Backbone Training] Resnet train in epoch: %d/%d, train_loss: %f, test_auc: %f, test_acc: %f" % (
            e+1, dynamic_epoch, loss_1.item(), auc_, acc_
        ))
        if auc_ > best_auc:
            # save train
            save_checkpoint({
                'state_dict': network.state_dict(),
                'best_auc': auc_,
            }, True,
                fpath=osp.join(config["save_param_dir"], 'pretrained_resnet.pth.tar' if epoch == 0 else (
                    "epoch_"+str(epoch-1)+'_model_best.pth.tar')),
                logger=logger)
            # predict back
            prediction=predict_bk(bk_loader, network)
            # save best
            best_auc = auc_
            best_acc = acc_
            best_cm = cm 
    if epoch == 0:
        return best_auc, best_acc, best_cm,prediction
    # gc.collect()
    # torch.cuda.empty_cache()
    logger.info("Begin to train BMM model...")
    epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss =\
          bmm_track_training_loss(network, torch.device("cuda" if torch.cuda.is_available() else "cpu") , train_loader=labeled_loader)
    
    return best_auc, best_acc, best_cm,prediction, epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss




def active_learning_train_epoch(config, logger, network, scaler, epoch, input, auc, acc, cm,pre,epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    logger.info("="*20+"active learning for epoch:%d" % (epoch)+"="*20)
    # # 初始化所有数据
    train_loader = torch.utils.data.DataLoader(dataset=Preprocessor(input.train_dataset),
                                               batch_size=config["batch_size"],
                                               num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    bk_loader= torch.utils.data.DataLoader(dataset=Preprocessor(input.bk_dataset),
                                              batch_size=config["batch_size"],
                                              num_workers=config['num_workers'], shuffle=False, pin_memory=True)
    #
    # init param
    checkpoint = load_checkpoint(
        os.path.join(config["save_param_dir"], 'pretrained_resnet.pth.tar' if epoch == 1 else "epoch_"+str(epoch-1)+"_model_best.pth.tar"), logger)
    network.load_state_dict(checkpoint['state_dict'], strict=False)

    fea, _, _, _ = extract_features(
        network, train_loader, logger=logger)
    bk_fea,_,_,_=extract_features(
        network, bk_loader, logger=logger)

    fea_cf = torch.stack(fea).numpy()
    bk_fea_cf = torch.stack(bk_fea).numpy()
    union_fea=np.vstack((fea_cf,bk_fea_cf))
  
    logger.info("PCA algorithm is running...")
    pca = PCA(n_components=config['PCA_components'],
              copy=False, random_state=config['seed'])
    cf=pca.fit_transform(union_fea)
    
    # 
    if config['visual_method'] == 'tsne':
        tsne = TSNE(n_components=2, init='pca', random_state=config['seed'])
        v_cf= tsne.fit_transform(cf)
    elif config['visual_method'] == 'umap':
        up = umap.UMAP(n_components=2, min_dist=0.02, n_neighbors=60)
        v_cf= up.fit_transform(cf)

    v_km = KMeans(n_clusters=config['Kmeans_Visual_cluster'],
                  random_state=config['seed']).fit(v_cf)
    v_km_label = v_km.labels_[:fea_cf.shape[0]]
    bk_v_km_label = v_km.labels_[fea_cf.shape[0]:]
  
    logger.info("Kmeans algorithm is running...")
    # km = DBSCAN(eps=0.5, min_samples=10).fit(cf) 
    km = KMeans(n_clusters=config['Kmeans_cluster'],
                random_state=config['seed']).fit(cf)
    target_label = km.labels_
    logger.info("cluster result:"+str(numCount(target_label)))
    """
    SPR
    """
    logger.info("SPR algorithm is running...")
    spr_clean_set, spr_noise_set, spr_noise_prob = spr_main(network, input ,logger ,config, train_loader) # SPR
    # 把spr_noise_prob取排序之后（由大到小）3%的数据作为噪声数据
    spr_noise_set_idx = [x for _,x in sorted(zip(spr_noise_prob,spr_noise_set),reverse=True)]
    spr_noise_set_idx = spr_noise_set_idx[:int(len(spr_noise_set_idx)*0.03)]
    # spr_clean_set随机取1%的数据作为置信数据
    spr_clean_set = list(spr_clean_set)
    np.random.shuffle(spr_clean_set)
    spr_clean_set = spr_clean_set[:int(len(spr_clean_set)*0.01)]
    # 转换成set
    spr_clean_set = set(spr_clean_set)
    # 对prob进行归一化
    spr_noise_prob = np.array(spr_noise_prob)
    spr_noise_prob = (spr_noise_prob - spr_noise_prob.min()) / (spr_noise_prob.max() - spr_noise_prob.min())

    logger.info(f"SPR detect noise_set num: {len(spr_noise_set_idx)}")

    
    """
    BMM
    """
    logger.info("BMM algorithm is running...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bmm_noise_idx, bmm_noise_prob,bmm_clean_confident_index = BMM_Noise_label( network,device, train_loader,bmm_model,bmm_model_maxLoss, bmm_model_minLoss)
    logger.info(f"BMM detect noise_set num: {len(bmm_noise_idx)}")
    
    if config["methods"] == "BMM":
        noise_idx = bmm_noise_idx
        clean_confident_index = bmm_clean_confident_index
    elif config["methods"] == "SPR":
        noise_idx = spr_noise_set_idx
        clean_confident_index = spr_clean_set
    elif config["methods"] == "Both":
    
        
        print("BMM noise_idx:",len(bmm_noise_idx))
        print("SPR noise_idx:",len(spr_noise_set_idx))
        noise_idx = list(set(bmm_noise_idx).union(set(spr_noise_set_idx)))
        clean_confident_index = list(set(bmm_clean_confident_index).union(set(spr_clean_set)))
        # 保证噪声数据不超过5%
        noise_idx = noise_idx[:int(len(noise_idx)*0.15)]
        # 保证置信数据不超过5%
        clean_confident_index = clean_confident_index[:int(len(clean_confident_index)*0.15)]

    input.reset(noise_idx, clean_confident_index)
    save_iteration_data(config,input, epoch, noise_idx, clean_confident_index, acc, auc,cm,pre, bmm_noise_prob,v_cf,v_km_label,bk_v_km_label,spr_clean_set, spr_noise_set, spr_noise_prob)
    # gradCAM
    if config['grad_save'] == 1 and epoch >= config["max_iteration"]-1:
        logger.info("Grad-CAM images are generated...")
        grad_cam = GradCam(network , target_layer_names=[
                           'base'], img_size=config['dataset']['size'])
        for img, label, pid, imgid, global_idx in input.train_dataset:
            img = np.asarray(img).reshape(
                config['dataset']['size'], config['dataset']['size'], 3)
            img = np.float32(cv2.resize(
                img, (config['dataset']['size'], config['dataset']['size']))) / 255
            pre_img = preprocess_image(img)
            pre_img.required_grad = True
            target_index = None
            mask = grad_cam(pre_img, target_index)
            CAM_save_path = make_dir(os.path.join(config['save_data_dir'], os.path.join(
                'init_data_image', 'image_CAM_'+str(imgid))))
            CAM_save_path = os.path.join(CAM_save_path, str(global_idx)+".png")
            show_cam_on_image(img, mask, CAM_save_path)

def save_iteration_data(
        config,dataset, epoch, noise_index, infor_index, acc, auc,cm,pre,noise_prob,visual_cf,km,bk_km,spr_clean_set, spr_noise_set, spr_noise_prob):
    

    #print("数据保存在"+base_path)
    # 保存BMM详细数据
    if not os.path.exists(os.path.join(config["save_data_dir"], 'BMM')):
        os.makedirs(os.path.join(config["save_data_dir"], 'BMM'))
    np.save(os.path.join(config["save_data_dir"], 'BMM',f"index_noise_{epoch}.npy"), noise_index)
    # 保存noise_prob到config["save_data_dir"]的./index_noise中，保存为bmm_noise_prob.npy
    np.save(os.path.join(config["save_data_dir"],  'BMM',f"noise_prob.npy_{epoch}"), noise_prob)
    # 保存clean_confident_index到config["save_data_dir"]的./index_noise中，保存为bmm_clean_confident_index.npy
    np.save(os.path.join(config["save_data_dir"], 'BMM', f"clean_confident_index_{epoch}.npy"), infor_index)
    
    # 保存SPR详细数据
    if not os.path.exists(os.path.join(config["save_data_dir"], 'SPR')):
        os.makedirs(os.path.join(config["save_data_dir"], 'SPR'))
    np.save(os.path.join(config["save_data_dir"], 'SPR',f"clean_set_{epoch}.npy"), np.array(spr_clean_set))
    np.save(os.path.join(config["save_data_dir"], 'SPR',f"noise_set_{epoch}.npy"), np.array(spr_noise_set))
    np.save(os.path.join(config["save_data_dir"], 'SPR',f"noise_prob_{epoch}.npy"), spr_noise_prob)

    # sample_data：样本数据
    for i, (_, label, pid, imgid, global_idx) in enumerate(dataset.train_dataset):
        dataset.sample_data['scatter_x'][global_idx] = visual_cf[global_idx][0]
        dataset.sample_data['scatter_y'][global_idx] = visual_cf[global_idx][1]
        #dataset.sample_data['grade'][global_idx] = grade_label[global_idx]
        #dataset.sample_data['o2u'][global_idx] = o2u_score[global_idx]
        dataset.sample_data['kmeans_label'][global_idx] = km[global_idx]

        dataset.sample_data['bmm_num'][global_idx].append(
            float(noise_prob[global_idx]))
        # dataset.sample_data['spr_num'][global_idx].append(
        #     float(spr_noise_prob[global_idx]))
        
        dataset.sample_data['heat_score'][global_idx] = 0
        dataset.sample_data['bmm'][global_idx] = float(noise_prob[global_idx])
        #dataset.sample_data['spr'][global_idx] = float(spr_noise_prob[global_idx])
        spr_num = 0.0
        if global_idx in spr_clean_set:
            dataset.sample_data['spr'][global_idx] = 0.0
        elif global_idx in spr_noise_set:
            spr_num = float(spr_noise_prob[list(spr_noise_set).index(global_idx)])
            dataset.sample_data['spr'][global_idx] = spr_num
        else:
            dataset.sample_data['spr'][global_idx] = 0.0
        dataset.sample_data['spr_num'][global_idx].append(
            spr_num)

        if global_idx in noise_index:
            dataset.sample_data['noise'][global_idx] = 1.0
        elif global_idx in infor_index:
            dataset.sample_data['noise'][global_idx] = 0.0
            
        # update WSI data
        #dataset.WSI_Data['spr'][imgid] = float(spr_noise_prob[global_idx])
        dataset.WSI_Data['bmm'][imgid] = float(noise_prob[global_idx])
        dataset.WSI_Data['spr'][imgid] += spr_num

    # updata back data
    for i, (_, label, pid, imgid, global_idx) in enumerate(dataset.bk_dataset):
        dataset.bk_data['class'][global_idx] = int(pre[global_idx])
        dataset.bk_data['kmeans_label'][global_idx] = bk_km[global_idx]
        

    # epoch_Data：每一轮的数据
    dataset.epoch_Data['epoch'].append(epoch)
    dataset.epoch_Data['acc'].append(acc)
    dataset.epoch_Data['auc'].append(auc)
    dataset.epoch_Data['labeled'].append(len(dataset.labeled_set))
    dataset.epoch_Data['unlabeled'].append(len(dataset.unlabeled_set))
    dataset.epoch_Data['noise_in_labeled'].append(len(noise_index))
    dataset.epoch_Data['infor_in_unlabled'].append(len(infor_index))

    # name = config['dataset']['name']+"_batch_"+str(config["batch_size"])+"_tileSize_"+str(
    #     config['dataset']["tile_size"])+"_noise_"+str(config["noise_rate"])
    name = config['methods']+"_"+config['loss']+"_"+config['dataset']['name']+"_batch_"+str(config["batch_size"])+"_tileSize_"+str(
        config['dataset']["tile_size"])+"_noise_"+str(config["noise_rate"])
    base_path = os.path.join(config['save_data_dir'], f"save_data_{epoch}/"+name)
    base_path = make_dir(base_path)

    # save to csv
    pd.DataFrame(dataset.sample_data).to_csv(
        os.path.join(base_path, "sample_data.csv"))
    pd.DataFrame(dataset.bk_data).to_csv(
        os.path.join(base_path, "bk_data.csv"))
    pd.DataFrame(dataset.epoch_Data).to_csv(
        os.path.join(base_path, "epoch_Data.csv"))
    pd.DataFrame(dataset.WSI_Data).to_csv(
        os.path.join(base_path, "WSI_Data.csv"))
    pd.DataFrame(cm).to_csv(
        os.path.join(base_path, "confusion.csv"))
    
    
# if __name__ == '__main__':
   
#   # read config
#     config = read_config(CONFIG_PATH)
#     # init logger
#     logger = init_logger(config)
#     # init dataset
#     peso_data = PesoTrain(config, logger)
#     peso_data.init_data()

#     # ===========================================
#     logger.info("Loading cuda....")
#     # set gpu
#     os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_id']
#     logger.info("Start!!!")
#     # ===========================================
    

#     # start a new wandb run to track this script
#     nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     # wandb.init(
#     #     # set the wandb project where this run will be logged
#     #     project="PANDA-AL-Project",
#     #     name = nowtime,
#     #     # track hyperparameters and run metadata
#     #     config={
#     #     "learning_rate": 0.01,
#     #     "architecture": "Renet50",
#     #     "dataset": "PANDA",
#     #     "epochs": config['max_iteration'],
#     #     "loss_function": config['loss'],
#     #     "batch_size": config["batch_size"],
#     #     "method":config["methods"],
#     #     "noise_rate":config["noise_rate"],
#     #     },
#     # )
#     # ===========================================
#     print("!!!!!!!!!!!!!!!开始正式数据获取训练!!!!!!!!!!!!!!!")
#     # create model
#     model, scaler = init_backbone(config, logger)
#     # train
#     train_backbone(model, scaler, peso_data, 0, config, logger)
#     # AL
#     for e in range(1, config['max_iteration']+1):
#         auc, acc, cm,pre,epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
#             train_backbone(
#             model, scaler, peso_data, e, config, logger)
#         # wandb.log({"acc": acc, "auc": auc})

#         # train_loader = torch.utils.data.DataLoader(dataset=Preprocessor(peso_data.labeled_set),
#         #                                          batch_size=config["batch_size"],
#         #                                          num_workers=config['num_workers'], shuffle=False, pin_memory=True)
        
#         # epoch 1
#         active_learning_train_epoch(
#             config, logger, model, scaler, e, peso_data, auc, acc, cm,pre,epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)

        
