import json
import logging
import math
import os
import cv2
import csv
import time
import random
import numpy as np

from tqdm import  tqdm
from PIL import Image
from contextlib import suppress
from itertools import chain
from functools import partial
from sklearn.metrics import roc_auc_score,matthews_corrcoef,f1_score,accuracy_score


import torch
import torch.nn.functional as F
from torch import nn


def get_sort_eachclass(metric_list,n_class):
    metric_5=[]
    metric_95=[]
    metric_mean=[]
    for i in range(n_class):
        sorted_metric_list = sorted(metric_list,key=lambda x:x[i])
        metric_5.append(sorted_metric_list[50][i])
        metric_95.append(sorted_metric_list[950][i])
        metric_mean.append(np.mean(np.array(sorted_metric_list),axis=0)[i])
    mean_metric_5 = np.mean(np.array(metric_5))
    metric_5.append(mean_metric_5)
    mean_metric_95 = np.mean(np.array(metric_95))
    metric_95.append(mean_metric_95)
    mean_metric_mean = np.mean(np.array(metric_mean))
    metric_mean.append(mean_metric_mean)
    return metric_5,metric_95,metric_mean


def compute_AUCs(gt, pred, n_class):
    AUROCs = []
    AUROCs.append('AUC')
    gt_np = gt 
    pred_np = pred 
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    mean_auc = np.mean(np.array(AUROCs[1:]))
    AUROCs.append(mean_auc)
    return AUROCs

def compute_F1s_threshold(gt, pred,threshold,n_class):
    gt_np = gt 
    pred_np = pred 
    
    F1s = []
    F1s.append('F1s')
    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>=threshold[i]]=1
        pred_np[:,i][pred_np[:,i]<threshold[i]]=0
        F1s.append(f1_score(gt_np[:, i], pred_np[:, i],average='binary'))
    mean_f1 = np.mean(np.array(F1s[1:]))
    F1s.append(mean_f1)
    return F1s

def compute_Accs_threshold(gt, pred,threshold,n_class):
    gt_np = gt 
    pred_np = pred 
    
    Accs = []
    Accs.append('Accs')
    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>=threshold[i]]=1
        pred_np[:,i][pred_np[:,i]<threshold[i]]=0
        Accs.append(accuracy_score(gt_np[:, i], pred_np[:, i]))
    mean_accs = np.mean(np.array(Accs[1:]))
    Accs.append(mean_accs)
    return Accs

def compute_mccs(gt, pred, n_class):
    gt_np = gt 
    pred_np = pred 
    select_best_thresholds = []

    for i in range(n_class):
        select_best_threshold_i = 0.0
        best_mcc_i = 0.0
        for threshold_idx in range(len(pred)):
            pred_np_ = pred_np.copy()
            thresholds = pred[threshold_idx]
            pred_np_[:,i][pred_np_[:,i]>=thresholds[i]]=1
            pred_np_[:,i][pred_np_[:,i]<thresholds[i]]=0
            mcc = matthews_corrcoef(gt_np[:, i], pred_np_[:, i])
            if mcc > best_mcc_i:
                best_mcc_i = mcc
                select_best_threshold_i = thresholds[i]
        select_best_thresholds.append(select_best_threshold_i)
            
    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>= select_best_thresholds[i]]=1
        pred_np[:,i][pred_np[:,i]< select_best_thresholds[i]]=0
    mccs = []
    mccs.append('mccs')
    for i in range(n_class):
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    mean_mcc = np.mean(np.array(mccs[1:]))
    mccs.append(mean_mcc)
    return mccs,select_best_thresholds


def get_text_features(model,text_list,tokenizer,device,max_length):
    text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    text_features = model.encode_text(text_token)
    return text_features


def test(model,image_encoder, text_encoder, tokenizer, data_loader,device,save_result_path,args,text_list,dist_csv_col):
    save_result_csvpath = os.path.join(save_result_path,'result.csv')
    f_result = open(save_result_csvpath,'w+',newline='')
    wf_result = csv.writer(f_result)
    wf_result.writerow(dist_csv_col)
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)

    model.eval()
    image_encoder.eval()
    text_encoder.eval()

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in tqdm(enumerate(data_loader)):
        image = sample['image'].to(device) 
        label = sample['label'].float().to(device) 
        
        gt = torch.cat((gt, label), 0)

        with torch.no_grad():
            image_features,_ = image_encoder(image)
            
            if args.add_dataset:
                pred_class,_ = model(image_features,text_features,args)
                pred = torch.cat((pred, pred_class[:,:,0]), 0)
            else:
                pred_class = model(image_features,text_features)
                if args.asl or args.bce:
                    pred_class = torch.sigmoid(pred_class) 
                    pred = torch.cat((pred, pred_class[:,:,0]), 0)
                else:
                    pred_class = torch.softmax(pred_class, dim=-1)
                    pred = torch.cat((pred, pred_class[:,:,1]), 0)

    array_gt = gt.cpu().numpy()
    array_pred = pred.cpu().numpy()
    np.save(os.path.join(save_result_path,'gt.npy'),array_gt)
    np.save(os.path.join(save_result_path,'pred.npy'),array_pred)

    n_class = array_gt.shape[1]
    AUROCs = compute_AUCs(array_gt, array_pred,n_class)

    wf_result.writerow(AUROCs)
    f_result.close()

    

   
