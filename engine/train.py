import json
import logging
import math
import os
import cv2
import time
import numpy as np
from torch.distributed import ReduceOp
import random

from PIL import Image
from contextlib import suppress
from itertools import chain
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score
import contextlib


import torch
import torch.nn.functional as F
from torch import nn

from factory import utils
from factory.loss import ClipLoss

try:
    import wandb
except ImportError:
    wandb = None

def moe_cl_loss(fea, label, tau=1.):
        batch_size = fea.shape[0]
        fea = F.normalize(fea)
        sim = fea.mm(fea.t())  

        sim = (sim / tau).exp()
        label = label.unsqueeze(1).repeat(1, batch_size)
        loss = []
        sim = sim - sim.diag().diag()
        for i in range(batch_size):
            for j in range(batch_size):
                if label[j, i] == label[i, i]:
                    if j != i:
                        loss_ = -(sim[j, i] / sim[:, i].sum()).log()
                        loss.append(loss_)
        loss = torch.stack(loss).mean()
        return loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossAdd(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLossAdd, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

def Shuffle_Batch_Data(data_in):
    '''
    打乱一个batch的数据
    '''
    len_total = len(data_in)
    idx_list = list(range(len_total))
    random.shuffle(idx_list)
    return data_in[idx_list]

def Combine_AmplitudeANDPhase(amp, phe):
    return torch.mul(amp, torch.exp(1j*phe))

def mixup_data(x, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    y = Shuffle_Batch_Data(x)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha) # beta分布
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda() # 随机打乱
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :] # 按照比例混合
    y_a, y_b = y, y[index]
    return mixed_x

def FFT2_Amp_MixUp(data_original, data_aug, lamda):
    '''
    将fft_data_original和fft_data_aug按照lamda系数对幅值进行对应的扰动
    相位信息以fft_data_original为准
    '''
    # fft操作
    fft_data_original = torch.fft.fft2(data_original)
    fft_data_aug = torch.fft.fft2(data_aug)
    
    aug_amp = lamda*torch.abs(fft_data_original) + (1-lamda)*torch.abs(fft_data_aug)
    fft_mixup_data = torch.mul(aug_amp, torch.exp(1j*torch.angle(fft_data_original)))
    return torch.real(torch.fft.ifft2(fft_mixup_data))

def fourier_aug(batch_data, p=0.5):
    batch_x = batch_data
    batch_y = Shuffle_Batch_Data(batch_data)
    apply_p = np.random.rand()
    if apply_p<=p:
        lamda_vector = np.random.rand(batch_x.size(0))
        for i in range(batch_x.size(0)):
            batch_x[i] = FFT2_Amp_MixUp(batch_x[i], batch_y[i], lamda_vector[i])
        return batch_x
    else:
        return batch_x
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_text_features(model,text_list,tokenizer,device,max_length):
   
    text_token = tokenizer(list(text_list),add_special_tokens=True, padding='max_length', truncation=True, max_length= max_length, return_tensors="pt").to(device=device)
    text_features = model.encode_text(text_token)
    return text_features


def train(model, image_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer):
    clip_loss = ClipLoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    if args.add_dataset:
        ASL_loss = AsymmetricLossAdd(gamma_neg=6, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    else:
        ASL_loss = AsymmetricLoss(gamma_neg=6, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_ce_m = AverageMeter()
    loss_ce_image_m = AverageMeter()
    loss_ce_text_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    model.train()  
    image_encoder.train()  
    text_encoder.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce_image', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    if args.use_entity_features:
        metric_logger.add_meter('loss_ce_text', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_clip', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)
    num_batches_per_epoch = data_loader.num_batches
    sample_digits = math.ceil(math.log(data_loader.num_samples + 1, 10))

    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.fourier:
            image = fourier_aug(sample['image'].to(device))
        else:
            image = sample['image'].to(device)  
        label = sample['label'].long().to(device)

        if args.ignore_index:
            pass
        else:
            label[label==-1]=0
        entity = sample['entity']

        if args.add_dataset:
            dataset_label = sample['label_dataset']

        data_time_m.update(time.time() - end)

        optimizer.zero_grad()

        if args.add_dataset:
            text_list = ['normal', 'pleural effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis',  'tube', 'consolidation','enlarged cardiomediastinum','tip', 'pneumonia','line','cardiomegaly', 'fracture','calcification',
            'device','engorgement',  'nodule', 'wire',  'pacemaker', 'pleural thicken', 'marking', 'scar', 'hyperinflate', 'blunt',  'collapse', 'emphysema', 'aerate', 'mass','infiltration', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'lesion', 'hardware', 'dilation',  'aspiration',
            'fibrosis',	'No Finding', 'Pleural Other', 'Support Devices', 'Aortic enlargement',
            'Clavicle fracture', 'Enlarged PA', 'ILD', 'Lung cavity', 'Lung cyst', 'Mediastinal shift',	
            'Nodule/Mass', 'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Tuberculosis',
            'Other diseases']

        else:

            text_list = ['normal', 'pleural effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis',  'tube', 'consolidation','enlarged cardiomediastinum','tip', 'pneumonia','line','cardiomegaly', 'fracture','calcification',
            'device','engorgement',  'nodule', 'wire',  'pacemaker', 'pleural thicken', 'marking', 'scar', 'hyperinflate', 'blunt',  'collapse', 'emphysema', 'aerate', 'mass','infiltration', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'lesion', 'hardware', 'dilation',  'aspiration']
        
        
        text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
        entity_features = get_text_features(text_encoder,entity,tokenizer,device,max_length=args.max_length)

        image_features,image_features_pool = image_encoder(image)
        if args.add_dataset:
            pred_class_image, moe_img = model(image_features,text_features,args)
        else:
            pred_class_image = model(image_features,text_features)


        if args.bce or args.asl:
            label = label.float()

        label_mask = (label != -1).squeeze()



        if args.add_dataset:
            loss_moe_img = moe_cl_loss(moe_img, dataset_label)

            if args.asl:
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask]  
                loss_ce_image = ASL_loss(pred_class_image.view(-1,1),label_image.view(-1,1))
            elif args.bce:
                pred_class_image = pred_class_image[label_mask]
                label_image = label[label_mask] 
                loss_ce_image = F.binary_cross_entropy(pred_class_image.view(-1,1),label_image.view(-1,1))
        else:
            if args.asl:
                loss_ce_image = ASL_loss(pred_class_image.view(-1,1),label.view(-1,1))
            elif args.bce:
                loss_ce_image = F.binary_cross_entropy_with_logits(pred_class_image.view(-1,1),label.view(-1,1)) 
            else:
                loss_ce_image = ce_loss(pred_class_image.view(-1,2),label.view(-1)) 

        if args.use_entity_features:
            if args.add_dataset:
                pred_class_text, moe_txt = model(entity_features.unsqueeze(1),text_features,args)
                loss_moe_txt = moe_cl_loss(moe_txt, dataset_label)
            else:
                pred_class_text = model(entity_features.unsqueeze(1),text_features)

            if args.add_dataset:
                if args.asl:
                    pred_class_text = pred_class_text[label_mask]
                    label_text = label[label_mask]  
                    loss_ce_text = ASL_loss(pred_class_text.view(-1,1),label_text.view(-1,1))
                    
                elif args.bce:
                    pred_class_text = pred_class_text[label_mask]
                    label_text = label[label_mask] 
                    loss_ce_text = F.binary_cross_entropy(pred_class_text.view(-1,1),label_text.view(-1,1))

            else:
                if args.asl:
                    loss_ce_text = ASL_loss(pred_class_text.view(-1,1),label.view(-1,1))
                elif args.bce:
                    loss_ce_text = F.binary_cross_entropy_with_logits(pred_class_text.view(-1,1),label.view(-1,1)) 
                else:
                    loss_ce_text = ce_loss(pred_class_text.view(-1,2),label.view(-1))

            loss_ce = loss_ce_image + loss_ce_text
            if args.add_dataset:
                loss_moe = loss_moe_img + loss_moe_txt

        else:
            loss_ce = loss_ce_image
            if args.add_dataset:
                loss_moe = loss_moe_img


        loss_clip = clip_loss(image_features_pool,entity_features)
        if args.add_dataset:
            loss = loss_ce + loss_clip * args.loss_ratio + args.moe_ratio * loss_moe
        else:
            loss = loss_ce + loss_clip * args.loss_ratio
        

        loss.backward()
        optimizer.step() 
    
        writer.add_scalar('loss/loss', loss, scalar_step)
        writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        writer.add_scalar('loss/loss_ce_image', loss_ce_image, scalar_step)
        if args.use_entity_features:
            writer.add_scalar('loss/loss_ce_text', loss_ce_text, scalar_step)
        writer.add_scalar('loss/loss_clip', loss_clip, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_ce_image=loss_ce_image.item())
        if args.use_entity_features:
            metric_logger.update(loss_ce_text=loss_ce_text.item())
        metric_logger.update(loss_clip=loss_clip.item())


        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        metric_logger.update(lr = scheduler._get_lr(epoch)[0])

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if i % 100 == 0:
            batch_size = len(image)
            num_samples = batch_count * batch_size
            samples_per_epoch = data_loader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(loss.item(), batch_size)
            loss_clip_m.update(loss_clip.item(), batch_size)
            loss_ce_m.update(loss_ce.item(), batch_size)
            loss_ce_image_m.update(loss_ce_image.item(), batch_size)
            if args.use_entity_features:
                loss_ce_text_m.update(loss_ce_text.item(), batch_size)
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_ce: {loss_ce_m.val:#.5g} ({loss_ce_m.avg:#.4g}) "
                    f"Loss_ce_image: {loss_ce_image_m.val:#.5g} ({loss_ce_image_m.avg:#.4g}) "
                    f"Loss_ce_text: {loss_ce_text_m.val:#.5g} ({loss_ce_text_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {batch_size/ batch_time_m.val:#g}/s "
                    f"LR: { scheduler._get_lr(epoch)[0]:5f} "
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_ce: {loss_ce_m.val:#.5g} ({loss_ce_m.avg:#.4g}) "
                    f"Loss_ce_image: {loss_ce_image_m.val:#.5g} ({loss_ce_image_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {batch_size/ batch_time_m.val:#g}/s "
                    f"LR: { scheduler._get_lr(epoch)[0]:5f} "
                )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()

def valid_on_chestxray14(model, image_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = ["atelectasis","cardiomegaly","pleural effusion","infiltration","lung mass","lung nodule","pneumonia","pneumothorax","consolidation","edema","emphysema","fibrosis","pleural thicken","hernia"]
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        if args.bce or args.asl:
            label = label.float()

        gt = torch.cat((gt, label), 0)
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)

            if args.add_dataset:
                pred_class,_ = model(image_features,text_features,args)#b,14,2/1
                val_loss = F.binary_cross_entropy(pred_class.view(-1,1),label.view(-1, 1))
                pred = torch.cat((pred, pred_class[:,:,0]), 0)
            else:
                pred_class = model(image_features,text_features)#b,14,2/1
                if args.bce or args.asl:
                    val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
                    pred_class = torch.sigmoid(pred_class)
                    pred = torch.cat((pred, pred_class[:,:,0]), 0)
                else:
                    val_loss = criterion(pred_class.view(-1,2),label.view(-1))
                    pred_class = torch.softmax(pred_class, dim=-1)
                    pred = torch.cat((pred, pred_class[:,:,1]), 0)



            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    metrics = compute_AUCs(gt, pred, n_class = 14)
    AUROC_avg = metrics['mean_auc']
    avg_val_loss = np.array(val_losses).mean()
    return avg_val_loss,AUROC_avg,metrics

def valid_on_cheXpert(model,image_encoder,text_encoder,tokenizer,data_loader, epoch, device, args, config, writer):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'pleural effusion']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        if args.bce or args.asl:
            label = label.float()

        gt = torch.cat((gt, label), 0)
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
           
            # 
            if args.add_dataset:
                pred_class,_ = model(image_features,text_features,args)#b,14,2/1
                val_loss = F.binary_cross_entropy(pred_class.view(-1,1),label.view(-1, 1))
                pred = torch.cat((pred, pred_class[:,:,0]), 0)
            else:
                pred_class = model(image_features,text_features)#b,14,2/1
                if args.bce or args.asl:
                    val_loss = F.binary_cross_entropy_with_logits(pred_class.view(-1,1),label.view(-1, 1))
                    pred_class = torch.sigmoid(pred_class)
                    pred = torch.cat((pred, pred_class[:,:,0]), 0)
                else:
                    val_loss = criterion(pred_class.view(-1,2),label.view(-1))
                    pred_class = torch.softmax(pred_class, dim=-1)
                    pred = torch.cat((pred, pred_class[:,:,1]), 0)
            
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    metrics = compute_AUCs(gt, pred, n_class=5)
    AUROC_avg = metrics['mean_auc']
    avg_val_loss = np.array(val_losses).mean()
    return avg_val_loss,AUROC_avg,metrics

def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    metrics = {}
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    metrics[f"mean_auc"] = np.mean(np.array(AUROCs))
    if n_class == 5:
        metrics[f"auc/class_0"]=AUROCs[0]
        metrics[f"auc/class_1"]=AUROCs[1]
        metrics[f"auc/class_2"]=AUROCs[2]
        metrics[f"auc/class_3"]=AUROCs[3]
        metrics[f"auc/class_4"]=AUROCs[4]
    else:
        metrics[f"auc/class_0"]=AUROCs[0]
        metrics[f"auc/class_1"]=AUROCs[1]
        metrics[f"auc/class_2"]=AUROCs[2]
        metrics[f"auc/class_3"]=AUROCs[3]
        metrics[f"auc/class_4"]=AUROCs[4]
        metrics[f"auc/class_5"]=AUROCs[5]
        metrics[f"auc/class_6"]=AUROCs[6]
        metrics[f"auc/class_7"]=AUROCs[7]
        metrics[f"auc/class_8"]=AUROCs[8]
        metrics[f"auc/class_9"]=AUROCs[9]
        metrics[f"auc/class_10"]=AUROCs[10]
        metrics[f"auc/class_11"]=AUROCs[11]
        metrics[f"auc/class_12"]=AUROCs[12]
        metrics[f"auc/class_13"]=AUROCs[13]
    return metrics
