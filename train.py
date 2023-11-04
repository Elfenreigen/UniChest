import argparse
import os

import logging
# import ruamel.yaml as yaml
import yaml
import numpy as np
import random
import time
import datetime
import json
import math
from pathlib import Path
from functools import partial
from sklearn.metrics import roc_auc_score

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from transformers import AutoModel,BertConfig,AutoTokenizer

from factory import utils
from scheduler import create_scheduler
from optim import create_optimizer
from engine.train import train,valid_on_cheXpert,valid_on_chestxray14
from models.clip_tqn import CLP_clinical,ModelRes,TQN_Model,TQN_Model_Add,ModelDense,CLP_clinical2
from models.tokenization_bert import BertTokenizer
from dataset.dataset_entity import MIMIC_Dataset,Mergetrain_Dataset, Chestxray14_Dataset,CheXpert_Dataset

import socket
from io import BytesIO


def main(args, config):
    torch.cuda.current_device()
    torch.cuda._initialized = True
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')
    
    utils.init_distributed_mode(args)
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    print('sampler_rank',sampler_rank,'num_tasks',num_tasks)

    #### Dataset #### 
    print("Creating dataset")
    
    if args.add_dataset == True:
        train_dataset = Mergetrain_Dataset(config['train_entity_file'], config['train_fg_query_file_v1'], config['mrsty_file'],config['image_res'], args)

    else:
        train_dataset = MIMIC_Dataset(config['train_entity_file'], config['train_fg_query_file_v1'], config['mrsty_file'],config['image_res'], args)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=8,
            pin_memory=True,
            sampler=train_sampler, 
            collate_fn=None,
            worker_init_fn=utils.seed_worker,
            drop_last=True,
        )    
    train_dataloader.num_samples = len(train_dataset)
    train_dataloader.num_batches = len(train_dataloader)  

    val_dataset = Chestxray14_Dataset(config['chestxray_valid_file'],config['image_res'])
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    val_dataloader =DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=8,
            pin_memory=True,
            sampler=val_sampler,
            collate_fn=None,
            worker_init_fn=utils.seed_worker,
            drop_last=True,
        )     
    val_dataloader.num_samples = len(val_dataset)
    val_dataloader.num_batches = len(val_dataloader)     

    test_dataset = Chestxray14_Dataset(config['chestxray_test_file'],config['image_res'])
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    test_dataloader =DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=8,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=None,
            worker_init_fn=utils.seed_worker,
            drop_last=True,
        )     
    test_dataloader.num_samples = len(test_dataset)
    test_dataloader.num_batches = len(test_dataloader) 


    test_dataset_chexpert = CheXpert_Dataset(config['chexpert_valid_file'],config['image_res'])
    test_sampler_chexpert = torch.utils.data.distributed.DistributedSampler(test_dataset_chexpert,num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    test_dataloader_chexpert =DataLoader(
            test_dataset_chexpert,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=test_sampler_chexpert,
            collate_fn=None,
            worker_init_fn=utils.seed_worker,
            drop_last=True,
        )     
    test_dataloader_chexpert.num_samples = len(test_dataset_chexpert)
    test_dataloader_chexpert.num_batches = len(test_dataloader_chexpert)
    

    if args.image_encoder_name == 'resnet':
        image_encoder = ModelRes(res_base_model='resnet50').cuda()
    elif args.image_encoder_name == 'dense':
        image_encoder = ModelDense(dense_base_model = 'densenet121').cuda()


    if args.bert_model_name == 'emilyalsentzer/Bio_ClinicalBERT':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
        text_encoder = CLP_clinical2(bert_model_name=args.bert_model_name).cuda()

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name,do_lower_case=True, local_files_only=True)
        text_encoder = CLP_clinical(bert_model_name=args.bert_model_name).cuda()


    if args.bert_pretrained:

        checkpoint = torch.load(args.bert_pretrained, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        text_encoder.load_state_dict(state_dict)
        print('Load pretrained bert success from: ',args.bert_pretrained)
        if args.freeze_bert:
            for param in text_encoder.parameters():
                param.requires_grad = False
    
    if args.add_dataset:
        if 'lam' in config:
            model = TQN_Model_Add(class_num = args.class_num, gate_num = args.gate_num, high_dim = args.high_dim, lam = config['lam']).cuda()
        else:
            model = TQN_Model_Add(class_num = args.class_num, gate_num = args.gate_num, high_dim = args.high_dim).cuda()
    else:
        if 'lam' in config:
            model = TQN_Model(class_num = args.class_num, lam = config['lam']).cuda()
        else:
            model = TQN_Model(class_num = args.class_num).cuda()  


    model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.gpu], find_unused_parameters=True, broadcast_buffers=False)
    model_without_ddp = model.module

    if args.finetune:
        image_encoder_without_ddp = image_encoder
    else:
        image_encoder = torch.nn.parallel.DistributedDataParallel(image_encoder, device_ids = [args.gpu], find_unused_parameters=True, broadcast_buffers=False)
        image_encoder_without_ddp = image_encoder.module

    text_encoder_without_ddp = text_encoder
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model_without_ddp,image_encoder_without_ddp,text_encoder_without_ddp)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 

    if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')

            image_state_dict = checkpoint['image_encoder']   
            new_image_state_dict = OrderedDict()
            for k, v in image_state_dict.items():
                  name = 'module.'+ k 
                  new_image_state_dict[name] = v 
            image_encoder.load_state_dict(new_image_state_dict)  

            text_state_dict =  checkpoint['text_encoder']     
            text_encoder.load_state_dict(text_state_dict)  

            state_dict = checkpoint['model']   
            new_state_dict = OrderedDict()   
            for k, v in state_dict.items():
                  name = 'module.'+ k 
                  new_state_dict[name] = v 
            model.load_state_dict(new_state_dict)    

            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1     


            print('load checkpoint from %s'%args.checkpoint)


    if args.finetune:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            image_state_dict =  checkpoint['image_encoder']     
            image_encoder.load_state_dict(image_state_dict) 

            state_dict = checkpoint['model']   
            new_state_dict = OrderedDict()   
            for k, v in state_dict.items():
                  name = 'module.'+ k 
                  new_state_dict[name] = v 
            
            model.load_state_dict(new_state_dict, strict = False)    

            for param in image_encoder.parameters():
                param.requires_grad = False

            print('load fine-tune checkpoint from %s'%args.finetune)
    
    print("Start training")
    start_time = time.time()
    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))


    best_val_auc = 0.0

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        train_dataloader.sampler.set_epoch(epoch)


        train_stats = train(model, image_encoder, text_encoder, tokenizer, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, args,config,writer) 

        for k, v in train_stats.items():
            if k == 'loss':
                train_loss_epoch = v
            elif k == 'loss_ce':
                train_loss_ce_epoch = v
            elif k == 'loss_clip':
                train_loss_clip_epoch = v
        
        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/train_loss_ce_epoch', float(train_loss_ce_epoch), epoch)
        writer.add_scalar('loss/train_loss_clip_epoch', float(train_loss_clip_epoch), epoch)
        writer.add_scalar('lr/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)

        val_dataloader.sampler.set_epoch(epoch)
        val_loss,val_auc,val_metrics = valid_on_chestxray14(model, image_encoder, text_encoder, tokenizer, val_dataloader,epoch,device,args,config,writer)
        writer.add_scalar('loss/val_loss_epoch', val_loss, epoch)
        writer.add_scalar('loss/val_auc_epoch', val_auc, epoch)

        if best_val_auc < val_auc and dist.get_rank() == 0:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write("Save best valid model.\n")
            best_val_auc = val_auc
            if args.finetune:
                save_obj = {
                'model': model.module.state_dict(),
                'image_encoder': image_encoder.state_dict(),
                'text_encoder':text_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            else:
                save_obj = {
                'model': model.module.state_dict(),
                'image_encoder': image_encoder.module.state_dict(),
                'text_encoder':text_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            torch.save(save_obj, os.path.join(args.aws_output_dir, f"best_valid.pt"))

            test_dataloader.sampler.set_epoch(epoch)
            test_loss, test_auc, test_metrics = valid_on_chestxray14(model, image_encoder, text_encoder, tokenizer, test_dataloader,epoch,device,args,config,writer)
            writer.add_scalar('loss/test_loss_epoch', test_loss, epoch)
            writer.add_scalar('loss/test_auc_epoch', test_auc, epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': val_loss.item(),
                         **{f'val_{k}': v for k, v in val_metrics.items()},
                         'test_loss': test_loss.item(),
                         **{f'test_{k}': v for k, v in test_metrics.items()},
                        }  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': val_loss.item(),
                         **{f'val_{k}': v for k, v in val_metrics.items()},
                        }  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if utils.is_main_process():  
            if args.finetune:
                save_obj = {
                'model': model.module.state_dict(),
                'image_encoder': image_encoder.state_dict(),
                'text_encoder':text_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }            
            else:

                save_obj = {
                'model': model.module.state_dict(),
                'image_encoder': image_encoder.module.state_dict(),
                'text_encoder':text_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            torch.save(save_obj, os.path.join(args.aws_output_dir, 'checkpoint_'+str(epoch)+'.pt'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', default=False, type=bool)
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--freeze_bert', default=True, type=bool)
    parser.add_argument("--use_entity_features", default=True, type=bool)
    parser.add_argument('--dist_backend', default='nccl')

    # Config
    parser.add_argument('--config', default='./config/config.yaml')

    # Data Augmentation
    parser.add_argument('--fourier', default=True, type=bool)
    parser.add_argument('--colourjitter', default=True, type=bool)
    
    # ASL loss & DQN output_dim
    parser.add_argument('--bce', default=False, type=bool) 
    parser.add_argument('--asl', default=True, type=bool) 
    parser.add_argument('--class_num', default=1, type=int) 

    # Port
    parser.add_argument('--port', default=80, type=int)

    # Dataset Enhance
    parser.add_argument('--ignore_index', default=True, type=bool)
    parser.add_argument('--add_dataset', default=True, type=bool)
    parser.add_argument('--gate_num', default=3, type=int)


    parser.add_argument('--high_dim', default=32, type=int)
    parser.add_argument('--main_ratio', default=1) 
    parser.add_argument('--bias_ratio', default=0)  

    parser.add_argument('--moe_ratio', default=1)

    parser.add_argument('--loss_ratio', default=1, type=int)

    # Divide Stage
    parser.add_argument('--finetune', default='')

    # Path
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--aws_output_dir', default='')


    parser.add_argument('--image_encoder_name', default='resnet')

    parser.add_argument('--bert_pretrained', default='./bert_pretrained/epoch_latest.pt')
    parser.add_argument('--bert_model_name', default='GanjinZero/UMLSBert_ENG')


    parser.add_argument('--max_length', default=256, type=int)
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--gpu', default='0', type=str, help='gpu')
    args = parser.parse_args()
    os.environ['MASTER_PORT'] = f'{args.port}'

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.aws_output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))  

    logging.info("Params:")
    params_file = os.path.join(args.output_dir, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")

    seed_torch(args.seed)
    main(args, config)
