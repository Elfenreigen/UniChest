# test on chexpert official
# test on chestxray14 official
# test on padchest dataset
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
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from transformers import AutoModel,BertConfig,AutoTokenizer


from models.clip_tqn import CLP_clinical,ModelRes,TQN_Model,TQN_Model_Add,ModelDense,CLP_clinical2
from dataset.test_dataset import Chestxray14_Dataset,CheXpert_Dataset,Padchest_Dataset,Vindr_Dataset,SIIMACR_Dataset, Shenzhen_Dataset, Openi_Dataset
from engine.test import test
from models.tokenization_bert import BertTokenizer




def main(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    seed = args.seed  
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    if args.test_data == 'chexpert':
        test_dataset = CheXpert_Dataset(config['chexpert_test_file'],config['image_res'])
        test_dataloader =DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                num_workers=4,
                pin_memory=True,
                sampler=None,
                shuffle=False,
                collate_fn=None,
                drop_last=True,
            )
        test_dataloader.num_samples = len(test_dataset)
        test_dataloader.num_batches = len(test_dataloader) 
        args.checkpoint = os.path.join(args.aws_output_dir)
    elif args.test_data == 'chestxray14':
        test_dataset = Chestxray14_Dataset(config['chestxray_test_file'],config['image_res'])
        test_dataloader =DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                num_workers=8,
                pin_memory=True,
                sampler=None,
                shuffle=False,
                collate_fn=None,
                drop_last=True,
            )
        test_dataloader.num_samples = len(test_dataset)
        test_dataloader.num_batches = len(test_dataloader) 
        args.checkpoint = os.path.join(args.aws_output_dir)
    elif args.test_data == 'padchest':
        test_dataset = Padchest_Dataset(config['padchest_all_test_file'],config['image_res'])
        test_dataloader =DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                num_workers=8,
                pin_memory=True,
                sampler=None,
                shuffle=False,
                collate_fn=None,
                drop_last=True,
            )
        test_dataloader.num_samples = len(test_dataset)
        test_dataloader.num_batches = len(test_dataloader) 
        args.checkpoint = os.path.join(args.aws_output_dir)
    elif args.test_data == 'vindr':
        test_dataset = Vindr_Dataset(config['vindrcxr_test_file'],config['image_res'])
        test_dataloader =DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                num_workers=8,
                pin_memory=True,
                sampler=None,
                shuffle=False,
                collate_fn=None,
                drop_last=True,
            )
        test_dataloader.num_samples = len(test_dataset)
        test_dataloader.num_batches = len(test_dataloader) 
        args.checkpoint = os.path.join(args.aws_output_dir)

    elif args.test_data == 'siimacr':
        test_dataset = SIIMACR_Dataset(config['siimacr_file'],config['image_res'])
        test_dataloader =DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                num_workers=8,
                pin_memory=True,
                sampler=None,
                shuffle=False,
                collate_fn=None,
                drop_last=True,
            )
        test_dataloader.num_samples = len(test_dataset)
        test_dataloader.num_batches = len(test_dataloader) 
        args.checkpoint = os.path.join(args.aws_output_dir)

    elif args.test_data == 'shenzhen':
        test_dataset = Shenzhen_Dataset(config['shenzhen_file'],config['image_res'])
        test_dataloader =DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                num_workers=8,
                pin_memory=True,
                sampler=None,
                shuffle=False,
                collate_fn=None,
                drop_last=True,
            )
        test_dataloader.num_samples = len(test_dataset)
        test_dataloader.num_batches = len(test_dataloader) 
        args.checkpoint = os.path.join(args.aws_output_dir)

    elif args.test_data == 'openi':
        test_dataset = Openi_Dataset(config['openi_test_file'],config['image_res'])
        test_dataloader =DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                num_workers=8,
                pin_memory=True,
                sampler=None,
                shuffle=False,
                collate_fn=None,
                drop_last=True,
            )
        test_dataloader.num_samples = len(test_dataset)
        test_dataloader.num_batches = len(test_dataloader) 
        args.checkpoint = os.path.join(args.aws_output_dir)



    if args.image_encoder_name == 'resnet':
        image_encoder = ModelRes(res_base_model='resnet50').to(device) 
    elif args.image_encoder_name == 'dense':
        image_encoder = ModelDense(dense_base_model = 'densenet121').to(device) 


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


    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    image_state_dict = checkpoint['image_encoder']     
    image_encoder.load_state_dict(image_state_dict)    
    text_state_dict =  checkpoint['text_encoder']     
    text_encoder.load_state_dict(text_state_dict)     
    state_dict = checkpoint['model']      
    model.load_state_dict(state_dict)    
    

    if args.test_data == 'chexpert':
        save_result_path = os.path.join(args.save_result_dir,'chexpert')
        Path(save_result_path).mkdir(parents=True, exist_ok=True)

        text_list =['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
         'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
         'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
         'Fracture', 'Support Devices']

        dist_csv_col =['Metric', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
         'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
         'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
         'Fracture', 'Support Devices']
        test(model,image_encoder, text_encoder, tokenizer, test_dataloader,device,save_result_path,args,text_list,dist_csv_col)
        
    elif args.test_data == 'openi':
        save_result_path = os.path.join(args.save_result_dir,'openi')
        Path(save_result_path).mkdir(parents=True, exist_ok=True)

        text_list =['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
         'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
         'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
         'Fracture', 'Support Devices']

        dist_csv_col =['Metric', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
         'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
         'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
         'Fracture', 'Support Devices']
        test(model,image_encoder, text_encoder, tokenizer, test_dataloader,device,save_result_path,args,text_list,dist_csv_col)
    
    elif args.test_data == 'siimacr':
        save_result_path = os.path.join(args.save_result_dir,'siimacr')
        Path(save_result_path).mkdir(parents=True, exist_ok=True)
        text_list = ['pneumothorax']
        dist_csv_col =  ['metric','pneumothorax']
        test(model,image_encoder, text_encoder, tokenizer, test_dataloader,device,save_result_path,args,text_list,dist_csv_col)
    
    elif args.test_data == 'shenzhen':
        save_result_path = os.path.join(args.save_result_dir,'shenzhen')
        Path(save_result_path).mkdir(parents=True, exist_ok=True)
        text_list = ['Tuberculosis']
        dist_csv_col =  ['metric','Tuberculosis']
        test(model,image_encoder, text_encoder, tokenizer, test_dataloader,device,save_result_path,args,text_list,dist_csv_col)

    elif args.test_data == 'chestxray14':
        save_result_path = os.path.join(args.save_result_dir,'chestxray14')
        Path(save_result_path).mkdir(parents=True, exist_ok=True)
        text_list = ["atelectasis","cardiomegaly","pleural effusion","infiltration","lung mass","lung nodule","pneumonia","pneumothorax","consolidation","edema","emphysema","fibrosis","pleural thicken","hernia"]
        dist_csv_col =  ['metric',"atelectasis","cardiomegaly","pleural effusion","infiltration","lung mass","lung nodule","pneumonia","pneumothorax","consolidation","edema","emphysema","fibrosis","pleural thicken","hernia",'mean']
        test(model,image_encoder, text_encoder, tokenizer, test_dataloader,device,save_result_path,args,text_list,dist_csv_col)
    elif args.test_data == 'padchest':
        save_result_path = os.path.join(args.save_result_dir,'padchest')
        Path(save_result_path).mkdir(parents=True, exist_ok=True)
        text_list = ['normal', 'pulmonary fibrosis', 'chronic changes', 'kyphosis', 'pseudonodule', 'ground glass pattern', 'unchanged', 'alveolar pattern', 'interstitial pattern', 'laminar atelectasis', 'pleural effusion', 'apical pleural thickening', 'suture material', 'sternotomy', 'endotracheal tube', 'infiltrates', 'heart insufficiency', 'hemidiaphragm elevation', 'superior mediastinal enlargement', 'aortic elongation', 'scoliosis', 'sclerotic bone lesion', 'supra aortic elongation', 'vertebral degenerative changes', 'goiter', 'COPD signs', 'air trapping', 'descendent aortic elongation', 'aortic atheromatosis', 'metal', 'hypoexpansion basal', 'abnormal foreign body', 'central venous catheter via subclavian vein', 'central venous catheter', 'vascular hilar enlargement', 'pacemaker', 'atelectasis', 'vertebral anterior compression', 'hiatal hernia', 'pneumonia', 'diaphragmatic eventration', 'consolidation', 'calcified densities', 'cardiomegaly', 'fibrotic band', 'tuberculosis sequelae', 'volume loss', 'bronchiectasis', 'single chamber device', 'emphysema', 'vertebral compression', 'bronchovascular markings', 'bullas', 'hilar congestion', 'exclude', 'axial hyperostosis', 'aortic button enlargement', 'calcified granuloma', 'clavicle fracture', 'pulmonary mass', 'dual chamber device', 'increased density', 'surgery neck', 'osteosynthesis material', 'costochondral junction hypertrophy', 'segmental atelectasis', 'costophrenic angle blunting', 'calcified pleural thickening', 'hyperinflated lung', 'callus rib fracture', 'pleural thickening', 'mediastinal mass', 'nipple shadow', 'surgery heart', 'pulmonary artery hypertension', 'central vascular redistribution', 'tuberculosis', 'nodule', 'cavitation', 'granuloma', 'osteopenia', 'lobar atelectasis', 'surgery breast', 'NSG tube', 'hilar enlargement', 'gynecomastia', 'atypical pneumonia', 'cervical rib', 'mediastinal enlargement', 'major fissure thickening', 'surgery', 'azygos lobe', 'adenopathy', 'miliary opacities', 'suboptimal study', 'dai', 'mediastinic lipomatosis', 'surgery lung', 'mammary prosthesis', 'humeral fracture', 'calcified adenopathy', 'reservoir central venous catheter', 'vascular redistribution', 'hypoexpansion', 'heart valve calcified', 'pleural mass', 'loculated pleural effusion', 'pectum carinatum', 'subacromial space narrowing', 'central venous catheter via jugular vein', 'vertebral fracture', 'osteoporosis', 'bone metastasis', 'lung metastasis', 'cyst', 'humeral prosthesis', 'artificial heart valve', 'mastectomy', 'pericardial effusion', 'lytic bone lesion', 'subcutaneous emphysema', 'pulmonary edema', 'flattened diaphragm', 'asbestosis signs', 'multiple nodules', 'prosthesis', 'pulmonary hypertension', 'soft tissue mass', 'tracheostomy tube', 'endoprosthesis', 'post radiotherapy changes', 'air bronchogram', 'pectum excavatum', 'calcified mediastinal adenopathy', 'central venous catheter via umbilical vein', 'thoracic cage deformation', 'obesity', 'tracheal shift', 'external foreign body', 'atelectasis basal', 'aortic endoprosthesis', 'rib fracture', 'calcified fibroadenoma', 'pneumothorax', 'reticulonodular interstitial pattern', 'reticular interstitial pattern', 'chest drain tube', 'minor fissure thickening', 'fissure thickening', 'hydropneumothorax', 'breast mass', 'blastic bone lesion', 'respiratory distress', 'azygoesophageal recess shift', 'ascendent aortic elongation', 'lung vascular paucity', 'kerley lines', 'electrical device', 'artificial mitral heart valve', 'artificial aortic heart valve', 'total atelectasis', 'non axial articular degenerative changes', 'pleural plaques', 'calcified pleural plaques', 'lymphangitis carcinomatosa', 'lepidic adenocarcinoma', 'mediastinal shift', 'ventriculoperitoneal drain tube', 'esophagic dilatation', 'dextrocardia', 'end on vessel', 'right sided aortic arch', 'Chilaiditi sign', 'aortic aneurysm', 'loculated fissural effusion', 'fracture', 'air fluid level', 'round atelectasis', 'mass', 'double J stent', 'pneumoperitoneo', 'abscess', 'pulmonary artery enlargement', 'bone cement', 'pneumomediastinum', 'catheter', 'surgery humeral', 'empyema', 'nephrostomy tube', 'sternoclavicular junction hypertrophy', 'pulmonary venous hypertension', 'gastrostomy tube', 'lipomatosis']
        
        dist_csv_col = ['metric', 'normal', 'pulmonary fibrosis', 'chronic changes', 'kyphosis', 'pseudonodule', 'ground glass pattern', 'unchanged', 'alveolar pattern', 'interstitial pattern', 'laminar atelectasis', 'pleural effusion', 'apical pleural thickening', 'suture material', 'sternotomy', 'endotracheal tube', 'infiltrates', 'heart insufficiency', 'hemidiaphragm elevation', 'superior mediastinal enlargement', 'aortic elongation', 'scoliosis', 'sclerotic bone lesion', 'supra aortic elongation', 'vertebral degenerative changes', 'goiter', 'COPD signs', 'air trapping', 'descendent aortic elongation', 'aortic atheromatosis', 'metal', 'hypoexpansion basal', 'abnormal foreign body', 'central venous catheter via subclavian vein', 'central venous catheter', 'vascular hilar enlargement', 'pacemaker', 'atelectasis', 'vertebral anterior compression', 'hiatal hernia', 'pneumonia', 'diaphragmatic eventration', 'consolidation', 'calcified densities', 'cardiomegaly', 'fibrotic band', 'tuberculosis sequelae', 'volume loss', 'bronchiectasis', 'single chamber device', 'emphysema', 'vertebral compression', 'bronchovascular markings', 'bullas', 'hilar congestion', 'exclude', 'axial hyperostosis', 'aortic button enlargement', 'calcified granuloma', 'clavicle fracture', 'pulmonary mass', 'dual chamber device', 'increased density', 'surgery neck', 'osteosynthesis material', 'costochondral junction hypertrophy', 'segmental atelectasis', 'costophrenic angle blunting', 'calcified pleural thickening', 'hyperinflated lung', 'callus rib fracture', 'pleural thickening', 'mediastinal mass', 'nipple shadow', 'surgery heart', 'pulmonary artery hypertension', 'central vascular redistribution', 'tuberculosis', 'nodule', 'cavitation', 'granuloma', 'osteopenia', 'lobar atelectasis', 'surgery breast', 'NSG tube', 'hilar enlargement', 'gynecomastia', 'atypical pneumonia', 'cervical rib', 'mediastinal enlargement', 'major fissure thickening', 'surgery', 'azygos lobe', 'adenopathy', 'miliary opacities', 'suboptimal study', 'dai', 'mediastinic lipomatosis', 'surgery lung', 'mammary prosthesis', 'humeral fracture', 'calcified adenopathy', 'reservoir central venous catheter', 'vascular redistribution', 'hypoexpansion', 'heart valve calcified', 'pleural mass', 'loculated pleural effusion', 'pectum carinatum', 'subacromial space narrowing', 'central venous catheter via jugular vein', 'vertebral fracture', 'osteoporosis', 'bone metastasis', 'lung metastasis', 'cyst', 'humeral prosthesis', 'artificial heart valve', 'mastectomy', 'pericardial effusion', 'lytic bone lesion', 'subcutaneous emphysema', 'pulmonary edema', 'flattened diaphragm', 'asbestosis signs', 'multiple nodules', 'prosthesis', 'pulmonary hypertension', 'soft tissue mass', 'tracheostomy tube', 'endoprosthesis', 'post radiotherapy changes', 'air bronchogram', 'pectum excavatum', 'calcified mediastinal adenopathy', 'central venous catheter via umbilical vein', 'thoracic cage deformation', 'obesity', 'tracheal shift', 'external foreign body', 'atelectasis basal', 'aortic endoprosthesis', 'rib fracture', 'calcified fibroadenoma', 'pneumothorax', 'reticulonodular interstitial pattern', 'reticular interstitial pattern', 'chest drain tube', 'minor fissure thickening', 'fissure thickening', 'hydropneumothorax', 'breast mass', 'blastic bone lesion', 'respiratory distress', 'azygoesophageal recess shift', 'ascendent aortic elongation', 'lung vascular paucity', 'kerley lines', 'electrical device', 'artificial mitral heart valve', 'artificial aortic heart valve', 'total atelectasis', 'non axial articular degenerative changes', 'pleural plaques', 'calcified pleural plaques', 'lymphangitis carcinomatosa', 'lepidic adenocarcinoma', 'mediastinal shift', 'ventriculoperitoneal drain tube', 'esophagic dilatation', 'dextrocardia', 'end on vessel', 'right sided aortic arch', 'Chilaiditi sign', 'aortic aneurysm', 'loculated fissural effusion', 'fracture', 'air fluid level', 'round atelectasis', 'mass', 'double J stent', 'pneumoperitoneo', 'abscess', 'pulmonary artery enlargement', 'bone cement', 'pneumomediastinum', 'catheter', 'surgery humeral', 'empyema', 'nephrostomy tube', 'sternoclavicular junction hypertrophy', 'pulmonary venous hypertension', 'gastrostomy tube', 'lipomatosis']
        test(model,image_encoder, text_encoder, tokenizer, test_dataloader,device,save_result_path,args,text_list,dist_csv_col)

    elif args.test_data == 'vindr':
        save_result_path = os.path.join(args.save_result_dir,'vindr')
        Path(save_result_path).mkdir(parents=True, exist_ok=True)
        text_list = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Clavicle fracture',
                     'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity',
                     'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion', 
                     'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture', 'Other lesion',
                     'COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other disease', 'No finding']
        dist_csv_col = ['metric', 'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Clavicle fracture',
                     'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity',
                     'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion', 
                     'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture', 'Other lesion',
                     'COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other disease', 'No finding']
        test(model,image_encoder, text_encoder, tokenizer, test_dataloader,device,save_result_path,args,text_list,dist_csv_col)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--momentum', default=False, type=bool)
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--freeze_bert', default=False, type=bool)
    parser.add_argument("--use_entity_features", action="store_true")
    parser.add_argument('--image_encoder_name', default='resnet')
    parser.add_argument('--bert_pretrained', default='./bert_pretrained/epoch_latest.pt')
    parser.add_argument('--bert_model_name', default='GanjinZero/UMLSBert_ENG')
    parser.add_argument('--output_dir', default='')

    parser.add_argument('--config', default='./config/config.yaml')
    parser.add_argument('--aws_output_dir', default='')
    parser.add_argument('--test_data', default='chexpert')
    parser.add_argument('--bce', default=False, type=bool) 
    parser.add_argument('--asl', default=True, type=bool) 

    parser.add_argument('--class_num', default=1, type=int) 
    parser.add_argument('--save_result_dir', default='./result')
    parser.add_argument('--ignore_index', default=True, type=bool) 
    parser.add_argument('--add_dataset', default=True, type=bool)


    parser.add_argument('--gate_num', default=3, type=int)
    parser.add_argument('--high_dim', default=32, type=int)
    parser.add_argument('--main_ratio', type=float,default=1)
    parser.add_argument('--bias_ratio', type=float,default=0)
    parser.add_argument('--moe_ratio', type=float, default=1) 


    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--loss_ratio', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.save_result_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
