import csv
import json
import logging
import os
import re
import sys
from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import cv2

from dataset.randaugment import RandomAugment
from io import BytesIO


class MIMIC_Dataset(Dataset):
    def __init__(self, json_path, csv_path, sty_path,image_res,args):
        self.json_info = json.load(open(json_path,'r'))
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])#40 class for fine-grained query list
        sty_info = pd.read_csv(sty_path)
        self.sty_dict_info = self.csv_to_dict(sty_info)

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if args.colourjitter:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomGrayscale(),

                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])

        else:
                self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])    

    
    def csv_to_dict(self,sty_info):
        tui_list = sty_info.iloc[:,0]
        sty_list = sty_info.iloc[:,1]
        sty_dict = defaultdict(list)
        for idx in tqdm(range(len(tui_list))):
            tui_idx = tui_list[idx]
            sty_idx = sty_list[idx]
            sty_dict[tui_idx] = sty_idx
        return sty_dict
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index].replace("/nvme/zhangruipeng/zhangxiaoman/dataset/MIMIC-CXR-DCM/files", '/remote-home/share/medical/public/MIMIC-CXR-JPG/MIMIC-CXR/small/files')
        class_label = self.class_list[index] 

        # index_transit = np.load("/remote-home/tianjiedai/KAD/R1_CLIP_LR/A1_DATA/small/index0626.npy")
        # new_index_json = index_transit[index]
        # entities = self.json_info[new_index_json]['entities']
        # captions = self.json_info[new_index_json]['caption']
        
        entities = self.json_info[index]['entities']
        captions = self.json_info[index]['caption']


        if len(entities) != 0:
            caption_list = ''
            entity_details = ''
            for entity in entities:
                sub_caption = entity['caption']
                sub_entities = entity['entity']#搞错了 还不是list
                sub_entity_details = ''
                for sub_entity in sub_entities:
                    try:
                        sub_entity_details += ' [ENT] ' + sub_entity['Entity'] 
                    except:
                        sub_entity_details += ' [ENT] ' + sub_entity['Entity']  
                entity_details = entity_details + sub_entity_details + ' [SEP] '
                caption_list = caption_list + sub_caption + ' [SEP] '
        else:
            caption_list = ''
            entity_details = ''
            for sub_caption in captions:
                caption_list = caption_list + sub_caption + ' [SEP] '
            entity_details = caption_list
        
        # img = open_jpg(img_path).convert('RGB')  
        img = Image.open(img_path).convert('RGB') 
        image = self.transform(img)
        return {
            "image": image,
            "label": class_label,
            "caption": caption_list,
            "entity": entity_details
            }
    

 
class Mergetrain_Dataset(Dataset):
    def __init__(self, json_path, csv_path, sty_path,image_res,args):
        self.json_info = json.load(open(json_path,'r'))
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,2:])#60 class for fine-grained query list
        self.label_dataset_list = np.asarray(data_info.iloc[:,1])

        sty_info = pd.read_csv(sty_path)
        self.sty_dict_info = self.csv_to_dict(sty_info)

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if args.colourjitter:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomGrayscale(),

                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])

        else:
                self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])    

    
    def csv_to_dict(self,sty_info):
        tui_list = sty_info.iloc[:,0]
        sty_list = sty_info.iloc[:,1]
        sty_dict = defaultdict(list)
        for idx in tqdm(range(len(tui_list))):
            tui_idx = tui_list[idx]
            sty_idx = sty_list[idx]
            sty_dict[tui_idx] = sty_idx
        return sty_dict
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):

        if self.label_dataset_list[index] == 0:
            img_path = self.img_path_list[index].replace("/nvme/zhangruipeng/zhangxiaoman/dataset/MIMIC-CXR-DCM/files", '/remote-home/share/medical/public/MIMIC-CXR-JPG/MIMIC-CXR/small/files')
            class_label = self.class_list[index] 

            # index_transit = np.load("/remote-home/tianjiedai/KAD/R1_CLIP_LR/A1_DATA/small/index0626.npy")
            # new_index_json = index_transit[index]
            # entities = self.json_info[new_index_json]['entities']
            # captions = self.json_info[new_index_json]['caption']
        
            entities = self.json_info[index]['entities']
            captions = self.json_info[index]['caption']


            if len(entities) != 0:
                caption_list = ''
                entity_details = ''
                for entity in entities:
                    sub_caption = entity['caption']
                    sub_entities = entity['entity']#搞错了 还不是list
                    sub_entity_details = ''
                    for sub_entity in sub_entities:
                        try:
                            sub_entity_details += ' [ENT] ' + sub_entity['Entity'] 
                        except:
                            sub_entity_details += ' [ENT] ' + sub_entity['Entity']  
                    entity_details = entity_details + sub_entity_details + ' [SEP] '
                    caption_list = caption_list + sub_caption + ' [SEP] '
            else:
                caption_list = ''
                entity_details = ''
                for sub_caption in captions:
                    caption_list = caption_list + sub_caption + ' [SEP] '
                entity_details = caption_list
        
            # img = open_jpg(img_path).convert('RGB')  
            # img = Image.open(img_path).convert('RGB') 
            # image = self.transform(img)
            # return {
            # "image": image,
            # "label": class_label,
            # "caption": caption_list,
            # "entity": entity_details
            #     }
        
        else:
            img_path = self.img_path_list[index]
            class_label = self.class_list[index]  
            caption_list = ''
            head = ['normal', 'pleural effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis',  'tube', 'consolidation','enlarged cardiomediastinum','tip', 'pneumonia','line','cardiomegaly', 'fracture','calcification',
            'device','engorgement',  'nodule', 'wire',  'pacemaker', 'pleural thicken', 'marking', 'scar', 'hyperinflate', 'blunt',  'collapse', 'emphysema', 'aerate', 'mass','infiltration', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'lesion', 'hardware', 'dilation',  'aspiration',
            'fibrosis',	'No Finding', 'Pleural Other', 'Support Devices', 'Aortic enlargement',
            'Clavicle fracture', 'Enlarged PA', 'ILD', 'Lung cavity', 'Lung cyst', 'Mediastinal shift',	
            'Nodule/Mass', 'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Tuberculosis',
            'Other diseases']
            index_positive = np.where(class_label == 1)
            entity =  np.array(head)[index_positive]
            entity_details = ''
            for sub_entity in entity:
                entity_details = entity_details + sub_entity + ' [SEP] '

        img = Image.open(img_path).convert('RGB') 
        image = self.transform(img)
        label_dataset = self.label_dataset_list[index]

        return {
            "image": image,
            "label": class_label,
            "label_dataset": label_dataset,
            "caption": caption_list,
            "entity": entity_details
            }



class Chestxray14_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize(image_res, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index].replace('/mnt/petrelfs/zhangxiaoman/DATA/Chestxray/ChestXray8/','/remote-home/share/medical/public/ChestXray8/')
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)


class CheXpert_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,[13,7,11,10,15]])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = os.path.join('/remote-home/share/tianjiedai/',self.img_path_list[index])
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

