import csv
import json
import logging
import os
import random
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
import pydicom
from skimage import exposure


class Openi_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,3])
        self.class_list = np.asarray(data_info.iloc[:,4:])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize(image_res, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        # img_path = self.img_path_list[index].replace('/mnt/cfs/xmzhang/DATA/ChestXray8/','/remote-home/share/medical/public/ChestXray8/')
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
        
        
        
class Chestxray14_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize(image_res, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index].replace('/mnt/petrelfs/zhangxiaoman/DATA/Chestxray/ChestXray8/','/remote-home/share/medical/public/ChestXray8/')
        
        # img_path = self.img_path_list[index].replace('/mnt/cfs/xmzhang/DATA/ChestXray8/','/remote-home/share/medical/public/ChestXray8/')
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)



class CheXpert_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        # self.class_list = np.asarray(data_info.iloc[:,[9,3,7,6,11]])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = os.path.join('/remote-home/share/xmzhang/CheXpert/',self.img_path_list[index])
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

class Padchest_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        # try:
            img_path = self.img_path_list[index].replace('/mnt/petrelfs/zhangxiaoman/DATA/Chestxray/PadChest/images/', '/remote-home/share/xmzhang/PadChest/LR_images/')
            class_label = self.class_list[index] 
            img_array = np.array(Image.open(img_path))
            img_array = (img_array/img_array.max())*255
            img = Image.fromarray(img_array.astype('uint8')).convert('RGB')   
            image = self.transform(img)
            return {
                "img_path": img_path,
                "image": image,
                "label": class_label
                }
        # except:
        #     select_index = random.randint(10000)
        #     img_path = self.img_path_list[select_index]
        #     class_label = self.class_list[select_index] 
        #     img_array = np.array(Image.open(img_path))
        #     img_array = (img_array/img_array.max())*255
        #     img = Image.fromarray(img_array.astype('uint8')).convert('RGB')   
        #     image = self.transform(img)
        #     return {
        #         "img_path": img_path,
        #         "image": image,
        #         "label": class_label
        #         }
    
    def __len__(self):
        return len(self.img_path_list)

class Vindr_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
        

class SIIMACR_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

    
class Shenzhen_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)


    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,5])
        self.class_list = np.asarray(data_info.iloc[:,6:])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = '/remote-home/share/medical/public/MIMIC-CXR-JPG/MIMIC-CXR/small/' + self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)