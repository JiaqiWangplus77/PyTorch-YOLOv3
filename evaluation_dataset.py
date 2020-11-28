from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test1 import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import logging
import pandas as pd
import numpy as np


pid = os.getpid()
import subprocess
subprocess.Popen("renice -n 10 -p {}".format(pid),shell=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
calculate precision, recall, AP, f1, ap_class with weights file on specific dataset
Notes: for gray-scale image and the model trained for 1 channel image,
Modify the function: class ListDataset(Dataset) in utils/dataset.py
def __getitem__(self, index): img = transforms.ToTensor()(Image.open(img_path).convert('L'))

an example at the terminal
python3 evaluation_dataset.py \
--model_def config/yolov3_gaps_1class.cfg \
--weights_path checkpoints/aug23_img576_test1/weights_173.pth \
--nms_thres 0.05 --conf_thres 0.5 \
--img_size 640 --batch_size 2 \
--valid_path data/custom/GAPs384/list_for_shuffle/predict_new.txt

'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3_gaps_1class.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, 
                        default="checkpoints/crop_Gaps/current/valid_img_size640/image640/aug23/CFD_cross/test3_save_weights/weights_233.pth", 
                        help="path to weights file")
#    parser.add_argument("--output_image_folder", type=str, default="output/delete_later", help="path to dataset")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension")
    parser.add_argument("--valid_path", type=str, default='data/custom/filtered_crop_dataset_one_class/file_list/list_for_shuffle/predict_new.txt', help="if True computes mAP every tenth batch")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="")
    parser.add_argument("--nms_thres", type=float, default=0.05, help="")

 
# data/custom/filtered_crop_dataset_one_class/file_list/list_for_shuffle/valid_shuffle.txt
# checkpoints/CFD/with_aug23/parameter/new_valid.txt
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
 
    file = open(opt.valid_path, 'r')
    lines = file.read().split('\n')
    #lines = [x for x in lines if x.startswith('data/custom/images')] 
    num_valid = len(lines)
    print('number of valid set:',num_valid)
    
#    os.makedirs(opt.output_image_folder, exist_ok=True)
#    with open(os.path.join(opt.output_image_folder,'A_note.txt'),'a') as f:
#        f.write('\n')
#        f.write(f"number of valid set:{num_valid} \n")
#        f.write(f"valid path: {opt.valid_path} \n")        
#    f.close()
          
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint

    model.load_state_dict(torch.load(opt.weights_path))

    # Get dataloader                  
    print("\n---- Evaluating Model ----")
    
    opt.iou_thres = 0.5
    ap = 0
    APs = []
#    for opt.iou_thres in np.arange(0.5,0.96,0.05):
    for opt.iou_thres in np.arange(0.5,0.54,0.05):

        loss, precision, recall, AP, f1, ap_class = evaluate(
            model,                
            path=opt.valid_path,
            iou_thres= opt.iou_thres,      # !!! origianl 0.50.5,
            conf_thres= opt.conf_thres , # !!! origianl 0.5 current 0.3
            nms_thres=opt.nms_thres, #!!! original 0.5 0.001
            img_size=opt.img_size,
            batch_size=opt.batch_size,
        )
        #print(AP.mean())
        ap += AP.mean()
        APs.append([opt.iou_thres,AP.mean()])

        
    evaluation_metrics = [
        ("val_precision", round(precision.mean(),4)),
        ("val_recall", round(recall.mean(),4)),
        ("val_mAP", round(AP.mean(),4)),
        ("val_f1", round(f1.mean(),4)),
    ]

    print(f"\n----  {round(AP.mean(),6)}")
    print(f"\n---- mAP {evaluation_metrics}")
    result=[loss.item(), precision.mean(),recall.mean(),AP.mean(),f1.mean()]
    APs = np.round(np.array(APs),4)
    print(ap/10)
         
 