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



"""
an example for command in the terminal:
python3 train.py --epochs 300 --data_config config/custom.data --img_size 416 \
--batch_size 8 \
--model_def config/yolov3_gaps_1class.cfg
--data_config
--pretrained_weights checkpoints/train_with_crop1/yolov3_one_channel_416_99.pth --starts_epochs 0


multi_class:
python3 train.py --epochs 80 \
--data_config config/custom_code_testing.data \
--img_size 416 \
--batch_size 4 \
--model_def config/yolov3_gaps_crop_3class.cfg \
--weights_folder checkpoints/test/    
    
"""
pid = os.getpid()
import subprocess
subprocess.Popen("renice -n 10 -p {}".format(pid),shell=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3_gaps_crop.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=0, help="allow for multi-scale training")
    parser.add_argument("--starts_epochs", type=int, default=0, help="if there is pretrained weights")
    parser.add_argument("--weights_folder", type=str, default='checkpoints/', help="path to save the weights")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")
    
    filename = 'test_log.log'
    logging.basicConfig(filename='example.log')
    logging.debug('This message should go to the log file')
    logging.info(opt)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    os.makedirs("output", exist_ok=True)
    os.makedirs(opt.weights_folder, exist_ok=True)   
    
    # write the parameter setting into txt files and save it in the same folder with the weights
    with open(os.path.join(opt.weights_folder,'A_note.txt'),'w') as f:
        json.dump(opt.__dict__,f)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)  
    # opt.data_config: config/custom.data
    # running result is a dictionary contains the basic information like 
    # the number of classes and the path to train.txt and valid.txt
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    valid_path2 = data_config["valid_cross"]
    

    class_names = load_classes(data_config["names"])

    file = open(train_path, 'r')
    lines = file.read().split('\n')
    #lines = [x for x in lines if x.startswith('data/custom/images')] 
    num_train = len(lines)
    
    file = open(valid_path, 'r')
    lines = file.read().split('\n')
    #lines = [x for x in lines if x.startswith('data/custom/images')] 
    num_valid = len(lines)
    
    file = open(valid_path2, 'r')
    lines = file.read().split('\n')
    #lines = [x for x in lines if x.startswith('data/custom/images')] 
    num_valid2 = len(lines)
    print('number of training set:',num_train)
    print('number of valid set:',num_valid)
    print('number of valid set2:',num_valid2)

    with open(os.path.join(opt.weights_folder,'A_note.txt'),'a') as f:
        f.write('\n')
        f.write('\n')
        f.write(f"number of training set:{num_train} \n")
        f.write(f"number of valid set:{num_valid} \n")
        f.write(f"number of valid set2:{num_valid2} \n")
        f.write(f"train path: {train_path} \n")
        f.write(f"valid path: {valid_path} \n")
        f.write(f"valid path2: {valid_path2} \n")
        
    f.close()
    
    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, img_size=opt.img_size, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    
    total_loss = []
    total_loss_title = []
    valid_mAP = []
    valid_mAP2 = []
    valid_mAP_title = []
    
    
    for epoch in range(opt.starts_epochs, opt.starts_epochs+opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.starts_epochs+opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            total_loss.append(loss.item())
            total_loss_title.append(f"Epoch {epoch} Batch {batch_i}")

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            
            logging.info(loss.item())            
            
            #print(time_left)
            if batch_i == 0:
                print(log_str)
                
            model.seen += imgs.size(0)
        
                
        if (epoch+1) % opt.evaluation_interval == 0 and epoch >= 19:
            print("\n---- Evaluating Model ----")
            print(opt.weights_folder)
            # Evaluate the model on the validation set
            if loss.item() < 4:
                conf_thres = 0.3
            else:
                conf_thres = 0.1
            loss, precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=conf_thres, # !!! origianl 0.5 current 0.3
                nms_thres=0.05, #!!! original 0.5
                img_size=640,
                batch_size=opt.batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            
            print(f"---- mAP {AP.mean()}")
            valid_mAP.append([loss.item(), precision.mean(),recall.mean(),
                              AP.mean(),f1.mean()])
            valid_mAP_title.append(f"Epoch {epoch}")            

            print("\n---- Cross Evaluating Model ----")
            # Evaluate the model on the validation set
            if loss.item() < 4:
                conf_thres = 0.3
            else:
                conf_thres = 0.1
            loss, precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path2,
                iou_thres=0.5,
                conf_thres=conf_thres, # !!! origianl 0.5 current 0.3
                nms_thres=0.05, #!!! original 0.5
                img_size=640,  #!!! check the size                
                batch_size=opt.batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            
            print(f"---- mAP2 {AP.mean()}")
            valid_mAP2.append([loss.item(), precision.mean(),recall.mean(),
                              AP.mean(),f1.mean()])

            
        if (epoch+1) % opt.checkpoint_interval == 0 and epoch >= 79:
            weights_path = os.path.join(opt.weights_folder,f"weights_{epoch}.pth")
            torch.save(model.state_dict(), weights_path)

    file_name = f"loss_train.csv"
    dict_loss = {'time':total_loss_title,'total_loss':total_loss}
    df = pd.DataFrame(dict_loss)
    df.to_csv(os.path.join(opt.weights_folder, file_name))
    
    file_name = f"validation.csv"
    dict_loss = {'epoch':valid_mAP_title,'metrics':valid_mAP} 
    df = pd.DataFrame(dict_loss)
    df.to_csv(os.path.join(opt.weights_folder, file_name))
    
    file_name = f"validation2.csv"
    dict_loss = {'epoch':valid_mAP_title,'metrics':valid_mAP2} 
    df = pd.DataFrame(dict_loss)
    df.to_csv(os.path.join(opt.weights_folder, file_name))
