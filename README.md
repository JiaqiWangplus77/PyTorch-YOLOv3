# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt



## Train
```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```


#### Custom model
Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your dataset.

```
$ cd config/                                # Navigate to config dir
$ bash create_custom_model.sh <num-classes> # Will create custom model 'yolov3-custom.cfg'
```

#### Classes
Add class names to `data/custom/classes.names`. This file should have one row per class name.

#### Image Folder
Move the images of your dataset to `data/custom/images/`.

#### Annotation Folder
Move your annotations to `data/custom/labels/`. The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

#### Define Train and Validation Sets
In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation data respectively.

#### Train
To train on the custom dataset run:

```
$ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
```

Add `--pretrained_weights weights/darknet53.conv.74` to train using a backend pretrained on ImageNet.

#### Detect
evaluation_detection.py. 
Ddetect the results with the weights file. An example in the terminal:
python3 evaluation_detection.py \
--model_def config/yolov3_gaps_1class.cfg \
--image_folder data/samples/predict \
--output_image_folder output/aug23_img576_test1_weights_173  \
--class_path data/custom/GAPs384/classes.names \
--weights_path checkpoints/aug23_img576_test1/weights_173.pth \
--nms_thres 0.05 --conf_thres 0.5 \
--img_size 640 --batch_size 2 

evaluation_add_groud_truth.py
Add groud truth on the images. An example in the terminal:
python3 evaluation_add_groud_truth.py \
--image_folder output/aug23_img576_test1_weights_173 \
--label_folder data/custom/GAPs384/labels

evaluation_dataset.py
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

train.py 
train the dataset
an example at the terminal
python3 train.py --epochs 500 \
--data_config config/GAPs384/with_aug123.data \
--multiscale_training 0 \
--img_size 640 \
--batch_size 4 \
--evaluation_interval 2 \
--checkpoint_interval 2 \
--model_def config/GAPs384/yolov3_gaps_1class.cfg \
--weights_folder checkpoints/GAPs384/test_delete/


## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
