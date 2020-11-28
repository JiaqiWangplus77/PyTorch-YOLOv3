
"""
add ground truth to the predicted images
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse

"""
add groud truth on the images

python3 evaluation_add_groud_truth.py \
--image_folder output/aug23_img576_test1_weights_173 \
--label_folder data/custom/GAPs384/labels

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to image folder")
    parser.add_argument("--label_folder", type=str, default="output/", help="path to label folder")

    opt = parser.parse_args()
        
    deal_with_all_files = 1
    image_num = 3
    images_folder = os.path.join(opt.image_folder)
    # path to the predicted results
    label_folder = os.path.join(opt.label_folder)
    result_folder = os.path.join(images_folder,'with_ground_truth')
    os.makedirs(result_folder, exist_ok= True)
    
    image_list = [file.replace('png','txt')
                     for file in os.listdir(images_folder)
                        if file.endswith('png')]
    
    if deal_with_all_files:
        image_num = 0
        plot_num = range(len(image_list))
    else:
        plot_num = [image_num]
    
    def coordinate_transform(yolo, imag_width, imag_height):
        """
        transfrom from yolo format [type, x_center,y_center,width,height] (scaled to 1)
        to the coordinate of the two corners, which makes it easier for bounding box plotting"""
         
        coord = np.zeros_like(yolo)
        coord[:,0] = yolo[:,0] 
        coord[:,1] = (yolo[:,1] - yolo[:,3] / 2) * imag_width
        coord[:,2] = (yolo[:,2] - yolo[:,4] / 2) * imag_height
        coord[:,3] = (yolo[:,1] + yolo[:,3] / 2) * imag_width
        coord[:,4] = (yolo[:,2] + yolo[:,4] / 2) * imag_height
        coord = np.int64(coord)    
        return coord
    
    color = [(255,0,0),(0,255,0),(0,0,255)] 
        
    for num in plot_num:
        filename = os.path.join(label_folder,image_list[num])
        if not os.path.isfile(filename):
            print(f"{image_list[num]} not found")
        
        else:
            image_name = os.path.join(images_folder,
                                      image_list[num].replace('txt','png'))
            image = np.array(Image.open(image_name).convert('RGB'))
            yolo = np.array(np.loadtxt(filename)).reshape(-1,5) 
    
            
            imag_height, imag_width = image.shape[0],image.shape[1]  
            coord = coordinate_transform(yolo,imag_width,imag_height)       
            for i in range(coord.shape[0]):
                bbox = [(coord[i,1],coord[i,2]),(coord[i,3],coord[i,4])]            
                cv2.rectangle(image, pt1=bbox[0], pt2=bbox[1],color=color[1],thickness=2) 
    #            cv2.putText(image,'Ground Truth',(coord[i,1],coord[i,2]+10), 
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            
            if not deal_with_all_files:
                plt.figure()   
                plt.imshow(image)
            cv2.imwrite(os.path.join(result_folder,
                                     image_list[num].replace('txt','png')),
                        image)  
    print(f"{len(plot_num)} finished")

        

        
    
