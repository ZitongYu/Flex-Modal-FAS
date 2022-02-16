from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa


 


#face_scale = 0.9  #default for test, for training , can be set from [0.8 to 1.0]

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])



# Tensor
class Cutout(object):
    def __init__(self, length=30):
        self.length = length

    def __call__(self, sample):
        img, image_x_depth, image_x_ir, spoofing_label, map_x1 = sample['image_x'],sample['image_x_depth'],sample['image_x_ir'],sample['spoofing_label'],sample['map_x1']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        image_x_depth *= mask
        image_x_ir *= mask
        
        return {'image_x': img, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir, 'spoofing_label': spoofing_label, 'map_x1': map_x1}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, image_x_depth, image_x_ir, spoofing_label, map_x1 = sample['image_x'],sample['image_x_depth'],sample['image_x_ir'],sample['spoofing_label'],sample['map_x1']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_image_x_depth = (image_x_depth - 127.5)/128     # [-1,1]
        new_image_x_ir = (image_x_ir - 127.5)/128     # [-1,1]
        return {'image_x': new_image_x, 'image_x_depth': new_image_x_depth, 'image_x_ir': new_image_x_ir, 'spoofing_label': spoofing_label, 'map_x1': map_x1}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, image_x_depth, image_x_ir, spoofing_label, map_x1 = sample['image_x'],sample['image_x_depth'],sample['image_x_ir'],sample['spoofing_label'],sample['map_x1']
        
        new_image_x = np.zeros((224, 224, 3))
        new_image_x_depth = np.zeros((224, 224, 3))
        new_image_x_ir = np.zeros((224, 224, 3))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_image_x_depth = cv2.flip(image_x_depth, 1)
            new_image_x_ir = cv2.flip(image_x_ir, 1)

                
            return {'image_x': new_image_x, 'image_x_depth': new_image_x_depth, 'image_x_ir': new_image_x_ir, 'spoofing_label': spoofing_label, 'map_x1': map_x1}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir, 'spoofing_label': spoofing_label, 'map_x1': map_x1}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, image_x_depth, image_x_ir, spoofing_label, map_x1 = sample['image_x'],sample['image_x_depth'],sample['image_x_ir'],sample['spoofing_label'],sample['map_x1']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        p = random.random()
        if p < 0.5:
            image_x_depth = image_x_depth[:,:,::-1].transpose((2, 0, 1))
            image_x_depth = np.array(image_x_depth)
        else:
            # Blocked modality
            image_x_depth = np.zeros((3, 224, 224))
            image_x_depth = np.array(image_x_depth)
        
        p1 = random.random()
        if p1 < 0.5:
            image_x_ir = image_x_ir[:,:,::-1].transpose((2, 0, 1))
            image_x_ir = np.array(image_x_ir)
        else:
            # Blocked modality
            image_x_ir = np.zeros((3, 224, 224))
            image_x_ir = np.array(image_x_ir)
            
        
        map_x1 = np.array(map_x1)
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'image_x_depth': torch.from_numpy(image_x_depth.astype(np.float)).float(), 'image_x_ir': torch.from_numpy(image_x_ir.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long(), 'map_x1': torch.from_numpy(map_x1.astype(np.float)).float()}


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_train(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
             
        videoname_depth = str(self.landmarks_frame.iloc[idx, 1])
        image_path_depth = os.path.join(self.root_dir, videoname_depth)     
        
        videoname_ir = str(self.landmarks_frame.iloc[idx, 2])
        image_path_ir = os.path.join(self.root_dir, videoname_ir)     
        
        
        #log_file2 = open('temp.txt', 'w')
        #log_file2.write('%s \n' % (image_path))
        #log_file2.write('%s \n' % (image_path_depth))
        #log_file2.write('%s \n' % (image_path_ir))
        #log_file2.flush()
             
        image_x, map_x1 = self.get_single_image_x_RGB(image_path)
        image_x_depth = self.get_single_image_x(image_path_depth)
        image_x_ir = self.get_single_image_x(image_path_ir)
		    
        spoofing_label = self.landmarks_frame.iloc[idx, 3]
        
        if spoofing_label == 1:            # real
            spoofing_label = 1            # real
            #map_x1 = np.zeros((28, 28))   # real
            #map_x1 = np.ones((28, 28))
        else:                              # fake
            spoofing_label = 0
            #map_x1 = np.ones((28, 28))    # fake
            map_x1 = np.zeros((28, 28))
        

        sample = {'image_x': image_x, 'image_x_depth': image_x_depth, 'image_x_ir': image_x_ir, 'spoofing_label': spoofing_label, 'map_x1': map_x1}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x_RGB(self, image_path):
        
        image_x = np.zeros((224, 224, 3))
        binary_mask = np.zeros((28, 28))

        # RGB
        image_x_temp = cv2.imread(image_path)
        
        #cv2.imwrite('temp.jpg', image_x_temp)
  
        image_x = cv2.resize(image_x_temp, (224, 224))
        
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        image_x_aug = seq.augment_image(image_x) 
        
        image_x_temp_gray = cv2.imread(image_path, 0)
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (28, 28))
        for i in range(28):
            for j in range(28):
                if image_x_temp_gray[i,j]>0:
                    binary_mask[i,j]=1
                else:
                    binary_mask[i,j]=0
        
        return image_x_aug, binary_mask
        
    def get_single_image_x(self, image_path):
        
        image_x = np.zeros((224, 224, 3))

        # RGB
        image_x_temp = cv2.imread(image_path)
        
        #cv2.imwrite('temp.jpg', image_x_temp)
  
        image_x = cv2.resize(image_x_temp, (224, 224))
        
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        image_x_aug = seq.augment_image(image_x) 
 
        
        return image_x_aug




