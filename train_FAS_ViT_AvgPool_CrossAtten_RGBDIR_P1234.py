from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from models.ViT_Base_CA import ViT_AvgPool_3modal_CrossAtten_Channel


#from Load_FAS_MultiModal_Blocked import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout
from Load_FAS_MultiModal import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout
from Load_FAS_MultiModal_Blocked_test import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils_FAS_MultiModal2 import AvgrageMeter, performances_FAS_MultiModal


##########    Dataset root    ##########

# root_dir    CASIA_SURF_CeFA; WMCA
root_FAS_dir = '/scratch/project_2004030/'

# train_list     CASIA_SURF_CeFA
train_CASIA_SURF_CeFA_list = 'FlexModal_Protocols/CASIA-SURF_CeFA_train.txt'

# val_list     CASIA_SURF_CeFA
val_CASIA_SURF_CeFA_list = 'FlexModal_Protocols/CASIA-SURF_CeFA_val.txt'

# Intra-test_list      CASIA_SURF_CeFA
test_CASIA_SURF_CeFA_list = 'FlexModal_Protocols/CASIA-SURF_CeFA_test.txt'

# Cross-test      WMCA
test_WMCA_list = 'FlexModal_Protocols/WMCA_test.txt' 
# finegrained types
test_WMCA_fakehead_list = 'FlexModal_Protocols/WMCA_test_fakehead.txt'
test_WMCA_flexiblemask_list = 'FlexModal_Protocols/WMCA_test_flexiblemask.txt'
test_WMCA_glasses_list = 'FlexModal_Protocols/WMCA_test_glasses.txt'
test_WMCA_papermask_list = 'FlexModal_Protocols/WMCA_test_papermask.txt'
test_WMCA_print_list = 'FlexModal_Protocols/WMCA_test_print.txt'
test_WMCA_replay_list = 'FlexModal_Protocols/WMCA_test_replay.txt'
test_WMCA_rigidmask_list = 'FlexModal_Protocols/WMCA_test_rigidmask.txt'



# feature  -->   [ batch, channel, height, width ]
def FeatureMap2Heatmap(x, x2, x3):
    ## initial images 
    ## initial images 
    org_img = x[0,:,:,:].cpu()  
    org_img = org_img.data.numpy()*128+127.5
    org_img = org_img.transpose((1, 2, 0))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log+'/'+args.log + '_x_visual.jpg', org_img)
    
    
    org_img = x2[0,:,:,:].cpu()  
    org_img = org_img.data.numpy()*128+127.5
    org_img = org_img.transpose((1, 2, 0))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log+'/'+args.log + '_x_depth.jpg', org_img)
    
    
    org_img = x3[0,:,:,:].cpu()  
    org_img = org_img.data.numpy()*128+127.5
    org_img = org_img.transpose((1, 2, 0))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log+'/'+args.log + '_x_ir.jpg', org_img)
    


# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')
    
    echo_batches = args.echo_batches



    print('finetune!\n')
    log_file.write('finetune!\n')
    log_file.flush()
    
    
    model = ViT_AvgPool_3modal_CrossAtten_Channel()
    
    
    model = model.cuda()
    
    lr = args.lr
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(model) 
    
    
  
    criterion = nn.CrossEntropyLoss()

    
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        loss_absolute_RGB = AvgrageMeter()
        
        
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
         
        train_data = Spoofing_train(train_CASIA_SURF_CeFA_list, root_FAS_dir, transform=transforms.Compose([RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

        for i, sample_batched in enumerate(dataloader_train):
            
            # get the inputs
            inputs = sample_batched['image_x'].cuda() 
            spoof_label = sample_batched['spoofing_label'].cuda()
            binary_mask = sample_batched['map_x1'].cuda()
            
            inputs_depth = sample_batched['image_x_depth'].cuda()
            inputs_ir = sample_batched['image_x_ir'].cuda()

            optimizer.zero_grad()
            
            logits =  model(inputs, inputs_depth, inputs_ir)
            
            loss_global =  criterion(logits, spoof_label.squeeze(-1))
 
             
            loss =  loss_global
             
            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            loss_absolute.update(loss_global.data, n)
            loss_contra.update(loss_global.data, n)
            loss_absolute_RGB.update(loss_global.data, n)
        

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
                
                # visualization
                FeatureMap2Heatmap(inputs, inputs_depth, inputs_ir)

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f , CE1= %.4f , CE2= %.4f \n' % (epoch + 1, i + 1, lr,  loss_absolute.avg,  loss_contra.avg,  loss_absolute_RGB.avg))
        
        # whole epoch average
        log_file.write('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f , CE1= %.4f , CE2= %.4f \n' % (epoch + 1, i + 1, lr,  loss_absolute.avg,  loss_contra.avg,  loss_absolute_RGB.avg))
        log_file.flush()
    
                    
        #### validation/test
        
        epoch_test = 2
        if epoch % epoch_test == epoch_test-1:    # test every 5 epochs  
            model.eval()
            
            with torch.no_grad():
                
                ###############################################################################################
                '''                                            P1    RGB                           '''
                ##############################################################################################
                
                ###########################################
                '''                val             '''
                ##########################################
                # val for CASIA_SURF_CeFA
                test_data = Spoofing_valtest(val_CASIA_SURF_CeFA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    #log_file.write('test SiW i= %d \n' % (i))
                    
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                CASIA_SURF_CeFA_val_filename = args.log+'/'+ args.log+'_CASIA_SURF_CeFA_val.txt'
                with open(CASIA_SURF_CeFA_val_filename, 'w') as file:
                    file.writelines(map_score_list)
                
                
                
                
                ###########################################
                '''                test             '''
                ##########################################
                # Intra-test for CASIA_SURF_CeFA
                test_data = Spoofing_valtest(test_CASIA_SURF_CeFA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    #log_file.write('test SiW i= %d \n' % (i))
                    
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                CASIA_SURF_CeFA_test_filename = args.log+'/'+ args.log+'_CASIA_SURF_CeFA_test.txt' 
                with open(CASIA_SURF_CeFA_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                
                ##########################################    
                # Inter-test for WMCA
                test_data = Spoofing_valtest(test_WMCA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_filename = args.log+'/'+ args.log+'_WMCA_test.txt' 
                with open(WMCA_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                    
                
                ##########################################    
                # sub-testing for WMCA
                
                
                test_data = Spoofing_valtest(test_WMCA_fakehead_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_fakehead_filename = args.log+'/'+ args.log+'_WMCA_test_fakehead.txt' 
                with open(WMCA_test_fakehead_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_flexiblemask_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_flexiblemask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                 
                WMCA_test_flexiblemask_filename = args.log+'/'+ args.log+'_WMCA_test_flexiblemask.txt' 
                with open(WMCA_test_flexiblemask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_glasses_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_glasses_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
              
                WMCA_test_glasses_filename = args.log+'/'+ args.log+'_WMCA_test_glasses.txt' 
                with open(WMCA_test_glasses_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_papermask_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_papermask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                 
                WMCA_test_papermask_filename = args.log+'/'+ args.log+'_WMCA_test_papermask.txt' 
                with open(WMCA_test_papermask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_print_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                test_data = Spoofing_valtest(test_WMCA_print_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_print_filename = args.log+'/'+ args.log+'_WMCA_test_print.txt' 
                with open(WMCA_test_print_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_replay_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                test_data = Spoofing_valtest(test_WMCA_replay_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_replay_filename = args.log+'/'+ args.log+'_WMCA_test_replay.txt' 
                with open(WMCA_test_replay_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                
                #test_WMCA_rigidmask_list = 'FlexModal_Protocols/WMCA_test.txt'

                test_data = Spoofing_valtest(test_WMCA_rigidmask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
              
                WMCA_test_rigidmask_filename = args.log+'/'+ args.log+'_WMCA_test_rigidmask.txt' 
                with open(WMCA_test_rigidmask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                   
                
                
                ##########################################################################   
                #       Performance measurement for both intra- and inter-testings
                ##########################################################################   
                ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001, ACER_WMCA, APCER_WMCA, BPCER_WMCA, ACER_WMCA_testBest, TPR_FPR001, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask = performances_FAS_MultiModal(CASIA_SURF_CeFA_val_filename, CASIA_SURF_CeFA_test_filename, WMCA_test_filename, WMCA_test_fakehead_filename, WMCA_test_flexiblemask_filename, WMCA_test_glasses_filename, WMCA_test_papermask_filename, WMCA_test_print_filename, WMCA_test_replay_filename, WMCA_test_rigidmask_filename)
                
                print('\n\n P1  RGB: \n epoch:%d, Intra-testing!\n CASIA_SURF_CeFA:  ACER= %.4f, ACER_testBest= %.4f, TPR_FPR0001= %.4f' % (epoch + 1, ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001))
                log_file.write('\n\n P1  RGB: \n epoch:%d, Intra-testing!\n CASIA_SURF_CeFA:  ACER= %.4f, ACER_testBest= %.4f, TPR_FPR0001= %.4f\n' % (epoch + 1, ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001))
                
                print('epoch:%d, Cross-testing!\n WMCA:  ACER= %.4f, APCER_WMCA= %.4f, BPCER_WMCA= %.4f, ACER_testBest= %.4f, TPR_FPR001= %.4f' % (epoch + 1, ACER_WMCA, APCER_WMCA, BPCER_WMCA, ACER_WMCA_testBest, TPR_FPR001))
                log_file.write('epoch:%d, Cross-testing!\n WMCA:  ACER= %.4f, APCER_WMCA= %.4f, BPCER_WMCA= %.4f, ACER_testBest= %.4f, TPR_FPR001= %.4f\n' % (epoch + 1, ACER_WMCA, APCER_WMCA, BPCER_WMCA, ACER_WMCA_testBest, TPR_FPR001))
                
                
                print('epoch:%d, WMCA: APCER_fakehead= %.4f, APCER_flexiblemask= %.4f, APCER_glasses= %.4f, APCER_papermask= %.4f, APCER_print= %.4f, APCER_replay= %.4f, APCER_rigidmask= %.4f' % (epoch + 1, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask))
                log_file.write('epoch:%d, WMCA: APCER_fakehead= %.4f, APCER_flexiblemask= %.4f, APCER_glasses= %.4f, APCER_papermask= %.4f, APCER_print= %.4f, APCER_replay= %.4f, APCER_rigidmask= %.4f\n\n' % (epoch + 1, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask))
                
                
                log_file.flush()
                
                
                
                ###############################################################################################
                '''                                            P2    RGBD                            '''
                ##############################################################################################
                
                ###########################################
                '''                val             '''
                ##########################################
                # val for CASIA_SURF_CeFA
                test_data = Spoofing_valtest(val_CASIA_SURF_CeFA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    #log_file.write('test SiW i= %d \n' % (i))
                    
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                CASIA_SURF_CeFA_val_filename = args.log+'/'+ args.log+'_CASIA_SURF_CeFA_val.txt'
                with open(CASIA_SURF_CeFA_val_filename, 'w') as file:
                    file.writelines(map_score_list)
                
                
                
                
                ###########################################
                '''                test             '''
                ##########################################
                # Intra-test for CASIA_SURF_CeFA
                test_data = Spoofing_valtest(test_CASIA_SURF_CeFA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    #log_file.write('test SiW i= %d \n' % (i))
                    
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                CASIA_SURF_CeFA_test_filename = args.log+'/'+ args.log+'_CASIA_SURF_CeFA_test.txt' 
                with open(CASIA_SURF_CeFA_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                
                ##########################################    
                # Inter-test for WMCA
                test_data = Spoofing_valtest(test_WMCA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_filename = args.log+'/'+ args.log+'_WMCA_test.txt' 
                with open(WMCA_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                    
                
                ##########################################    
                # sub-testing for WMCA
                
                
                test_data = Spoofing_valtest(test_WMCA_fakehead_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_fakehead_filename = args.log+'/'+ args.log+'_WMCA_test_fakehead.txt' 
                with open(WMCA_test_fakehead_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_flexiblemask_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_flexiblemask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                 
                WMCA_test_flexiblemask_filename = args.log+'/'+ args.log+'_WMCA_test_flexiblemask.txt' 
                with open(WMCA_test_flexiblemask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_glasses_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_glasses_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
              
                WMCA_test_glasses_filename = args.log+'/'+ args.log+'_WMCA_test_glasses.txt' 
                with open(WMCA_test_glasses_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_papermask_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_papermask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                 
                WMCA_test_papermask_filename = args.log+'/'+ args.log+'_WMCA_test_papermask.txt' 
                with open(WMCA_test_papermask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_print_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                test_data = Spoofing_valtest(test_WMCA_print_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_print_filename = args.log+'/'+ args.log+'_WMCA_test_print.txt' 
                with open(WMCA_test_print_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_replay_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                test_data = Spoofing_valtest(test_WMCA_replay_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_replay_filename = args.log+'/'+ args.log+'_WMCA_test_replay.txt' 
                with open(WMCA_test_replay_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                
                #test_WMCA_rigidmask_list = 'FlexModal_Protocols/WMCA_test.txt'

                test_data = Spoofing_valtest(test_WMCA_rigidmask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, image_x_zeros)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
              
                WMCA_test_rigidmask_filename = args.log+'/'+ args.log+'_WMCA_test_rigidmask.txt' 
                with open(WMCA_test_rigidmask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                   
                
                
                ##########################################################################   
                #       Performance measurement for both intra- and inter-testings
                ##########################################################################   
                ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001, ACER_WMCA, APCER_WMCA, BPCER_WMCA, ACER_WMCA_testBest, TPR_FPR001, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask = performances_FAS_MultiModal(CASIA_SURF_CeFA_val_filename, CASIA_SURF_CeFA_test_filename, WMCA_test_filename, WMCA_test_fakehead_filename, WMCA_test_flexiblemask_filename, WMCA_test_glasses_filename, WMCA_test_papermask_filename, WMCA_test_print_filename, WMCA_test_replay_filename, WMCA_test_rigidmask_filename)
                
                print('\n\n P2   RGBD: \n epoch:%d, Intra-testing!\n CASIA_SURF_CeFA:  ACER= %.4f, ACER_testBest= %.4f, TPR_FPR0001= %.4f' % (epoch + 1, ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001))
                log_file.write('\n\n P2  RGBD: \n epoch:%d, Intra-testing!\n CASIA_SURF_CeFA:  ACER= %.4f, ACER_testBest= %.4f, TPR_FPR0001= %.4f\n' % (epoch + 1, ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001))
                
                print('epoch:%d, Cross-testing!\n WMCA:  ACER= %.4f, APCER_WMCA= %.4f, BPCER_WMCA= %.4f, ACER_testBest= %.4f, TPR_FPR001= %.4f' % (epoch + 1, ACER_WMCA, APCER_WMCA, BPCER_WMCA, ACER_WMCA_testBest, TPR_FPR001))
                log_file.write('epoch:%d, Cross-testing!\n WMCA:  ACER= %.4f, APCER_WMCA= %.4f, BPCER_WMCA= %.4f, ACER_testBest= %.4f, TPR_FPR001= %.4f\n' % (epoch + 1, ACER_WMCA, APCER_WMCA, BPCER_WMCA, ACER_WMCA_testBest, TPR_FPR001))
                
                
                print('epoch:%d, WMCA: APCER_fakehead= %.4f, APCER_flexiblemask= %.4f, APCER_glasses= %.4f, APCER_papermask= %.4f, APCER_print= %.4f, APCER_replay= %.4f, APCER_rigidmask= %.4f' % (epoch + 1, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask))
                log_file.write('epoch:%d, WMCA: APCER_fakehead= %.4f, APCER_flexiblemask= %.4f, APCER_glasses= %.4f, APCER_papermask= %.4f, APCER_print= %.4f, APCER_replay= %.4f, APCER_rigidmask= %.4f\n\n' % (epoch + 1, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask))
                
                
                log_file.flush()
                
                
                
                ###############################################################################################
                '''                                            P3      RGBIR                         '''
                ##############################################################################################
                
                ###########################################
                '''                val             '''
                ##########################################
                # val for CASIA_SURF_CeFA
                test_data = Spoofing_valtest(val_CASIA_SURF_CeFA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    #log_file.write('test SiW i= %d \n' % (i))
                    
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                CASIA_SURF_CeFA_val_filename = args.log+'/'+ args.log+'_CASIA_SURF_CeFA_val.txt'
                with open(CASIA_SURF_CeFA_val_filename, 'w') as file:
                    file.writelines(map_score_list)
                
                
                
                
                ###########################################
                '''                test             '''
                ##########################################
                # Intra-test for CASIA_SURF_CeFA
                test_data = Spoofing_valtest(test_CASIA_SURF_CeFA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    #log_file.write('test SiW i= %d \n' % (i))
                    
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                CASIA_SURF_CeFA_test_filename = args.log+'/'+ args.log+'_CASIA_SURF_CeFA_test.txt' 
                with open(CASIA_SURF_CeFA_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                
                ##########################################    
                # Inter-test for WMCA
                test_data = Spoofing_valtest(test_WMCA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_filename = args.log+'/'+ args.log+'_WMCA_test.txt' 
                with open(WMCA_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                    
                
                ##########################################    
                # sub-testing for WMCA
                
                
                test_data = Spoofing_valtest(test_WMCA_fakehead_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_fakehead_filename = args.log+'/'+ args.log+'_WMCA_test_fakehead.txt' 
                with open(WMCA_test_fakehead_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_flexiblemask_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_flexiblemask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                 
                WMCA_test_flexiblemask_filename = args.log+'/'+ args.log+'_WMCA_test_flexiblemask.txt' 
                with open(WMCA_test_flexiblemask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_glasses_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_glasses_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
              
                WMCA_test_glasses_filename = args.log+'/'+ args.log+'_WMCA_test_glasses.txt' 
                with open(WMCA_test_glasses_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_papermask_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_papermask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                 
                WMCA_test_papermask_filename = args.log+'/'+ args.log+'_WMCA_test_papermask.txt' 
                with open(WMCA_test_papermask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_print_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                test_data = Spoofing_valtest(test_WMCA_print_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_print_filename = args.log+'/'+ args.log+'_WMCA_test_print.txt' 
                with open(WMCA_test_print_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_replay_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                test_data = Spoofing_valtest(test_WMCA_replay_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_replay_filename = args.log+'/'+ args.log+'_WMCA_test_replay.txt' 
                with open(WMCA_test_replay_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                
                #test_WMCA_rigidmask_list = 'FlexModal_Protocols/WMCA_test.txt'

                test_data = Spoofing_valtest(test_WMCA_rigidmask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
                    image_x_zeros = sample_batched['image_x_zeros'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, image_x_zeros, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
              
                WMCA_test_rigidmask_filename = args.log+'/'+ args.log+'_WMCA_test_rigidmask.txt' 
                with open(WMCA_test_rigidmask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                   
                
                
                ##########################################################################   
                #       Performance measurement for both intra- and inter-testings
                ##########################################################################   
                ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001, ACER_WMCA, APCER_WMCA, BPCER_WMCA, ACER_WMCA_testBest, TPR_FPR001, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask = performances_FAS_MultiModal(CASIA_SURF_CeFA_val_filename, CASIA_SURF_CeFA_test_filename, WMCA_test_filename, WMCA_test_fakehead_filename, WMCA_test_flexiblemask_filename, WMCA_test_glasses_filename, WMCA_test_papermask_filename, WMCA_test_print_filename, WMCA_test_replay_filename, WMCA_test_rigidmask_filename)
                
                print('\n\n P3   RGBIR: \n epoch:%d, Intra-testing!\n CASIA_SURF_CeFA:  ACER= %.4f, ACER_testBest= %.4f, TPR_FPR0001= %.4f' % (epoch + 1, ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001))
                log_file.write('\n\n P3   RGBIR: \n epoch:%d, Intra-testing!\n CASIA_SURF_CeFA:  ACER= %.4f, ACER_testBest= %.4f, TPR_FPR0001= %.4f\n' % (epoch + 1, ACER_CASIA_SURF_CeFA, ACER_CASIA_SURF_CeFA_testBest, TPR_FPR0001))
                
                print('epoch:%d, Cross-testing!\n WMCA:  ACER= %.4f, APCER_WMCA= %.4f, BPCER_WMCA= %.4f, ACER_testBest= %.4f, TPR_FPR001= %.4f' % (epoch + 1, ACER_WMCA, APCER_WMCA, BPCER_WMCA, ACER_WMCA_testBest, TPR_FPR001))
                log_file.write('epoch:%d, Cross-testing!\n WMCA:  ACER= %.4f, APCER_WMCA= %.4f, BPCER_WMCA= %.4f, ACER_testBest= %.4f, TPR_FPR001= %.4f\n' % (epoch + 1, ACER_WMCA, APCER_WMCA, BPCER_WMCA, ACER_WMCA_testBest, TPR_FPR001))
                
                
                print('epoch:%d, WMCA: APCER_fakehead= %.4f, APCER_flexiblemask= %.4f, APCER_glasses= %.4f, APCER_papermask= %.4f, APCER_print= %.4f, APCER_replay= %.4f, APCER_rigidmask= %.4f' % (epoch + 1, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask))
                log_file.write('epoch:%d, WMCA: APCER_fakehead= %.4f, APCER_flexiblemask= %.4f, APCER_glasses= %.4f, APCER_papermask= %.4f, APCER_print= %.4f, APCER_replay= %.4f, APCER_rigidmask= %.4f\n\n' % (epoch + 1, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask))
                
                
                log_file.flush()
                
                
                
                
                ###############################################################################################
                '''                                            P4    RGBDIR                               '''
                ##############################################################################################
                
                ###########################################
                '''                val             '''
                ##########################################
                # val for CASIA_SURF_CeFA
                test_data = Spoofing_valtest(val_CASIA_SURF_CeFA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    #log_file.write('test SiW i= %d \n' % (i))
                    
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                CASIA_SURF_CeFA_val_filename = args.log+'/'+ args.log+'_CASIA_SURF_CeFA_val.txt'
                with open(CASIA_SURF_CeFA_val_filename, 'w') as file:
                    file.writelines(map_score_list)
                
                
                
                
                ###########################################
                '''                test             '''
                ##########################################
                # Intra-test for CASIA_SURF_CeFA
                test_data = Spoofing_valtest(test_CASIA_SURF_CeFA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    #log_file.write('test SiW i= %d \n' % (i))
                    
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda()
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                CASIA_SURF_CeFA_test_filename = args.log+'/'+ args.log+'_CASIA_SURF_CeFA_test.txt' 
                with open(CASIA_SURF_CeFA_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                
                ##########################################    
                # Inter-test for WMCA
                test_data = Spoofing_valtest(test_WMCA_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_filename = args.log+'/'+ args.log+'_WMCA_test.txt' 
                with open(WMCA_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                    
                
                ##########################################    
                # sub-testing for WMCA
                
                
                test_data = Spoofing_valtest(test_WMCA_fakehead_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_fakehead_filename = args.log+'/'+ args.log+'_WMCA_test_fakehead.txt' 
                with open(WMCA_test_fakehead_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_flexiblemask_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_flexiblemask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                 
                WMCA_test_flexiblemask_filename = args.log+'/'+ args.log+'_WMCA_test_flexiblemask.txt' 
                with open(WMCA_test_flexiblemask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_glasses_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_glasses_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
              
                WMCA_test_glasses_filename = args.log+'/'+ args.log+'_WMCA_test_glasses.txt' 
                with open(WMCA_test_glasses_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_papermask_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                
                test_data = Spoofing_valtest(test_WMCA_papermask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                 
                WMCA_test_papermask_filename = args.log+'/'+ args.log+'_WMCA_test_papermask.txt' 
                with open(WMCA_test_papermask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_print_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                test_data = Spoofing_valtest(test_WMCA_print_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_print_filename = args.log+'/'+ args.log+'_WMCA_test_print.txt' 
                with open(WMCA_test_print_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                #test_WMCA_replay_list = 'FlexModal_Protocols/WMCA_test.txt'
                
                test_data = Spoofing_valtest(test_WMCA_replay_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                WMCA_test_replay_filename = args.log+'/'+ args.log+'_WMCA_test_replay.txt' 
                with open(WMCA_test_replay_filename, 'w') as file:
                    file.writelines(map_score_list) 
                
                
                #test_WMCA_rigidmask_list = 'FlexModal_Protocols/WMCA_test.txt'

                test_data = Spoofing_valtest(test_WMCA_rigidmask_list, root_FAS_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    inputs_depth = sample_batched['image_x_depth'].cuda()
                    inputs_ir = sample_batched['image_x_ir'].cuda() 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    logits  =  model(inputs, inputs_depth, inputs_ir)
                    for test_batch in range(inputs.shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits)[test_batch][1]
                        map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
              
                WMCA_test_rigidmask_filename = args.log+'/'+ args.log+'_WMCA_test_rigidmask.txt' 
                with open(WMCA_test_rigidmask_filename, 'w') as file:
                    file.writelines(map_score_list) 
                   
                
                
                ##########################################################################   
                #       Performance measurement for both intra- and inter-testings
                ##########################################################################   
                ACER_CASIA_SURF_CeFA, TPR_FPR0001, ACER_WMCA, APCER_WMCA, BPCER_WMCA, TPR_FPR001, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask = performances_FAS_MultiModal(CASIA_SURF_CeFA_val_filename, CASIA_SURF_CeFA_test_filename, WMCA_test_filename, WMCA_test_fakehead_filename, WMCA_test_flexiblemask_filename, WMCA_test_glasses_filename, WMCA_test_papermask_filename, WMCA_test_print_filename, WMCA_test_replay_filename, WMCA_test_rigidmask_filename)
                
                print('\n\n P4  RGBDIR: \n epoch:%d, Intra-testing!\n CASIA_SURF_CeFA:  ACER= %.4f, TPR_FPR0001= %.4f' % (epoch + 1, ACER_CASIA_SURF_CeFA, TPR_FPR0001))
                log_file.write('\n\n P4   RGBDIR: \n epoch:%d, Intra-testing!\n CASIA_SURF_CeFA:  ACER= %.4f, TPR_FPR0001= %.4f\n' % (epoch + 1, ACER_CASIA_SURF_CeFA, TPR_FPR0001))
                
                print('epoch:%d, Cross-testing!\n WMCA:  ACER= %.4f, APCER_WMCA= %.4f, BPCER_WMCA= %.4f, TPR_FPR001= %.4f' % (epoch + 1, ACER_WMCA, APCER_WMCA, BPCER_WMCA, TPR_FPR001))
                log_file.write('epoch:%d, Cross-testing!\n WMCA:  ACER= %.4f, APCER_WMCA= %.4f, BPCER_WMCA= %.4f, TPR_FPR001= %.4f\n' % (epoch + 1, ACER_WMCA, APCER_WMCA, BPCER_WMCA, TPR_FPR001))
                
                
                print('epoch:%d, WMCA: APCER_fakehead= %.4f, APCER_flexiblemask= %.4f, APCER_glasses= %.4f, APCER_papermask= %.4f, APCER_print= %.4f, APCER_replay= %.4f, APCER_rigidmask= %.4f' % (epoch + 1, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask))
                log_file.write('epoch:%d, WMCA: APCER_fakehead= %.4f, APCER_flexiblemask= %.4f, APCER_glasses= %.4f, APCER_papermask= %.4f, APCER_print= %.4f, APCER_replay= %.4f, APCER_rigidmask= %.4f\n\n\n' % (epoch + 1, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask))
                
                
                log_file.flush()
                
                
          


    print('Finished Training')
    log_file.close()
  
 
  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=64, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=100, help='how many batches display once')  
    parser.add_argument('--epochs', type=int, default=30, help='total training epochs')
    parser.add_argument('--log', type=str, default="ViT_AvgPool_CrossAtten_Channel_RGBDIR_P1234", help='log and save model name')

    args = parser.parse_args()
    train_test()
