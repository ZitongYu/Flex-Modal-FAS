import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pandas as pd
import pdb


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt




def get_threshold(score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        #pdb.set_trace()
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type==1:
            num_real += 1
        else:
            num_fake += 1

    min_error = count    # account ACER (or ACC)
    min_threshold = 0.0
    min_ACC = 0.0
    min_ACER = 0.0
    min_APCER = 0.0
    min_BPCER = 0.0
    
    
    for d in data:
        threshold = d['map_score']
        
        type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
        type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])
        
        ACC = 1-(type1 + type2) / count
        APCER = type2 / num_fake
        BPCER = type1 / num_real
        ACER = (APCER + BPCER) / 2.0
        
        if ACER < min_error:
            min_error = ACER
            min_threshold = threshold
            min_ACC = ACC
            min_ACER = ACER
            min_APCER = APCER
            min_BPCER = min_BPCER

    # print(min_error, min_threshold)
    return min_threshold, min_ACC, min_APCER, min_BPCER, min_ACER



def test_threshold_based(threshold, score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type==1:
            num_real += 1
        else:
            num_fake += 1
    
 
    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])
    
    ACC = 1-(type1 + type2) / count
    APCER = type2 / num_fake
    BPCER = type1 / num_real
    ACER = (APCER + BPCER) / 2.0
    
    return ACC, APCER, BPCER, ACER



def get_err_threhold(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1); 
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0
  
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    

    #print(err, best_th)
    return err, best_th


# TPR@FPR=0.001
def tpr_fpr0001_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    #TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    #TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return TR1
    

# TPR@FPR=0.01
def tpr_fpr001_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    #TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    #TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return TR1




def performances_FAS_MultiModal(CASIA_SURF_CeFA_val_filename, CASIA_SURF_CeFA_test_filename, WMCA_test_filename, WMCA_test_fakehead_filename, WMCA_test_flexiblemask_filename, WMCA_test_glasses_filename, WMCA_test_papermask_filename, WMCA_test_print_filename, WMCA_test_replay_filename, WMCA_test_rigidmask_filename):

    # val 
    with open(CASIA_SURF_CeFA_val_filename, 'r') as file1:
        lines = file1.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  #label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)
    
    ################################################
    # intra-test    ACER_CASIA_SURF_CeFA
    with open(CASIA_SURF_CeFA_test_filename, 'r') as file2:
        lines = file2.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    # test based on val_threshold     
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    

    test_APCER = type2 / num_fake
    test_BPCER = type1 / num_real
    ACER_CASIA_SURF_CeFA = (test_APCER + test_BPCER) / 2.0
    
    
    
    # TPR_FPR0001
    TPR_FPR0001 = tpr_fpr0001_funtion(test_labels, test_scores)
    
    
    
    ################################################
    # cross-test    WMCA
    
    val_threshold = 0.5
    
    with open(WMCA_test_filename, 'r') as file3:
        lines = file3.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    # test based on val_threshold     
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    

    test_APCER = type2 / num_fake
    test_BPCER = type1 / num_real
    ACER_WMCA = (test_APCER + test_BPCER) / 2.0
    
    
    
    # TPR_FPR001
    TPR_FPR001 = tpr_fpr001_funtion(test_labels, test_scores)
    
    
    # cross-test    fakehead
    with open(WMCA_test_fakehead_filename, 'r') as file5:
        lines = file5.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    # test based on val_threshold     
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    APCER_fakehead = type2 / num_fake
    
    
    # cross-test    flexiblemask
    with open(WMCA_test_flexiblemask_filename, 'r') as file6:
        lines = file6.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    # test based on val_threshold     
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    APCER_flexiblemask = type2 / num_fake
    
    
    # cross-test    glasses
    with open(WMCA_test_glasses_filename, 'r') as file7:
        lines = file7.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    # test based on val_threshold     
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    APCER_glasses = type2 / num_fake
    
    
    # cross-test    papermask
    with open(WMCA_test_papermask_filename, 'r') as file8:
        lines = file8.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    # test based on val_threshold     
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    APCER_papermask = type2 / num_fake
    
    
    # cross-test    print
    with open(WMCA_test_print_filename, 'r') as file9:
        lines = file9.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    # test based on val_threshold     
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    APCER_print = type2 / num_fake
    
    
    
    # cross-test    replay
    with open(WMCA_test_replay_filename, 'r') as file10:
        lines = file10.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    # test based on val_threshold     
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    APCER_replay = type2 / num_fake
    
    
    
    # cross-test    rigidmask
    with open(WMCA_test_rigidmask_filename, 'r') as file11:
        lines = file11.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    # test based on val_threshold     
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    APCER_rigidmask = type2 / num_fake
    
    
    
    return ACER_CASIA_SURF_CeFA, TPR_FPR0001, ACER_WMCA, test_APCER, test_BPCER, TPR_FPR001, APCER_fakehead, APCER_flexiblemask, APCER_glasses, APCER_papermask, APCER_print, APCER_replay,  APCER_rigidmask



