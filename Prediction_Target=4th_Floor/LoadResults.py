# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:38:17 2022

@author: Sheng Shi
"""

import torch
import pickle
from utils_MyNet_train.MyNet_train10 import Test_MyNet
import numpy as np
from tqdm import tqdm
from alarm_deepsvdd.utils_SVDD.DeepSVDD import DeepSVDD
import os

# Transform data format
def ToNN(data,window_size,prednum): 
    train = data
    data = np.zeros([train.shape[0]-window_size,window_size,train.shape[1]])
    for i in range(len(train) - window_size):
        data[i,:,:] = train[(i+1): (i+window_size+1),:]
        data[i,:,prednum] = train[i: (i + window_size),prednum].T
    data = np.array(data).astype('float64')
    train_x = torch.from_numpy(data).permute(0,2,1)
    train_y = torch.from_numpy(train[window_size:,prednum])
    return train_x,train_y

window_size_all = {0:41, 2:51, 4:51, 6:56, 8:96, 10:96, 20:96}
in_channel = range(0,16)
prednum    = [12,13,14,15]
outputdim  = len(prednum)
dt = 0.05
scale1_1_summary = np.zeros((7,5))
scale2_1_summary = np.zeros((7,5))
scale3_1_summary = np.zeros((7,5))
scale4_1_summary = np.zeros((7,5))
mad_summary = np.zeros((7,6))
mfad_summary = np.zeros((7,6))
confuse_index = np.zeros((7,4))
confuse_index_mean = np.zeros((7,1))
scale0_mean = np.zeros((7,4))
k = 0
path_pre = os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))
for noise_level in [0,2,4,6,8,10,20]:
    print('Loading/Calculating results for the case when noise level = {}...'.format(noise_level))
    window_size = window_size_all[noise_level]
    kmlp        = window_size-1
    best_net = torch.load('train/%d/predictor_finetune_%d.pth' %(noise_level,window_size))
    with open('train/%d/train_loss_history_finetune_%d.pth' %(noise_level,window_size), 'rb') as f:
        train_loss_history = pickle.load(f)
    with open('train/%d/valid_loss_history_finetune_%d.pth' %(noise_level,window_size), 'rb') as f:
        valid_loss_history = pickle.load(f)   
    with open('train/%d/valid_loss_wb_history_finetune_%d.pth' %(noise_level,window_size), 'rb') as f:
        valid_loss_wb_history = pickle.load(f) 
    with open('train/%d/valid_loss_scale_history_finetune_%d.pth' %(noise_level,window_size), 'rb') as f:
        valid_loss_scale_history = pickle.load(f)
    if not os.path.isfile('alarm_deepsvdd/innovations/%d/e0_1.csv' %(noise_level)):
        with tqdm(total=9) as pbar:
            pbar.set_description('Generating innovations:')
            ## Undamaged feature for training
            data0_1 = np.loadtxt('{}/{}/train.csv'.format(path_pre,noise_level),dtype="float",delimiter=',')[:,in_channel]
            scale = max(np.sqrt(np.mean(data0_1**2,axis=0)))
            data0_1 = data0_1/scale
            data0_2 = np.loadtxt('{}/{}/validation.csv'.format(path_pre,noise_level),dtype="float",delimiter=',')[:,in_channel]
            data0_2 = data0_2/scale
            Train_x,Train_y = ToNN(data0_1,kmlp,prednum)
            Valid_x,Valid_y = ToNN(data0_2,kmlp,prednum)
            feature0_1 = Test_MyNet(best_net,Train_x,Train_y); feature0_1 = feature0_1.numpy()
            np.savetxt('alarm_deepsvdd/innovations/%d/e0_1.csv' %(noise_level), feature0_1, delimiter=',')
            pbar.update(1)
            ## Undamaged feature for validation
            feature0_2 = Test_MyNet(best_net,Valid_x,Valid_y); feature0_2 = feature0_2.numpy()
            np.savetxt('alarm_deepsvdd/innovations/%d/e0_2.csv' %(noise_level), feature0_2, delimiter=',')
            pbar.update(1)
            ## Undamaged feature for testing 
            data0_3 = np.loadtxt('{}/{}/test0.csv'.format(path_pre,noise_level),dtype="float",delimiter=',')[:,in_channel]
            np.random.seed(3)
            data0_3 = np.concatenate((data0_3/scale,np.random.standard_normal(72000).reshape(-1,1)),axis=1)
            Test0_x,Test0_y = ToNN(data0_3,kmlp,prednum)
            feature0_3 = Test_MyNet(best_net,Test0_x,Test0_y)
            feature0_3 = feature0_3.numpy() 
            np.savetxt('alarm_deepsvdd/innovations/%d/e0_3.csv' %(noise_level), feature0_3, delimiter=',')
            pbar.update(1)
            ## Damaged feature for testing (DP 1)
            data1 = np.loadtxt('{}/{}/test{}.csv'.format(path_pre,noise_level,1),dtype="float",delimiter=',')[:,in_channel]
            np.random.seed(10)
            data1 = np.concatenate((data1/scale,np.random.standard_normal(72000).reshape(-1,1)),axis=1)
            Test1_x,Test1_y = ToNN(data1,kmlp,prednum)
            feature1 = Test_MyNet(best_net,Test1_x,Test1_y)
            feature1 = feature1.numpy() 
            np.savetxt('alarm_deepsvdd/innovations/%d/e1.csv' %(noise_level), feature1, delimiter=',')
            pbar.update(1)
            ## Damaged feature for testing (DP 2)
            data2 = np.loadtxt('{}/{}/test{}.csv'.format(path_pre,noise_level,2),dtype="float",delimiter=',')[:,in_channel]
            np.random.seed(20)
            data2 = np.concatenate((data2/scale,np.random.standard_normal(72000).reshape(-1,1)),axis=1)
            Test2_x,Test2_y = ToNN(data2,kmlp,prednum)
            feature2 = Test_MyNet(best_net,Test2_x,Test2_y)
            feature2 = feature2.numpy() 
            np.savetxt('alarm_deepsvdd/innovations/%d/e2.csv' %(noise_level), feature2, delimiter=',')
            pbar.update(1)
            ## Damaged feature for testing (DP 3)
            data3 = np.loadtxt('{}/{}/test{}.csv'.format(path_pre,noise_level,3),dtype="float",delimiter=',')[:,in_channel]
            np.random.seed(30)
            data3 = np.concatenate((data3/scale,np.random.standard_normal(72000).reshape(-1,1)),axis=1)
            Test3_x,Test3_y = ToNN(data3,kmlp,prednum)
            feature3 = Test_MyNet(best_net,Test3_x,Test3_y)
            feature3 = feature3.numpy() 
            np.savetxt('alarm_deepsvdd/innovations/%d/e3.csv' %(noise_level), feature3, delimiter=',')
            pbar.update(1)
            ## Damaged feature for testing (DP 4)
            data4 = np.loadtxt('{}/{}/test{}.csv'.format(path_pre,noise_level,4),dtype="float",delimiter=',')[:,in_channel]
            np.random.seed(40)
            data4 = np.concatenate((data4/scale,np.random.standard_normal(72000).reshape(-1,1)),axis=1)
            Test4_x,Test4_y = ToNN(data4,kmlp,prednum)
            feature4 = Test_MyNet(best_net,Test4_x,Test4_y)
            feature4 = feature4.numpy() 
            np.savetxt('alarm_deepsvdd/innovations/%d/e4.csv' %(noise_level), feature4, delimiter=',')
            pbar.update(1)
            ## Damaged feature for testing (DP 5)
            data5 = np.loadtxt('{}/{}/test{}.csv'.format(path_pre,noise_level,5),dtype="float",delimiter=',')[:,in_channel]
            np.random.seed(50)
            data5 = np.concatenate((data5/scale,np.random.standard_normal(72000).reshape(-1,1)),axis=1)
            Test5_x,Test5_y = ToNN(data5,kmlp,prednum)
            feature5 = Test_MyNet(best_net,Test5_x,Test5_y)
            feature5 = feature5.numpy() 
            np.savetxt('alarm_deepsvdd/innovations/%d/e5.csv' %(noise_level), feature5, delimiter=',')
            pbar.update(1)
            ## Damaged feature for testing (DP 6)
            data6 = np.loadtxt('{}/{}/test{}.csv'.format(path_pre,noise_level,6),dtype="float",delimiter=',')[:,in_channel]
            np.random.seed(60)
            data6 = np.concatenate((data6/scale,np.random.standard_normal(72000).reshape(-1,1)),axis=1)
            Test6_x,Test6_y = ToNN(data6,kmlp,prednum)
            feature6 = Test_MyNet(best_net,Test6_x,Test6_y)
            feature6 = feature6.numpy() 
            np.savetxt('alarm_deepsvdd/innovations/%d/e6.csv' %(noise_level), feature6, delimiter=',')
            pbar.update(1)
    else:
        print('Loading innovations ...' )
        feature0_1 = np.loadtxt('alarm_deepsvdd/innovations/%d/e0_1.csv' %(noise_level),delimiter = ",")
        feature0_2 = np.loadtxt('alarm_deepsvdd/innovations/%d/e0_2.csv' %(noise_level),delimiter = ",")
        feature0_3 = np.loadtxt('alarm_deepsvdd/innovations/%d/e0_3.csv' %(noise_level),delimiter = ",")
        feature1 = np.loadtxt('alarm_deepsvdd/innovations/%d/e1.csv' %(noise_level),delimiter = ",")
        feature2 = np.loadtxt('alarm_deepsvdd/innovations/%d/e2.csv' %(noise_level),delimiter = ",")
        feature3 = np.loadtxt('alarm_deepsvdd/innovations/%d/e3.csv' %(noise_level),delimiter = ",")
        feature4 = np.loadtxt('alarm_deepsvdd/innovations/%d/e4.csv' %(noise_level),delimiter = ",")
        feature5 = np.loadtxt('alarm_deepsvdd/innovations/%d/e5.csv' %(noise_level),delimiter = ",")
        feature6 = np.loadtxt('alarm_deepsvdd/innovations/%d/e6.csv' %(noise_level),delimiter = ",")

    print('Loading decisoin-maker ...')
    DecisionMaker = DeepSVDD()
    DecisionMaker.build_network(outputdim,8+outputdim,12+outputdim)
    DecisionMaker.load_model('alarm_deepsvdd/saved_models/%d/DecisionMaker_finetune%d.pth' %(noise_level,window_size))
    R = DecisionMaker.R
    
    # Calculate mfad
    print('Calculating MAD&MFAD ...')
    ## Training mfad
    threshold = np.array(0.0)
    DecisionMaker.test((feature0_1))
    score0_1 = DecisionMaker.test_scores
    mfad0_1 = np.mean(score0_1>threshold)/dt
    ## Validation mfad
    DecisionMaker.test((feature0_2))
    score0_2 = DecisionMaker.test_scores
    mfad0_2 = np.mean(score0_2>threshold)/dt
    ## Testing mfad
    DecisionMaker.test((feature0_3))
    score0_3 = DecisionMaker.test_scores
    mfad0_3 = np.mean(score0_3>threshold)/dt
    # Calculate mad
    ## Trial 1
    DecisionMaker.test((feature1))
    score1 = DecisionMaker.test_scores
    mad1 = np.mean(score1>threshold)/dt
    ## Trial 2
    DecisionMaker.test((feature2))
    score2 = DecisionMaker.test_scores
    mad2 = np.mean(score2>threshold)/dt
    ## Trial 3
    DecisionMaker.test((feature3))
    score3 = DecisionMaker.test_scores
    mad3 = np.mean(score3>threshold)/dt
    ## Trial 4
    DecisionMaker.test((feature4))
    score4 = DecisionMaker.test_scores
    mad4 = np.mean(score4>threshold)/dt
    ## Trial 5
    DecisionMaker.test((feature5))
    score5 = DecisionMaker.test_scores
    mad5 = np.mean(score5>threshold)/dt
    ## Trial 6
    DecisionMaker.test((feature6))
    score6 = DecisionMaker.test_scores
    mad6 = np.mean(score6>threshold)/dt
    mad_summary[k,:] =[mad1,mad2,mad3,mad4,mad5,mad6]
    mfad_summary[k,:] = mfad0_3/[mad1,mad2,mad3,mad4,mad5,mad6]
        
    # Calculate Channel Importance Factors
    scale1_1 = best_net.scale1_1.scaler.detach().numpy()
    scale1_1_summary[k,:] = scale1_1
    scale2_1 = best_net.scale2_1.scaler.detach().numpy()
    scale2_1_summary[k,:] = scale2_1
    scale3_1 = best_net.scale3_1.scaler.detach().numpy()
    scale3_1_summary[k,:] = scale3_1
    scale4_1 = best_net.scale4_1.scaler.detach().numpy()
    scale4_1_summary[k,:] = scale4_1
    
    k += 1