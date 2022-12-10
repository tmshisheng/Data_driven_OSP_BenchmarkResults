# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 15:33:47 2021
At University of Toronto
@author: Sheng Shi
Retrain after pruning
"""

import torch
import torch.nn as nn

# Define a new layer 
class scale_conv(nn.Module):
    def __init__(self,nfeature):
        super(scale_conv,self).__init__()
        self.register_parameter('scaler',nn.Parameter(torch.ones(nfeature)))
    def forward(self,input):
        return torch.mul(input.transpose(1,2),self.scaler).transpose(1,2)


class inception(nn.Module):
    def __init__(self,cnn_in):
        super(inception,self).__init__()
        self.conv1 = nn.Conv1d(cnn_in,1,1,bias=False,padding=0)
        self.conv2 = nn.Conv1d(cnn_in,2,3,bias=False,padding=1)
        self.conv3 = nn.Conv1d(cnn_in,2,5,bias=False,padding=2)
        self.conv4 = nn.Conv1d(cnn_in,2,7,bias=False,padding=3)
        self.conv5 = nn.Conv1d(cnn_in,2,9,bias=False,padding=4)
        self.conv6 = nn.Conv1d(cnn_in,2,11,bias=False,padding=5)
    def forward(self,input):
        out1 = self.conv1(input)
        out2 = self.conv2(input)
        out3 = self.conv3(input)
        out4 = self.conv4(input)
        out5 = self.conv5(input)
        out6 = self.conv6(input)
        out = torch.cat((out1,out2,out3,out4,out5,out6),1)
        return out


# Define NN models ------------------------------------------------------------
class MyNet_CNN(nn.Module):
    def __init__(self,cnn_in,cnn_out,kernel_size,predict_out,k):
        super(MyNet_CNN,self).__init__()
        # for out1
        self.scale1_1 = scale_conv(cnn_in)
        # self.scale1_2 = scale_dens(k)     
        self.incept1_1 = inception(cnn_in)
        self.incept1_2 = inception(11)
        self.dens1 = nn.Linear(11*k,32)
        # for out2
        self.scale2_1 = scale_conv(cnn_in)
        # self.scale2_2 = scale_dens(k)    
        self.incept2_1 = inception(cnn_in)
        self.incept2_2 = inception(11)
        self.dens2 = nn.Linear(11*k,32)
        # for out3
        self.scale3_1 = scale_conv(cnn_in)
        # self.scale3_2 = scale_dens(k)    
        self.incept3_1 = inception(cnn_in)
        self.incept3_2 = inception(11)
        self.dens3 = nn.Linear(11*k,32)
        # for out4
        self.scale4_1 = scale_conv(cnn_in)
        # self.scale4_2 = scale_dens(k)    
        self.incept4_1 = inception(cnn_in)
        self.incept4_2 = inception(11)
        self.dens4 = nn.Linear(11*k,32)
        # for out6
        self.dens5 = nn.Linear(128,predict_out)
        
    def forward(self,input):
        # out1
        out1 = input[:,0:4,:]
        out1 = torch.cat((out1,torch.mul(torch.randn((out1.size(0),1,out1.size(2))),1/torch.max(torch.sqrt(torch.mean(out1**2,axis=2)),axis=1)[0].reshape(-1,1,1))),1)
        out1 = self.scale1_1(out1)
        # out1 = self.scale1_2(out1)
        out1 = self.incept1_1(out1)
        out1 = nn.LeakyReLU(negative_slope=0.05)(out1)
        # out1 = self.incept1_2(out1)
        # out1 = nn.LeakyReLU(negative_slope=0.05)(out1)
        out1 = torch.flatten(out1,start_dim=1, end_dim=2)
        out1 = self.dens1(out1)
        out1 = nn.LeakyReLU(negative_slope=0.05)(out1)
                
        # out2
        out2 = input[:,4:8,:]
        out2 = torch.cat((out2,torch.mul(torch.randn((out2.size(0),1,out2.size(2))),1/torch.max(torch.sqrt(torch.mean(out2**2,axis=2)),axis=1)[0].reshape(-1,1,1))),1)
        out2 = self.scale2_1(out2)
        # out2 = self.scale2_2(out2)
        out2 = self.incept2_1(out2)
        out2 = nn.LeakyReLU(negative_slope=0.05)(out2)
        # out2 = self.incept2_2(out2)
        # out2 = nn.LeakyReLU(negative_slope=0.05)(out2)
        out2 = torch.flatten(out2,start_dim=1, end_dim=2)
        out2 = self.dens2(out2)
        out2 = nn.LeakyReLU(negative_slope=0.05)(out2)
        
        # out3
        out3 = input[:,8:12,:]
        out3 = torch.cat((out3,torch.mul(torch.randn((out3.size(0),1,out3.size(2))),1/torch.max(torch.sqrt(torch.mean(out3**2,axis=2)),axis=1)[0].reshape(-1,1,1))),1)
        out3 = self.scale3_1(out3)
        # out3 = self.scale3_2(out3)
        out3 = self.incept3_1(out3)
        out3 = nn.LeakyReLU(negative_slope=0.05)(out3)
        # out3 = self.incept3_2(out3)
        # out3 = nn.LeakyReLU(negative_slope=0.05)(out3)
        out3 = torch.flatten(out3,start_dim=1, end_dim=2)
        out3 = self.dens3(out3)
        out3 = nn.LeakyReLU(negative_slope=0.05)(out3)
        
        # out4
        out4 = input[:,12:16,:]
        out4 = torch.cat((out4,torch.mul(torch.randn((out4.size(0),1,out4.size(2))),1/torch.max(torch.sqrt(torch.mean(out4**2,axis=2)),axis=1)[0].reshape(-1,1,1))),1)
        out4 = self.scale4_1(out4)
        # out4 = self.scale4_2(out4)
        out4 = self.incept4_1(out4)
        out4 = nn.LeakyReLU(negative_slope=0.05)(out4)
        # out4 = self.incept4_2(out4)
        # out4 = nn.LeakyReLU(negative_slope=0.05)(out4)
        out4 = torch.flatten(out4,start_dim=1, end_dim=2)
        out4 = self.dens4(out4)
        out4 = nn.LeakyReLU(negative_slope=0.05)(out4)
        
        # out6
        out6 = torch.cat((out1,out2,out3,out4),1)
        out6 = self.dens5(out6)
        return out6


# Test models -----------------------------------------------------------------
def Test_MyNet(net,test_x,test_y):
    net.eval()
    with torch.no_grad():
        test_output = net(test_x)
        test_err = (test_y-test_output)
    return test_err