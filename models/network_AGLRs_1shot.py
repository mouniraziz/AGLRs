import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import pdb
import math
import argparse
import sys
import random

sys.dont_write_bytecode = True


###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_FewShotNet(pretrained=False, model_root=None, which_model='Conv64', norm='batch', init_type='normal',
                      use_gpu=True, shot_num=1, **kwargs):
    FewShotNet = None
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model == 'Conv64F':
        FewShotNet = Conv_64F(norm_layer=norm_layer, **kwargs)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % which_model)
    init_weights(FewShotNet, init_type=init_type)

    if use_gpu:
        FewShotNet.cuda()

    if pretrained:
        FewShotNet.load_state_dict(model_root)

    return FewShotNet


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)




class Conv_64F(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3, batch_size=4, shot_num=1):
        super(Conv_64F, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  #

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = AGLRs_Metric(neighbor_k=neighbor_k, batch_size=batch_size, shot_num=shot_num)  # 1*num_classes

    def forward(self, input1, input2):


        # extract features of input1--query image
        q = self.features(input1).contiguous()
        q = q.view(q.size(0), q.size(1), -1)
        q = q.permute(0, 2, 1)

        # extract features of input2--support set
        S = self.features(input2).contiguous()
        S = S.view(S.size(0), S.size(1), -1)
        S = S.permute(0, 2, 1)

        Similarity_list = self.classifier(q, S)

        return Similarity_list


# ========================== Define MML ==========================#
class AGLRs_Metric(nn.Module):
    def __init__(self, num_classes=5, neighbor_k=3, batch_size=4, shot_num=1):
        super(AGLRs_Metric, self).__init__()
        self.neighbor_k = neighbor_k
        self.batch_size = batch_size
        self.shot_num = shot_num
        self.num_classes = num_classes
        self.W0 = nn.Sequential(
                nn.Linear( 25*25, 256),
                nn.LeakyReLU(0.2, True),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2, True),
                nn.Linear(256, 8),
                nn.LeakyReLU(0.2, True),
                nn.Linear(8, 1))
        self.m = nn.AdaptiveAvgPool2d((5,5))
        self.Norm_layer = nn.BatchNorm1d(self.num_classes * 3, affine=True)
        self.FC_layer = nn.Conv1d(1, 1, kernel_size=3, stride=1, dilation=5, bias=False)
          

    def calculate_sim(self, input1, input2):
        
        # L2 Normalization
        input1_norm = torch.norm(input1, 2, 2, True)
        input2_norm = torch.norm(input2, 2, 2, True)
        
        # Calculate the Image-to-Class Similarity
        query_norm = input1 / input1_norm
        support_norm = input2 / input2_norm
        assert (torch.min(input1_norm) > 0)
        assert (torch.min(input2_norm) > 0)

        support_norm = support_norm.contiguous().view(-1,
                                                        self.shot_num * support_norm.size(1),
                                                        support_norm.size(2))

        # local level and part level cosine similarity between a query set and a support set
        innerproduct_matrix = torch.matmul(query_norm.unsqueeze(1),
                                              support_norm.permute(0, 2, 1))

        
        topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 3)
        inner_sim = torch.sum(torch.sum(topk_value, 3), 2)
    
        return   inner_sim
    
    def euclidean_metric(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)**2).sum(2)
        return logits

    def calculate_sim_matrix(self, input1, input2):
        
        # L2 Normalization
        input1_norm = torch.norm(input1, 2, 2, True)
        input2_norm = torch.norm(input2, 2, 2, True)
        
        # Calculate the Image-to-Class Similarity
        query_norm = input1 / input1_norm
        support_norm = input2 / input2_norm
        assert (torch.min(input1_norm) > 0)
        assert (torch.min(input2_norm) > 0)

        support_norm = support_norm.contiguous().view(-1,
                                                        support_norm.size(1),
                                                        support_norm.size(2))

        innerproduct_matrix = torch.matmul(query_norm.unsqueeze(1),
                                              support_norm.permute(0, 2, 1))


        return   innerproduct_matrix

    def call_AGLRs_similarity(self, input1_batch, input2_batch):

        Similarity_list = []
        input1_batch = input1_batch.contiguous().view(self.batch_size, -1, input1_batch.size(1),
                                                      input1_batch.size(2))
        input2_batch = input2_batch.contiguous().view(self.batch_size, -1, input2_batch.size(1),
                                                      input2_batch.size(2))

        for i in range(self.batch_size):
            input1 = input1_batch[i]
            input2 = input2_batch[i]
            b1,hw,c = input1.size()
            b2,hw,c = input2.size()
            h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
            
            inner_sim1= self.calculate_sim(input1, input2)

            input1 = input1.permute(0,2,1).view(b1,c,h,w)
            input2 = input2.permute(0,2,1).view(b2,c,h,w)

            input1 = self.m(input1).contiguous().view(b1,-1)
            input2 = self.m(input2).contiguous().view(b2,-1)

            inner_sim2= self.euclidean_metric(input1, input2)
            inner_sim2 = 1/(1 + torch.exp(1e-04*inner_sim2))

            input1 = input1.view(b1,c,5*5).permute(0,2,1)
            input2 = input2.view(b2,c,5*5).permute(0,2,1)

            input1 = input1- input1.mean(-1).unsqueeze(-1)
            input2 = input2- input2.mean(-1).unsqueeze(-1)

            inner_sim3= self.W0(self.calculate_sim_matrix(input1, input2).view(75*self.num_classes,25*25)).view(75,self.num_classes)
            
            sim_soft = torch.cat(( inner_sim1, inner_sim2, inner_sim3), 1)
            sim_soft = self.Norm_layer(sim_soft).unsqueeze(1)
            sim_soft = self.FC_layer(sim_soft).squeeze(1)

            Similarity_list.append(sim_soft)
            

        return Similarity_list

    def forward(self, x1, x2):

        Similarity_list = self.call_AGLRs_similarity(x1, x2)

        return Similarity_list
