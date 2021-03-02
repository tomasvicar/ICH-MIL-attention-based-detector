import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import matplotlib.pyplot as plt



class myAttention(nn.Module):
    
    def __init__(self,in_size, out_size):
        super().__init__()
        
        self.conv_tanh = nn.Conv2d(in_size, in_size, 1)
        self.conv_sigm = nn.Conv2d(in_size, in_size, 1)
        self.conv_w = nn.Conv2d(in_size, 1, 1)
    
    def forward(self, inputs):
        
        tanh = torch.tanh(self.conv_tanh(inputs))
        
        sigm = torch.sigmoid(self.conv_sigm(inputs))
        
        z = self.conv_w(tanh * sigm) 
        
        shape1 = list(z.size())
        z = z.view(shape1[0],shape1[1],-1)
        
        shape2 = list(inputs.size())
        inputs = inputs.view(shape2[0],shape2[1],-1)
        
        a = torch.softmax(z,dim=2)
        
        output = torch.sum(a * inputs,2)
        
        a = torch.reshape(a,shape1)
        
        return output, a
        
        
    
    

class myConv(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=None,do_batch=1,dov=0):
        super().__init__()
        
        pad=int((filter_size-1)/2)
        
        self.do_batch=do_batch
        self.dov=dov
        self.conv=nn.Conv2d(in_size, out_size,filter_size,stride,pad)
        
        self.bn=nn.BatchNorm2d(out_size,momentum=0.1)
        # self.bn = nn.GroupNorm(num_groups=int(out_size/4), num_channels=out_size )
        
        
        if self.dov>0:
            self.do=nn.Dropout(dov)
    
    def forward(self, inputs):
     
        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)  
        
        outputs=F.relu(outputs)
        
        if self.dov>0:
            outputs = self.do(outputs)
        
        return outputs




class Resnet_2D_heatmap(nn.Module):
    
    
    def __init__(self, input_size,output_size,levels=3,lvl1_size=12, layers_in_lvl = 2):
        super().__init__()
        self.lvl1_size=lvl1_size
        self.levels=levels
        self.output_size=output_size
        self.input_size=input_size
        self.layers_in_lvl = layers_in_lvl
        
        
        self.init_conv=myConv(input_size,lvl1_size)
        

        self.layers=nn.ModuleList()
        for lvl_num in range(levels):
            
            if lvl_num!=0:
                self.layers.append(myConv( int(lvl1_size*(lvl_num)), int(lvl1_size*(lvl_num+1))))
             
            for layer_num_in_lvl in range(layers_in_lvl):
                self.layers.append(myConv( int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1))))
                
                self.layers.append(myConv( int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1))))
                
                self.layers.append(myConv( int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1))))
            
            
        self.attention = myAttention(int(lvl1_size * (self.levels)),int(lvl1_size * (self.levels)))
        
        # self.conv_final = nn.Conv2d(int(lvl1_size * (self.levels)),output_size,3 ,1, 1)
        
        self.fc=nn.Linear(int(self.lvl1_size*self.levels), output_size)
        
        
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
        
        
    def forward(self, x):   
        
        
        y=self.init_conv(x)
        
        layer_num=-1
        for lvl_num in range(self.levels):
            
            
            if lvl_num!=0:
                layer_num=layer_num+1
                y=self.layers[layer_num](x)
            
            
            for layer_num_in_lvl in range(self.layers_in_lvl):
                
                layer_num=layer_num+1
                x=self.layers[layer_num](y)
                layer_num=layer_num+1
                x=self.layers[layer_num](x)
                layer_num=layer_num+1
                x=self.layers[layer_num](x)
            
                x=x+y
            
            x=F.max_pool2d(x, kernel_size = 2,  stride = 2)
            
        
        # x = self.conv_final(x)
        x,heatmap  = self.attention(x)
        # heatmap = x
        
        # x  = F.adaptive_max_pool2d(x,1)
        # x  = self.attention(x)
        
        shape=list(x.size())
        x=x.view(shape[0],-1)
        x=self.fc(x)
        
        return x,heatmap
