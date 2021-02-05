import torch
import numpy as np
import matplotlib.pyplot as plt


class Log():
    def __init__(self,names=['loss','acc']):
        
        self.names=names
        
        self.model_names=[]
            
        
        self.train_logs=dict(zip(names, [[]]*len(names)))
        self.valid_logs=dict(zip(names, [[]]*len(names)))
        
        self.train_log_tmp=dict(zip(names, [[]]*len(names)))
        self.valid_log_tmp=dict(zip(names, [[]]*len(names)))


        
    def append_train(self,list_to_save):
        
        for value,name in zip(list_to_save,self.names):
            self.train_log_tmp[name] = self.train_log_tmp[name] + [value]
        
        
    def append_valid(self,list_to_save):
        for value,name in zip(list_to_save,self.names):
            self.valid_log_tmp[name] = self.valid_log_tmp[name] + [value]
        
        
    def save_and_reset(self):
        
        
        for name in self.names:
            self.train_logs[name] =  self.train_logs[name] + [np.mean(self.train_log_tmp[name])]
            self.valid_logs[name] =  self.valid_logs[name] + [np.mean(self.valid_log_tmp[name])]
        
        
        self.train_log_tmp=dict(zip(self.names, [[]]*len(self.names)))
        self.valid_log_tmp=dict(zip(self.names, [[]]*len(self.names)))
        
        
        
    def plot(self,save_name=None):
        
        for name in self.names:
            plt.plot( self.train_logs[name], label = 'train')
            plt.plot(self.valid_logs[name], label = 'valid')
            plt.title(name)
            if save_name:
                plt.savefig(save_name)
            plt.show()
            
            
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']