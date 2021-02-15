import numpy as np
from small_resnet3D_noheatmap import Small_resnet3D_noheatmap
from small_resnet3D import Small_resnet3D

class Config:
    
    tmp_save_dir='../models_python'
    
    # train_num_workers=6
    # test_num_workers=3
    
    # train_num_workers=2
    # test_num_workers=2
    
    train_num_workers=0
    test_num_workers=0
    
    
    # data_path='../raw2'
    # data_table_path = '../VFN_Annotations.csv'
    # data_path='../raw_cq'
    data_path='../raw_cq_subsampled4x'
    data_table_path = '../CQ_Annotations1xor4x.csv'
    # data_table_path = '../CQ_Annotations.csv'
    
    model_name='model'
    
    train_batch_size = 1
    test_batch_size = 1
    
    
    # lr_steps = np.cumsum([50,20,10])
    lr_steps = np.cumsum([300,50,50])
    gamma = 0.1
    init_lr = 0.001
    max_epochs = lr_steps[-1]
    
    # net = Small_resnet3D_noheatmap
    net = Small_resnet3D

    
    SPLIT_RATIO = [7,3]
    
    ###### Augmentation parameters
    max_multiplier  = 0.05   # multiply augmentation 
    max_add  = 10/1000          # Add augmentation
    max_crop_perc = 0.1     # Maximal random crop percentage
    max_resize_perc = 0.1   # Maximal resize percentage
    max_rot_angle = 0      # Maximal angle for axial rotation



        
    
    