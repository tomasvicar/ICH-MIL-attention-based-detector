import numpy as np
from resnet_2D_heatmap import Resnet_2D_heatmap

class Config:
    
    tmp_save_dir='../models_python'
    

    # train_num_workers=2
    # test_num_workers=2
    
    train_num_workers = 12
    test_num_workers = 12
    
    # train_num_workers = 0
    # test_num_workers = 0

    # data_path='../RSNA_sub_sample'
    # data_table_path = '../label_table_dicomcol_merge_sample.csv'
    
    data_path='../../../obrazari_shared/RSNA_sub'
    data_table_path = '../../../obrazari_shared/label_table_dicomcol_merge.csv'
    
    
    model_name='model'
    
    train_batch_size = 128
    test_batch_size = 128
    
    # train_batch_size = 64
    # test_batch_size = 64
    
    
    # lr_steps = np.cumsum([50,20,10])
    lr_steps = np.cumsum([20,5,5])
    gamma = 0.1
    init_lr = 0.001
    max_epochs = lr_steps[-1]
    
    # net = Small_resnet3D_noheatmap
    net = Resnet_2D_heatmap

    
    SPLIT_RATIO = [97,3]
    plots_in_epoch = 5
    
    # SPLIT_RATIO = [8,2]
    # plots_in_epoch = 4
    
    
    ###### Augmentation parameters
    max_multiplier  = 0.02   # multiply augmentation 
    max_add  = 20          # Add augmentation
    scale_range = 0.1
    shears_range = 0.05
    tilt_range = 0
    translation_range = 50
    rotation_range = 30
    blur_sharp_range = [-0.5, 0.5]
    
    individual_p = 0.90
    global_p = 0.95
    



        
    
    
