import numpy as np
from resnet_2D_heatmap import Resnet_2D_heatmap

class Config:
    
    tmp_save_dir='../models_python'
    

    # train_num_workers=2
    # test_num_workers=2
    
    train_num_workers=6
    test_num_workers=2
    
    

    data_path='..'
    data_table_path = '../bBoxAnnotations_All_patients2.csv'
    
    
    model_name='model'
    
    train_batch_size = 16
    test_batch_size = 16
    
    
    # lr_steps = np.cumsum([50,20,10])
    lr_steps = np.cumsum([30,10,5])
    gamma = 0.1
    init_lr = 0.001
    max_epochs = lr_steps[-1]
    
    # net = Small_resnet3D_noheatmap
    net = Resnet_2D_heatmap

    
    SPLIT_RATIO = [7,3]
    
    ###### Augmentation parameters
    max_multiplier  = 0.05   # multiply augmentation 
    max_add  = 10/1000          # Add augmentation
    max_crop_perc = 0.1     # Maximal random crop percentage
    max_resize_perc = 0.1   # Maximal resize percentage
    max_rot_angle = 0      # Maximal angle for axial rotation



        
    
    