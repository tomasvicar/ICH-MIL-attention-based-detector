class Config:
    
    tmp_save_dir='../models_python'
    
    # train_num_workers=6
    # test_num_workers=3
    
    train_num_workers=0
    test_num_workers=0
    
    
    data_path=r'Z:\CELL_MUNI\verse2020\training_data_resaved'
    
    
    model_name='model'
    
    train_batch_size = 8
    test_batch_size = 4
    
    max_epochs = 12
    step_size=5
    gamma=0.1
    init_lr=0.001
    
    crop_size_train = [128,128,25]
    crop_size_valid = [128,128,25]
    
    
    