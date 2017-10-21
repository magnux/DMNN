{
    # Datasets: HumanEvaI, Human36, NTURGBD
    'data_set' : 'MSRC12',
    'data_set_version' : '',
    # Backend of the data source: hdf5, tfrecord, ram
    'data_source' : 'ram',
    # Model for the inference rnn: standard, bidirectional, siamese, staged
    'inference_model' : 'conv3d_resnext',
    # Model for the rnn cell: lstm, bnlstm, phlstm, convlstm, convphlstm, conv_variant
    'cell_model' : None,
    # Number of recurrent layers
    'num_layers' : 3,

    # Use only 3d coords, discarding joint rotation if available
    'only_3dpos' : True,
    # Crop the sequences equally spaced and pick a random frame of each subsequence
    'random_pick' : True,
    # Number of the random picks
    'pick_num' : 20,

    # It's the batch size
    'batch_size' : 32,
    # Final epoch
    'max_max_epoch' : 80,
    # How fast should we learn?
    'learning_rate' : 1.0e-4,
    # Train using learning rate decay
    'lr_decay' : False,
    # Learning rate decay steps
    'decay_steps' : 65536,
    # Learning rate decay steps
    'decay_rate' : 0.1,
    # Custom learning rate schedule
    'custom_lr' : True,
    # List for the custom schedule
    'custom_lr_list' : [(50,1.0e-3), (4000,1.0e-4), (6000,1.0e-5)],

    # learn a linear combination over the joints
    'learn_comb' : True,
    # Jitter skeleton height
    'jitter_height': False,
    # Swap skeletons randomly
    'swap_input': True,
    # Sim occlussions in body regions
    'sim_occlusions': False,


    # Normalize distance matrixes by channel
    'norm_dms_ch': True,
    
    'resnet_blocks': [
        {'size': 32, 'stride': 1},
        {'size': 128, 'bottleneck': 32, 'cardinality': 16, 'stride': 1},
        {'size': 256, 'bottleneck': 64, 'cardinality': 16, 'stride': 2},
        {'size': 512, 'bottleneck': 128, 'cardinality': 16, 'stride': 2}],
    
    # Load pretrained model
    # 'restore_pretrained': True,
    # Pretrained model path
    # 'pretrained_path': '/home/lporzi/src/PoseSeqRNN/save_ntu_resnext_learn_comb_unc.bak/model.ckpt-100560'
}
