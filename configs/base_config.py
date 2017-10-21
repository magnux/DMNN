{
    ## Data Options
    'data_path': './data/',
    # Datasets: HumanEvaI, Human36, NTURGBD
    'data_set': 'NTURGBD',
    'data_set_version': 'v1',
    # Backend of the data source: hdf5, tfrecord, ram
    'data_source': 'tfrecord',

    ## Model Options
    # Model for the inference: standard, bidirectional, siamese, staged, conv3d_resnet, conv3d_resnext, conv2d_resnet, conv2d_resnext
    'inference_model': 'standard',
    # Model for the rnn cell: lstm, bnlstm, phlstm, convlstm, convphlstm, conv_variant, None (for resnet based models)
    'cell_model': 'convlstm',
    # Apply batch normalization to the convolutions
    'batch_norm': True,
    # Keeping Probabilities (1 - Dropout Rate)
    'keep_prob': 1,
    # Number of recurrent layers
    'num_layers': 5,
    # Number of units in each recurrent layer
    'hidden_size': 64,
    # Specs for the conv variant
    'conv_specs': None,
    # Model for the classifier: mean_pool, class_per_frame, last_n_outs
    'loss_model': 'mean_pool',
    # Subject loss to minimize
    'sub_loss': False,
    # Randomly crop the sequences, or use the whole
    'random_crop': False,
    # Length of the random crops
    'crop_len': 64,
    # Crop the sequences equally spaced and pick a random frame of each subsequence
    'random_pick': True,
    # Number of the random picks
    'pick_num': 20,
    # Skeleton sequence smoothing
    'seq_smooth': False,
    # Identity Preserving LSTM
    'pres_ident': False,
    # Use new pooling style
    'new_pool': False,
    # Extract the triu from convs and reshape them, to avoid redundancy
    'extract_triu': False,
    # Bypass matrix embedding, useful to test LSTMs
    'no_dm': False,
    # ResNet or ResNeXt configuration
    'resnet_blocks': None,
    # ResNet variants multi block concat output
    'resnet_multiout': False,

    ## Training Options
    # It's the batch size
    'batch_size': 100,
    # Final epoch
    'max_max_epoch': 200,
    # How fast should we learn?
    'learning_rate': 1.0e-3,
    # Train using learning rate decay
    'lr_decay': True,
    # Learning rate decay steps
    'decay_steps': 1.0e4,
    # Learning rate decay steps
    'decay_rate': 0.75,
    # Use gradient clipping
    'grad_clipping': False,
    # Max gradient norm for clipping
    'max_grad_norm': 1,
    # Train using curriculum learning
    'curriculum_l': False,
    # Custom learning rate schedule
    'custom_lr': False,
    # List for the custom schedule
    'custom_lr_list': None,

    ## Data Augmentation (or Modification)
    # Use only 3d coords, discarding joint rotation if available
    'only_3dpos': False,
    # Use a single skeleton if two are available
    'single_input': False,
    # Normalize skeleton
    'norm_skel': False,
    # Jitter skeleton height
    'jitter_height': False,
    # Duplicate skeleton in classes with only one
    'dup_input': False,
    # Swap skeletons randomly
    'swap_input': False,
    # Normalize distance matrixes by channel
    'norm_dms_ch': False,
    # Normalize distance matrixes by pixel
    'norm_dms_px': False,
    # Normalize distance matrixes by BatchNorm
    'norm_dms_bn': False,
    # Split body by different regions
    'split_bod': False,
    # Sim occlussions in body regions
    'sim_occlusions': False,
    # Sim translations between skeletons
    'sim_translations': False,
    # Transform 2D DMs to 3D DMs
    'dm_transform': False,
    # Transform skeleton format to NTURGBD
    'skel_transform': False,
    # Learn a linear combination over the joints, normalized output
    'learn_comb': False,
    # Learn a linear combination over the joints, softmax output
    'learn_comb_sm': False,
    # Learn a orthogonal linear combination over the joints
    'learn_comb_orth': False,
    # Learn a orthogonal linear combination over the joints, with RMSProp gradient
    'learn_comb_orth_rmsprop': False,
    # Learn a orthogonal linear combination over the joints, no constraints
    'learn_comb_unc': False,
    # Learn a linear combination over the joints
    'learn_comb_centered' : False,
    # Permutation over the joints
    'joint_permutation': None,

    ## Environment Options
    # Is it Pascal time? ... NOT SAFE YET
    'use_type16': False,
    # Random inintializer scale
    'init_scale': 0.1,
    # Load pretrained model
    'restore_pretrained': False,
    # Pretrained model path
    'pretrained_path': None
}
