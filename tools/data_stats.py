from __future__ import absolute_import, division, print_function

import numpy as np
from tqdm import trange
import os
from pose_seq_input import PoseSeqInput


def print_max_plen(pose_seq_input):
    max_plen = 0
    for k in range(len(pose_seq_input.train_keys)):
        _, _, _, _, plen = pose_seq_input.read_h5_data(k, True)
        if max_plen < plen:
            max_plen = plen

    print(max_plen)


def save_arr(pose_seq_input, arr, name):
    file_path = os.path.join(pose_seq_input.data_path, pose_seq_input.data_set + pose_seq_input.data_set_version + '_' + name + '.npy')
    with open(file_path, 'wb') as f:
        np.save(f, arr)


def compute_moments(pose_seq_input):
    labs, poses = pose_seq_input.load_to_ram(True)

    mask = np.zeros(np.shape(poses), dtype=np.bool)
    print('Computing Masks...')
    t = trange(len(pose_seq_input.train_keys), dynamic_ncols=True)
    for k in t:
        mask[k, :, :, :labs[k, 3]] = True

    poses = np.transpose(poses, [1, 2, 3, 0])
    mask = np.transpose(mask, [1, 2, 3, 0])

    masked_pose = np.reshape(poses[mask], [pose_seq_input.pshape[0], pose_seq_input.pshape[1], -1])

    del mask
    del labs
    del poses
    print('Computing Moments...')
    skel_mean = np.reshape(np.mean(masked_pose, axis=(0, 2)), [1, 1, 1, 3])
    skel_std = np.reshape(np.std(masked_pose, axis=(0, 2)), [1, 1, 1, 3])

    del masked_pose
    print(skel_mean, skel_std)
    save_arr(pose_seq_input, skel_mean, 'skel_mean')
    save_arr(pose_seq_input, skel_std, 'skel_std')


def compute_dm_moments(pose_seq_input):
    # from matplotlib import pyplot as plt
    def td_dist(x_td, consider_conf=False):
        if consider_conf:
            conf_x = x_td[:, 2, :]
            x_td = x_td[:, :2, :]
        x_tile = np.tile(np.expand_dims(x_td, axis=1), [1, np.shape(x_td)[0], 1, 1])
        x_sub = x_tile - np.transpose(x_tile, [1, 0, 2, 3])
        x_dist = np.sum(np.square(x_sub), axis=2)
        if consider_conf:
            conf_mask = np.expand_dims(conf_x > 0, axis=1)
            x_dist = x_dist * conf_mask
            conf_mask = np.transpose(conf_mask, [1, 0, 2])
            x_dist = x_dist * conf_mask
        return x_dist

    def rot_dist(x_rot):
        x_tile = np.tile(np.expand_dims(x_rot, axis=1), [1, np.shape(x_rot)[0], 1, 1])
        x_sub = x_tile * np.transpose(x_tile, [1, 0, 2, 3])
        rot_dist = 1 - np.abs(np.sum(x_sub, axis=2))
        return rot_dist

    labs, poses = pose_seq_input.load_to_ram(True)

    mask = np.zeros(np.shape(poses), dtype=np.bool)
    print('Computing Masks...')
    t = trange(len(pose_seq_input.train_keys), dynamic_ncols=True)
    for k in t:
        mask[k, :, :, :labs[k, 3]] = True

    poses = np.transpose(poses, [1, 2, 3, 0])
    mask = np.transpose(mask, [1, 2, 3, 0])

    print('Computing DMs...')
    masked_pose = np.reshape(poses[mask], [pose_seq_input.pshape[0], pose_seq_input.pshape[1], -1])

    del mask
    del labs
    del poses

    all_dists = masked_pose[:, :3, :]
    print(np.shape(all_dists))
    if pose_seq_input.pshape[1] > 3:
        all_rots = masked_pose[:, 3:, :]
        print(np.shape(all_rots))

    del masked_pose

    if pose_seq_input.data_set == 'NTURGBD':
        all_dists_0 = td_dist(all_dists[:25, :, :])
        all_dists_1 = td_dist(all_dists[25:, :, :])
        del all_dists

        all_dists = np.concatenate(
            [all_dists_0, all_dists_1], axis=2)
        del all_dists_0
        del all_dists_1

        if pose_seq_input.pshape[1] > 3:
            all_rots_0 = rot_dist(all_rots[:25, :, :])
            all_rots_1 = rot_dist(all_rots[25:, :, :])
            del all_rots

            all_rots = np.concatenate(
                [all_rots_0, all_rots_1], axis=2)
            del all_rots_0
            del all_rots_1

    # elif pose_seq_input.data_set == 'SBU_inter':
    #     all_dists_0 = td_dist(all_dists[:15,:,:], True)
    #     all_dists_1 = td_dist(all_dists[15:,:,:], True)
    #     del all_dists
    #
    #     all_dists = np.concatenate(
    #         [all_dists_0, all_dists_1], axis=2)
    #     del all_dists_0
    #     del all_dists_1
    else:
        all_dists = td_dist(all_dists)

    print(np.shape(all_dists))
    mean_ch_dist = np.mean(all_dists)
    mean_px_dist = np.mean(all_dists, axis=2)
    std_ch_dist = np.std(all_dists)
    std_px_dist = np.std(all_dists, axis=2)

    print(np.shape(mean_px_dist))
    # plt.imshow(mean_px_dist, interpolation='nearest')
    # plt.show()

    if pose_seq_input.pshape[1] > 3:
        print(np.shape(all_rots))
        mean_ch_rot = np.mean(all_rots)
        mean_px_rot = np.mean(all_rots, axis=2)
        std_ch_rot = np.std(all_rots)
        std_px_rot = np.std(all_rots, axis=2)

        print(np.shape(mean_px_rot))
        # plt.imshow(mean_px_rot, interpolation='nearest')
        # plt.show()

    print('Saving results...')
    save_arr(pose_seq_input, mean_ch_dist, 'mean_ch_dist')
    save_arr(pose_seq_input, mean_px_dist, 'mean_px_dist')
    save_arr(pose_seq_input, std_ch_dist, 'std_ch_dist')
    save_arr(pose_seq_input, std_px_dist, 'std_px_dist')

    if pose_seq_input.pshape[1] > 3:
        save_arr(pose_seq_input, mean_ch_rot, 'mean_ch_rot')
        save_arr(pose_seq_input, mean_px_rot, 'mean_px_rot')
        save_arr(pose_seq_input, std_ch_rot, 'std_ch_rot')
        save_arr(pose_seq_input, std_px_rot, 'std_px_rot')


def plot_dms(pose_seq_input):
    def td_dist(x_td):
        x_tile = np.tile(np.expand_dims(x_td, axis=1), [1, np.shape(x_td)[0], 1, 1])
        x_sub = x_tile - np.transpose(x_tile, [1, 0, 2, 3])
        x_dist = np.sum(np.square(x_sub), axis=2)
        return x_dist

    import random
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    labs, poses = pose_seq_input.load_to_ram(False)
    rand_seq = random.randint(0, pose_seq_input.len_val_keys)
    proc_pose, _ = pose_seq_input.process_pose(poses[rand_seq, :, :, :labs[rand_seq, 3]])
    pose_dms = td_dist(proc_pose[:25, :, :])
    for n in range(pose_seq_input.pick_num):
        plt.imshow(pose_dms[:, :, n], interpolation='nearest')
        plt.savefig('demo_dm_matrix_%02d.png' % n)


if __name__ == "__main__":
    class Config(object):
        data_path = './data/'
        data_set = 'NTURGBD'
        data_set_version = ''
        batch_size = 10
        random_crop = False
        crop_len = 100
        random_pick = True
        pick_num = 20
        data_source = 'hdf5'
        curriculum_l = False
        only_val = False
        only_3dpos = True
        data_set_version = ''

    config = Config()
    for i in range(2):
        config.data_set_version = 'v%d' % (i + 1)
        pose_seq_input = PoseSeqInput(config)
        compute_moments(pose_seq_input)
        compute_dm_moments(pose_seq_input)
        plot_dms(pose_seq_input)
