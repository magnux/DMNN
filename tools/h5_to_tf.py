from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tqdm import trange
import os
from pose_seq_input import PoseSeqInput


def export_to_tf(self):
    def make_example(key_idx, subject, action, pose, plen):
        ex = tf.train.Example()
        ex.features.feature["key_idx"].int64_list.value.append(int(key_idx))
        ex.features.feature["subject"].int64_list.value.append(int(subject))
        ex.features.feature["action"].int64_list.value.append(int(action))
        ex.features.feature["plen"].int64_list.value.append(int(plen))
        for sublist in pose.tolist():
            for subsublist in sublist:
                for value in subsublist:
                    ex.features.feature["pose"].float_list.value.append(value)
        return ex

    def write_split(is_training, keys):
        writer = None
        shard = 0
        splitname = 'train' if is_training else 'val'
        print('Transforming "%s" split...' % splitname)
        t = trange(len(keys), dynamic_ncols=True)
        for k in t:
            if writer == None:
                writer = tf.python_io.TFRecordWriter(
                    os.path.join(self.data_path, self.data_set + '_' + splitname + '_shard' + str(shard) + '.tf')
                )
            key_idx, subject, action, pose, plen = self.read_h5_data(k, is_training)
            ex = make_example(key_idx, subject, action, pose, plen)
            writer.write(ex.SerializeToString())
            if ((k + 1) % 4096) == 0:
                writer.close()
                writer = None
                shard += 1
        if writer != None:
            writer.close()

    write_split(True, self.train_keys)
    write_split(False, self.val_keys)


if __name__ == "__main__":
    class Config(object):
        data_path = './data/'
        data_set = 'NTURGBD'
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
    # for i in range(5):
    #     config.data_set_version = 'v%d'%(i+1)
    pose_seq_input = PoseSeqInput(config)

    print("do stuff")
