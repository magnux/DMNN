import os
import numpy as np
import h5py as h5
import re
from glob import glob
from tqdm import trange
import shutil
import time

from multiprocessing import Process, Queue, current_process, freeze_support

def worker(input, output):
    prog = re.compile('ActionsSkeleton/a(\d+)_s(\d+)_e(\d+)_v(\d+).mat')
    for found_file in iter(input.get, 'STOP'):
        confpars = prog.findall(found_file)[0]

        action = int(confpars[0])
        subject = int(confpars[1])
        view = int(confpars[3])

        matfile = h5.File(found_file,'r')
        posearray = np.array(matfile['/A'])
        matfile.close()
        framecount = np.size(posearray,0)

        posearray = np.reshape(posearray, [framecount, 15, 3])
        posearray = np.transpose(posearray, [1,2,0])

        output.put((subject, action, view, posearray, framecount))

if __name__ == '__main__':

    found_files = [file for file in glob('ActionsSkeleton/*.mat')]
    print('Processing %d files...' % (len(found_files)))

    dataset = 'UWA3DII'

    h5files = [h5.File(dataset+"v%d.h5"%(f+1), "w") for f in range(12)]

    train_sets = [{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}]
    val_sets = [3,4,2,4,2,3,1,4,1,3,1,2]

    subjects = set()
    actions = set()
    maxframecount = 0

    num_procs = 4

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for found_file in found_files:
        task_queue.put(found_file)

    # Start worker processes
    print('Spawning processes...')
    for _ in range(num_procs):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print('Processed Files:')
    t = trange(len(found_files), dynamic_ncols=True)
    for seqnum in t:
        subject, action, view, posearray, framecount = done_queue.get()

        if posearray is not None:
            subjects.add(subject)
            actions.add(action)
            maxframecount = max(framecount, maxframecount)

            subarray = np.array(subject)
            actarray = np.array(action)

            for f, h5file in enumerate(h5files):
                if view in train_sets[f//2] or view == val_sets[f]:
                    datasplit = 'Validate' if view == val_sets[f] else 'Train'
                    datapath = '{}/{}/SEQ{}/'.format(dataset,datasplit,seqnum)
                    h5file.create_dataset(
                        datapath+'Subject', np.shape(subarray),
                        dtype='int32', data=subarray
                    )
                    h5file.create_dataset(
                        datapath+'Action', np.shape(actarray),
                        dtype='int32', data=actarray
                    )
                    h5file.create_dataset(
                        datapath+'Pose', np.shape(posearray),
                        dtype='float32', data=posearray
                    )


    # Tell child processes to stop
    print('Stopping processes...')
    for _ in range(num_procs):
        task_queue.put('STOP')

    for h5file in h5files:
        h5file.flush()
        h5file.close()

    print("")
    print("done.")
    print("Subjects: ", subjects)
    print("Actions: ", actions)
    print("Max Frame Count:", maxframecount)
