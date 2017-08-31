import os
import numpy as np
import h5py as h5
import re
import csv
from glob import glob
from tqdm import trange
import shutil
import time

from multiprocessing import Process, Queue, current_process, freeze_support

def worker(input, output):
    prog = re.compile('multiview_action/view_(\d*)/a(\d*)_s(\d*)_e(\d*)')
    for found_dir in iter(input.get, 'STOP'):
        confpars = prog.findall(found_dir)[0]
        view = int(confpars[0])
        action = int(confpars[1])
        subject = int(confpars[2])
        environment = int(confpars[3])

        found_files = [file for file in glob(found_dir+'/*skeletons.txt')]



        framecount = len(found_files)
        posearray = np.zeros([20,3,framecount])
        for f, found_file in enumerate(found_files):
            skel_l = []
            with open(found_file) as csvfile:
                skelreader = csv.reader(csvfile)
                for r, row in enumerate(skelreader):
                    if (r % 21) == 0:
                        skel = np.empty([20,3])
                        skel_id = int(row[0])

                    skel[(r % 21)-1,:] = np.array(row[:3])

                    if (r % 21) == 20:
                        skel_l.append((skel,np.mean(skel[:,2])))

            # ordering principal subject
            if len(skel_l) > 1:
                skel_l.sort(key=lambda tup: tup[1], reverse=False)
            if len(skel_l) > 0:
                posearray[:,:,f] = skel_l[0][0]

        output.put((subject, action, view, posearray, framecount))

if __name__ == '__main__':

    found_dirs = [file for file in glob('multiview_action/*/*')]
    print('Processing %d files...' % (len(found_dirs)))

    dataset = 'NUCLA'
    train_views = {1, 2}
    h5file = h5.File(dataset+".h5", "w")

    subjects = set()
    actions = set()
    maxframecount = 0

    num_procs = 8

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for found_dir in found_dirs:
        task_queue.put(found_dir)

    # Start worker processes
    print('Spawning processes...')
    for _ in range(num_procs):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print('Processed Files:')
    t = trange(len(found_dirs), dynamic_ncols=True)
    for seqnum in t:
        subject, action, view, posearray, framecount = done_queue.get()

        if framecount > 0:
            subjects.add(subject)
            actions.add(action)
            maxframecount = max(framecount, maxframecount)

            subarray = np.array(subject)
            actarray = np.array(action)

            # v1 split (cross view protocol)
            datasplit = 'Train' if view in train_views else 'Validate'

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

    h5file.flush()
    h5file.close()

    print("")
    print("done.")
    print("Subjects: ", subjects)
    print("Actions: ", actions)
    print("Max Frame Count:", maxframecount)
