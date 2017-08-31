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
    prog = re.compile('skels/(s\d+s\d+)/(\d+)/\d+/skeleton_pos.txt')
    for found_file in iter(input.get, 'STOP'):
        confpars = prog.findall(found_file)[0]

        sub_pair = confpars[0]
        action = int(confpars[1])

        with open(found_file) as csvfile:
            framecount = sum(1 for line in csvfile)
            posearray = np.empty([framecount, 90])
            csvfile.seek(0)
            skelreader = csv.reader(csvfile)
            for r, row in enumerate(skelreader):
                posearray[r,:] = np.array(row[1:])

            posearray = np.reshape(posearray, [framecount, 30, 3])
            posearray = np.transpose(posearray, [1,2,0])

            output.put((sub_pair, action, posearray, framecount))

if __name__ == '__main__':

    found_files = [file for file in glob('skels/*/*/*/skeleton_pos.txt')]
    # found_files += [file for file in glob('noisy_skels/*/*/*/skeleton_pos.txt')]
    print('Processing %d files...' % (len(found_files)))

    dataset = 'SBU_inter'

    h5files = [h5.File(dataset+"v%d.h5"%(f+1), "w") for f in range(5)]

    val_sets = [ {'s01s02', 's03s04', 's05s02', 's06s04'},
                 {'s02s03', 's02s07', 's03s05', 's05s03'},
                 {'s01s03', 's01s07', 's07s01', 's07s03'},
                 {'s02s01', 's02s06', 's03s02', 's03s06'},
                 {'s04s02', 's04s03', 's04s06', 's06s02', 's06s03'}]

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

    prog = re.compile('s(\d+)s(\d+)')
    # Get and print results
    print('Processed Files:')
    t = trange(len(found_files), dynamic_ncols=True)
    for seqnum in t:
        sub_pair, action, posearray, framecount = done_queue.get()

        if posearray is not None:
            subjects_seq = prog.findall(sub_pair)[0]
            for subject in subjects_seq:
                subjects.add(subject)
            actions.add(action)
            maxframecount = max(framecount, maxframecount)

            subarray = np.array(int(subjects_seq[0]))
            actarray = np.array(action)

            for f, h5file in enumerate(h5files):
                datasplit = 'Validate' if sub_pair in val_sets[f] else 'Train'
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
