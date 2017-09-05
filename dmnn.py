from __future__ import absolute_import, division, print_function

import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python import debug as tf_debug

from dmnn_input import DmnnInput
from dmnn_model import DmnnModel
from config import get_config

from tqdm import trange
import colorama as col
col.init(autoreset=True)

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("verbose", False, "To talk or not to talk")
flags.DEFINE_string("save_path", None, "Model output directory")
flags.DEFINE_string("config_file", None, "Model config file")
flags.DEFINE_bool("clean_start", False, "Should we start over again? (clears save_path and __pycache__)")
flags.DEFINE_bool("validation_epoch", False, "Should we perform a validation epoch?, it's unnecesary if using tensorboard")
flags.DEFINE_bool("run_metadata", False, "Save profiling metadata during training")
flags.DEFINE_bool("only_val", False, "Only perform a validation epoch")
flags.DEFINE_bool("debug", False, "Use debugger to track down bad values during training")
flags.DEFINE_string("ui_type", "curses", "Command-line user interface type (curses | readline)")

FLAGS = flags.FLAGS


def run_epoch(session, model, input_, is_training, global_step, summary_writer, config, epoch_rem=0, COUNT=0):
    """Runs the model on the given data."""
    loss_acum = 0.0
    acc_acum = 0.0
    if FLAGS.only_val:
        matrices = np.zeros([config.num_actions, config.num_actions], dtype=np.int32)
        n_mistakes = np.zeros([1, config.num_actions], dtype=np.int32)
        mistake_labs = np.empty([0, 4], dtype=np.int32)

    fetches = {
        "loss": model.loss,
        # "final_state": model.final_state,
        "accuracy": model.accuracy
    }
    if is_training:
        fetches.update({
            "train_op": model.train_op,
            "global_step": global_step,
            "learning_rate": model.lr
        })
    if FLAGS.only_val:
        fetches.update({
            "confusion_matrix": model.confusion_matrix,
            "mistakes_per_class": model.mistakes_per_class,
            "mistake_labs": model.mistake_labs,
            "logits": model.logits,
            "labels": model.labels,
            "idxs": model.idxs
        })
        of = open('/tmp/%02d.txt' % COUNT, 'w')

    epoch_size = epoch_rem if epoch_rem > 0 else (input_.train_epoch_size if is_training else input_.val_epoch_size)
    t = trange(epoch_size, disable=not FLAGS.verbose, dynamic_ncols=True)
    for batch in t:
        options = None
        run_metadata = None
        if is_training and (batch % epoch_size == epoch_size - 1) and FLAGS.run_metadata:  # Record execution stats
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        vals = session.run(fetches=fetches, options=options, run_metadata=run_metadata)
        loss = vals["loss"]
        # state = vals["final_state"]
        accuracy = vals["accuracy"]

        if is_training:
            learning_rate = vals["learning_rate"]
            global_step_val = vals["global_step"]

        loss_acum += loss
        acc_acum += accuracy
        if FLAGS.only_val:
            matrices += np.squeeze(vals["confusion_matrix"], axis=(0,3))
            n_mistakes += np.squeeze(vals["mistakes_per_class"], axis=(0,3))
            mistake_labs = np.concatenate((mistake_labs,vals["mistake_labs"]), axis=0)
            for logits, label, idx in zip(vals["logits"], vals["labels"], vals["idxs"]):
                out = ", ".join(str(l) for l in logits)
                out += ", " + str(label)
                out += ", " + str(idx)
                print(out, file=of)

        postfix = {
            'loss': "%.3f" % (loss_acum / (batch + 1)),
            'acc': "%.3f" % (acc_acum / (batch + 1))
        }
        if is_training:
            postfix.update({'l_rate': "%.1E" % learning_rate})
            if run_metadata is not None:
                summary_writer.add_run_metadata(run_metadata, 'S:%d' % global_step_val)

        t.set_postfix(postfix)

    if FLAGS.only_val:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        plt.imshow(matrices/epoch_size, interpolation='nearest')
        plt.savefig(FLAGS.save_path+'/confusion_matrix.png')

        plt.imshow(n_mistakes/epoch_size, interpolation='nearest')
        plt.savefig(FLAGS.save_path+'/mistakes_per_class.png')

        np.savetxt(FLAGS.save_path+'/mistake_labs.txt', mistake_labs, fmt='%i,%i,%i,%i')
        print(np.histogram(mistake_labs[:,1], bins=np.arange(config.num_subjects)))
        print(np.histogram(mistake_labs[:,2], bins=np.arange(config.num_actions)))
        print(np.mean(mistake_labs[:,3]))
        of.close()

    return loss_acum / epoch_size, acc_acum / epoch_size


def backup_pickle(filename, to_backup):
    if not tf.gfile.Exists(FLAGS.save_path):
        tf.gfile.MkDir(FLAGS.save_path)
        filename = os.path.join(FLAGS.save_path, filename + ".pickle")
        if not tf.gfile.Exists(filename):
            with open(filename, 'wb') as f:
                pickle.dump(to_backup, f)


def main(_):
    config = get_config(FLAGS)
    backup_pickle('config', config)
    config.only_val=FLAGS.only_val
    if FLAGS.only_val:
        config.data_source = 'ram'
        config.max_max_epoch = 0 # stop after first eval...
    session_config = None
    # session_config = tf.ConfigProto()
    # session_config.gpu_options.allow_growth=True
    # session_config.gpu_options.per_process_gpu_memory_fraction=1.0

    with tf.Graph().as_default():
        pose_seq_input = DmnnInput(config=config)

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None):
                mtrain = DmnnModel(
                    is_training=True,
                    config=config,
                    input_=pose_seq_input
                )
                if not FLAGS.only_val:
                    tf.summary.scalar("Training Loss", mtrain.loss)
                    tf.summary.scalar("Training Accuracy", mtrain.accuracy)
                    tf.summary.image('Training Mistakes per Class', mtrain.mistakes_per_class)
                    tf.summary.image('Training Confusion Matrix', mtrain.confusion_matrix)
                    if config.learn_comb or config.learn_comb_sm or config.learn_comb_orth or\
                            config.learn_comb_orth_rmsprop or config.learn_comb_unc:
                        tf.summary.image('Shuffle Matrix', mtrain.comb_matrix_image)
                    tf.summary.scalar("Learning Rate", mtrain.lr)

        with tf.name_scope("Validate"):
            with tf.variable_scope("Model", reuse=True):
                mvalid = DmnnModel(
                    is_training=False,
                    config=config,
                    input_=pose_seq_input
                )
                if not FLAGS.only_val:
                    tf.summary.scalar("Validation Loss", mvalid.loss)
                    tf.summary.scalar("Validation Accuracy", mvalid.accuracy)
                    tf.summary.image('Validation Mistakes per Class', mvalid.mistakes_per_class)
                    tf.summary.image('Validation Confusion Matrix', mvalid.confusion_matrix)
                    tf.summary.scalar("Loss Gap", (mvalid.loss - mtrain.loss))

        def restore_pretrained(session_to_restore):
            rnn_saver = tf.train.Saver(mtrain.rnn_vars)
            rnn_saver.restore(session_to_restore, config.pretrained_path)
            print("%s%s*** Restoring pretrained model ***"%(col.Style.BRIGHT,col.Fore.BLUE))

        global_step = tf.contrib.framework.get_or_create_global_step()
        sv = tf.train.Supervisor(
            logdir=FLAGS.save_path,
            global_step=global_step,
            save_summaries_secs=30 if not (FLAGS.only_val or FLAGS.debug) else 0,
            save_model_secs=900 if not (FLAGS.only_val or FLAGS.debug) else 0,
            init_fn=restore_pretrained if config.restore_pretrained else None
        )

        with sv.managed_session(config=session_config) as session:
            if FLAGS.debug:
                session = tf_debug.LocalCLIDebugWrapperSession(session, ui_type=FLAGS.ui_type)

                def my_break(datum, tensor):
                    return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
                session.add_tensor_filter("my_break", my_break)

            global_step_val = session.run(global_step)
            epoch = (global_step_val // pose_seq_input.train_epoch_size) + 1
            epoch_rem = pose_seq_input.train_epoch_size - (global_step_val % pose_seq_input.train_epoch_size)

            def finalize_nicely():
                summary_strs, global_step_val = session.run([sv.summary_op, sv.global_step])
                sv.summary_writer.add_summary(summary_strs, global_step_val)
                sv.summary_writer.flush()
                sv.saver.save(session, sv.save_path, global_step=sv.global_step)
                sv.summary_writer.add_session_log(
                    SessionLog(status=SessionLog.CHECKPOINT, checkpoint_path=sv.save_path),
                    global_step_val)
                sv.stop()

            # TODO: remove this ugly hack
            COUNT = 0
            while not sv.should_stop():
                try:
                    print("%sEpoch %d ..." % (col.Style.BRIGHT, epoch), end=("\n" if FLAGS.verbose else "\r"))

                    if not FLAGS.only_val:
                        if FLAGS.verbose:
                            print("    Training ...")
                        train_loss, train_acc = run_epoch(
                            session, mtrain, pose_seq_input, True, global_step,
                            sv.summary_writer, config, epoch_rem=epoch_rem
                        )
                    if FLAGS.validation_epoch or FLAGS.only_val:
                        if FLAGS.verbose:
                            print("    Validation ...")
                        val_accs = np.empty(10)
                        for i in range(1 if not FLAGS.only_val else 10):
                            valid_loss, val_acc = run_epoch(
                                session, mvalid, pose_seq_input, False, global_step,
                                sv.summary_writer, config, COUNT=COUNT
                            )
                            COUNT = COUNT + 1
                            val_accs[i] = val_acc
                        if FLAGS.only_val:
                            print("Accuracy Mean :%.3f ... Std:%.1E"%(np.mean(val_accs),np.std(val_accs)))

                except KeyboardInterrupt:
                    if FLAGS.only_val:
                        sv.stop()
                    else:
                        finalize_nicely()
                else:
                    if FLAGS.verbose:
                        print("")
                    if epoch % 10 == 0 and not FLAGS.only_val:
                        print("\n* * * Summary * * *")
                        print("Epoch: %d Learning rate: %s %.1E" %
                              (epoch, col.Style.BRIGHT, session.run(mtrain.lr)))
                        print("Epoch: %d Moving Mean Train Loss: %s%s %.3f" %
                              (epoch, col.Style.BRIGHT, col.Fore.YELLOW, train_loss))
                        if FLAGS.validation_epoch:
                            print("Epoch: %d Moving Mean Valid Loss: %s%s %.3f" %
                                  (epoch, col.Style.BRIGHT, col.Fore.GREEN, valid_loss))

                    if FLAGS.only_val:
                        sv.stop()
                    elif epoch >= config.max_max_epoch:
                        finalize_nicely()
                    else:
                        epoch += 1
                        epoch_rem = 0

if __name__ == "__main__":
    if not tf.gfile.Exists('./save'):
        tf.gfile.MkDir('./save')

    if FLAGS.save_path is None:
        FLAGS.__dict__.update({'save_path': './save/save_' + FLAGS.config_file})

    if FLAGS.only_val:
        assert tf.gfile.Exists(FLAGS.save_path)

    if FLAGS.clean_start:
        # If model dir already exists delete it
        if tf.gfile.Exists(FLAGS.save_path):
            tf.gfile.DeleteRecursively(FLAGS.save_path)

        # Clean python cache
        if tf.gfile.Exists("__pycache__"):
            tf.gfile.DeleteRecursively("__pycache__")

    tf.app.run()
