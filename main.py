import datetime
import os
import time
from collections import deque
from pprint import pprint
from statistics import mean

import tensorflow as tf
from absl import app, flags, logging
from pytz import timezone
from termcolor import colored
from tqdm import tqdm

import network
from dataset import prepare_dataset, get_num_classes
from train import train_step, test_step
from utils import delete_existing, dann_scheduler

ETA_FREQUENCY = 5
FLAGS = flags.FLAGS

flags.DEFINE_string('gpu', '0', 'GPU number')
flags.DEFINE_string('dataset', 'office', 'Dataset')
flags.DEFINE_string('source', 'amazon', 'Source domain')
flags.DEFINE_string('target', 'webcam', 'Target domain')
flags.DEFINE_string('network', 'resnet', 'Network architecture')
flags.DEFINE_string('logdir', 'results/logs', 'Logfile directory')
flags.DEFINE_string('datadir', '/home/intel/omega/datasets', 'Directory for datasets')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('lr_decay', 1.0, 'Learning rate decay per epoch')
flags.DEFINE_integer('bs', 36, 'Batch size')
flags.DEFINE_integer('steps', 500, 'Number of steps per epoch')
flags.DEFINE_integer('epochs', 200, 'Number of epochs')
flags.DEFINE_float('wd', 5e-4, 'Weight decay hyper-parameter')
flags.DEFINE_float('dw', 1e-2, 'Adversarial domain adaptation hyper-parameter')
flags.DEFINE_integer('trim', 0, 'Feature layer selection')
flags.DEFINE_integer('differ_gradients', 1, 'Differentiation on the dense layer gradients')

logging.set_verbosity(logging.INFO)


def main(_):
    # Print FLAG values
    pprint(FLAGS.flag_values_dict())

    # Define GPU configuration
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu], 'GPU')
    # tf.config.experimental.set_memory_growth(gpus[FLAGS.gpu], True)
    # gpu_config = tf.compat.v1.ConfigProto()
    # gpu_config.gpu_options.allow_growth = True
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    # Define model name
    setup_list = [
        FLAGS.source,
        FLAGS.target,
        FLAGS.network,
        f"dw_{FLAGS.dw}",
        f"lrd_{FLAGS.lr_decay}",
        f"trim_{FLAGS.trim}",
        f"dg_{FLAGS.differ_gradients}"
    ]
    model_name = '_'.join(setup_list)
    model_dir = os.path.join(FLAGS.logdir, model_name)
    delete_existing(model_dir)
    os.mkdir(model_dir)
    print(colored(f"Model name: {model_name}", 'blue'))

    # Prepare dataset
    s_train_ds, t_train_ds, t_test_ds = prepare_dataset(FLAGS)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(FLAGS.lr, decay_steps=FLAGS.steps,
                                                                 decay_rate=FLAGS.lr_decay, staircase=False)

    classifier = network.Classifier(network=FLAGS.network, num_classes=get_num_classes(FLAGS.dataset))
    discriminator = network.Discriminator()
    disc_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    main_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

    train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')
    train_loss_main = tf.keras.metrics.Mean(name='train_loss_main')
    train_loss_s_class = tf.keras.metrics.Mean(name='train_loss_s_class')
    train_loss_dann = tf.keras.metrics.Mean(name='train_loss_dann')
    s_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='s_train_accuracy')
    t_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='t_test_accuracy')

    summary_writer = tf.summary.create_file_writer(model_dir)

    start_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %I:%M:%S %p')
    print(colored(start_time, "blue"))
    epoch_time = deque(maxlen=5)
    temp_time = time.time()

    with summary_writer.as_default():
        for epoch in range(FLAGS.epochs):
            dann_schedule = dann_scheduler(epoch / FLAGS.epochs)

            s_train_it = iter(s_train_ds)
            t_train_it = iter(t_train_ds)

            # Make a progress bar
            steps = tqdm(range(FLAGS.steps), leave=False)

            # Training
            for _ in steps:
                s_images, s_labels = next(s_train_it)
                t_images = next(t_train_it)
                train_step(s_images, t_images, s_labels, FLAGS, classifier, discriminator,
                           disc_optimizer, main_optimizer,
                           train_loss_disc, train_loss_main, train_loss_s_class, train_loss_dann,
                           s_train_accuracy, dann_schedule)
            steps.close()

            # Testing
            for test_images, test_labels in t_test_ds:
                test_step(test_images, test_labels, classifier, t_test_accuracy)

            tf.summary.scalar('t_test_accuracy', t_test_accuracy.result(), step=main_optimizer.iterations)

            template = 'Epoch {}, Disc_loss: {:.5f}, Main_loss: {:.5f}, S_class_loss: {:.5f}, Dann_loss: {:.5f}, ' \
                       'Source train accuracy: {:.3f}, Target test Accuracy: {:.3f}'
            print(template.format(epoch + 1,
                                  train_loss_disc.result().numpy(),
                                  train_loss_main.result().numpy(),
                                  train_loss_s_class.result().numpy(),
                                  train_loss_dann.result().numpy(),
                                  s_train_accuracy.result().numpy() * 100,
                                  t_test_accuracy.result().numpy() * 100))

            # Reset the metric for the next epoch
            t_test_accuracy.reset_states()

            epoch_time.appendleft(time.time() - temp_time)
            temp_time = time.time()
            if (epoch + 1) % ETA_FREQUENCY == 0:
                needed_time = mean(epoch_time) * (FLAGS.epochs - epoch - 1)
                eta = datetime.datetime.now(timezone('Asia/Seoul')) + datetime.timedelta(seconds=needed_time)
                print(f"Needed time: {str(datetime.timedelta(seconds=round(needed_time)))}, "
                      f"ETA: {eta.strftime('%Y-%m-%d %I:%M:%S %p')}")


if __name__ == "__main__":
    app.run(main)
