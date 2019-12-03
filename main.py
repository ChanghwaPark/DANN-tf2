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
from train import train_epoch, test_epoch
from utils import delete_existing, dann_scheduler

ETA_FREQUENCY = 5
FLAGS = flags.FLAGS

flags.DEFINE_string('gpu', '0', 'GPU number')
flags.DEFINE_string('dataset', 'office', 'Dataset')
flags.DEFINE_string('source', 'amazon', 'Source domain')
flags.DEFINE_string('target', 'webcam', 'Target domain')
flags.DEFINE_string('network', 'resnet', 'Network architecture')
flags.DEFINE_string('logdir', 'results/logs', 'Logfile directory')
flags.DEFINE_string('datadir', 'datasets', 'Directory for datasets')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('bs', 64, 'Batch size')
flags.DEFINE_integer('steps', 500, 'Number of steps per epoch')
flags.DEFINE_integer('epochs', 200, 'Number of epochs')
flags.DEFINE_float('wd', 5e-4, 'Weight decay hyper-parameter')
flags.DEFINE_float('dw', 1e-2, 'Adversarial domain adaptation hyper-parameter')
flags.DEFINE_integer('trim', 0, 'Feature layer selection')
flags.DEFINE_integer('differ_gradients', 1, 'Differentiation on the dense layer gradients')
flags.DEFINE_boolean('debug', False, 'Debug mode flag')

logging.set_verbosity(logging.INFO)


def main(_):
    # Print FLAG values
    pprint(FLAGS.flag_values_dict())

    # Define GPU configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    # Define model name
    setup_list = [
        FLAGS.source,
        FLAGS.target,
        FLAGS.network,
        f"dw_{FLAGS.dw}",
        f"trim_{FLAGS.trim}",
        f"dg_{FLAGS.differ_gradients}"
    ]
    model_name = '_'.join(setup_list)
    model_dir = os.path.join(FLAGS.logdir, model_name)
    delete_existing(model_dir)
    os.mkdir(model_dir)
    print(colored(f"Model name: {model_name}", 'blue'))

    # Debug mode
    if FLAGS.debug:
        tf.config.experimental_run_functions_eagerly(True)

    classifier = network.Classifier(network=FLAGS.network, num_classes=get_num_classes(FLAGS.dataset))
    discriminator = network.Discriminator()
    disc_optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.lr, momentum=0.9, nesterov=True)
    main_optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.lr, momentum=0.9, nesterov=True)

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
            # Prepare dataset
            s_train_ds, t_train_ds, t_test_ds = prepare_dataset(FLAGS)
            dann_schedule = dann_scheduler(epoch / FLAGS.epochs)

            train_epoch(FLAGS, s_train_ds, t_train_ds, classifier, discriminator, main_optimizer, disc_optimizer,
                        dann_schedule, train_loss_main, train_loss_s_class, train_loss_dann, train_loss_disc,
                        s_train_accuracy, summary_writer)

            test_epoch(t_test_ds, classifier, t_test_accuracy)

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
            train_loss_main.reset_states()
            train_loss_s_class.reset_states()
            train_loss_dann.reset_states()
            train_loss_disc.reset_states()
            s_train_accuracy.reset_states()
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
