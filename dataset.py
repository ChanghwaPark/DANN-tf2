import os

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_RESIZE_HEIGHT = 256
IMG_RESIZE_WIDTH = 256
SHUFFLE_BUFFER_SIZE = 5000


def _get_image_fraction(network):
    if network == 'resnet':
        return 224. / 256.
    elif network == 'alexnet':
        return 227. / 256.
    else:
        raise NotImplementedError(f"Network {network} is not implemented yet.")


def _get_image_size(network):
    if network == 'resnet':
        return [224, 224, 3]
    elif network == 'alexnet':
        return [227, 227, 3]
    else:
        raise NotImplementedError(f"Network {network} is not implemented yet.")


def _get_mean_values(network):
    if network == 'resnet':
        R_MEAN = 123.68 / 255.
        G_MEAN = 116.78 / 255.
        B_MEAN = 103.94 / 255.
    else:
        raise NotImplementedError(f"Network {network} is not implemented yet.")
    return [R_MEAN, G_MEAN, B_MEAN]


def _get_std_values(network):
    if network == 'resnet':
        return [0.229, 0.224, 0.225]
    elif network == 'alexnet':
        return [1., 1., 1.]
    else:
        raise NotImplementedError(f"Network {network} is not implemented yet.")


def normalize(image, network):
    image = tf.subtract(image, _get_mean_values(network))
    return tf.divide(image, _get_std_values(network))


def read_lines(fname):
    data = open(fname).readlines()
    fnames = []
    labels = []
    for line in data:
        fnames.append(line.split()[0])
        labels.append(int(line.split()[1]))
    return fnames, labels


def train_prep(fname, label=None, network='resnet'):
    image_string = tf.io.read_file(fname)
    image = tf.image.decode_jpeg(image_string, channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [IMG_RESIZE_HEIGHT, IMG_RESIZE_WIDTH])
    image = tf.image.random_crop(image, _get_image_size(network))
    image = tf.image.random_flip_left_right(image)
    image = normalize(image, network)
    if label is None:
        return image
    else:
        return image, label


def test_prep(fname, label, network='resnet'):
    image_string = tf.io.read_file(fname)
    image = tf.image.decode_jpeg(image_string, channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [IMG_RESIZE_HEIGHT, IMG_RESIZE_WIDTH])
    image = tf.image.central_crop(image, _get_image_fraction(network))
    image = normalize(image, network)
    return image, label


def prepare_dataset(FLAGS):
    source_file_path = os.path.join(FLAGS.datadir, FLAGS.dataset, FLAGS.source + '_list.txt')
    target_file_path = os.path.join(FLAGS.datadir, FLAGS.dataset, FLAGS.target + '_list.txt')

    s_fnames, s_labels = read_lines(source_file_path)
    t_fnames, t_labels = read_lines(target_file_path)

    s_train_prep = lambda x, y: train_prep(fname=x, label=y, network=FLAGS.network)
    t_train_prep = lambda x: train_prep(fname=x, network=FLAGS.network)
    t_test_prep = lambda x, y: test_prep(fname=x, label=y, network=FLAGS.network)

    s_train_ds = tf.data.Dataset \
        .from_tensor_slices((s_fnames, s_labels)) \
        .map(map_func=s_train_prep, num_parallel_calls=AUTOTUNE) \
        .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE) \
        .repeat(count=((FLAGS.steps * FLAGS.bs) // len(s_fnames) + 1)) \
        .batch(batch_size=FLAGS.bs) \
        .prefetch(buffer_size=AUTOTUNE)

    t_train_ds = tf.data.Dataset \
        .from_tensor_slices(t_fnames) \
        .map(map_func=t_train_prep, num_parallel_calls=AUTOTUNE) \
        .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE) \
        .repeat(count=((FLAGS.steps * FLAGS.bs) // len(t_fnames) + 1)) \
        .batch(batch_size=FLAGS.bs) \
        .prefetch(buffer_size=AUTOTUNE)

    t_test_ds = tf.data.Dataset \
        .from_tensor_slices((t_fnames, t_labels)) \
        .map(map_func=t_test_prep, num_parallel_calls=AUTOTUNE) \
        .batch(batch_size=4) \
        .prefetch(buffer_size=AUTOTUNE)

    return s_train_ds, t_train_ds, t_test_ds


def get_num_classes(dataset):
    if dataset == 'office':
        return 31
    elif dataset == 'office-home':
        return 65
    elif dataset == 'image-clef':
        return 12
    elif dataset == 'visda':
        return 12
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented yet.")
