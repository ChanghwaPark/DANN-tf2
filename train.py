import tensorflow as tf
from tqdm import tqdm

from utils import lr_scheduler

LOG_FREQUENCY = 10
cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bce = tf.keras.losses.BinaryCrossentropy()


@tf.function
def train_step(FLAGS, s_images, s_labels, t_images, classifier, discriminator,
               main_optimizer, disc_optimizer,
               train_loss_main, train_loss_s_class, train_loss_dann, train_loss_disc,
               s_train_accuracy, dann_schedule):
    with tf.GradientTape(persistent=True) as tape:
        dw = tf.constant(FLAGS.dw, dtype=tf.float32)
        wd = tf.constant(FLAGS.wd, dtype=tf.float32)
        ds = tf.constant(dann_schedule, dtype=tf.float32)

        s_predictions, s_features = classifier(s_images, training=True, trim=FLAGS.trim)
        _, t_features = classifier(t_images, training=True, trim=FLAGS.trim)
        loss_s_class = cce(s_labels, s_predictions)

        # DANN loss
        if FLAGS.dw > 0:
            s_disc_outputs = discriminator(s_features, training=True)
            t_disc_outputs = discriminator(t_features, training=True)
            loss_disc = 0.5 * (bce(tf.ones_like(s_disc_outputs), s_disc_outputs) +
                               bce(tf.zeros_like(t_disc_outputs), t_disc_outputs))
            loss_dann = 0.5 * (bce(tf.zeros_like(s_disc_outputs), s_disc_outputs) +
                               bce(tf.ones_like(t_disc_outputs), t_disc_outputs))
        else:
            loss_disc = tf.constant(0, dtype=tf.float32)
            loss_dann = tf.constant(0, dtype=tf.float32)

        # Weight decay
        if FLAGS.wd > 0:
            if FLAGS.dw > 0:
                var_d = discriminator.trainable_variables
                d_decay = tf.reduce_mean([tf.nn.l2_loss(v) for v in var_d])
            else:
                d_decay = tf.constant(0, dtype=tf.float32)
            var_g = classifier.trainable_variables
            g_decay = tf.reduce_mean([tf.nn.l2_loss(v) for v in var_g if 'bn' not in v.name])
        else:
            d_decay = tf.constant(0, dtype=tf.float32)
            g_decay = tf.constant(0, dtype=tf.float32)

        # Optimizer
        if FLAGS.dw > 0:
            loss_disc = loss_disc + wd * d_decay
        else:
            loss_disc = tf.constant(0, dtype=tf.float32)

        loss_main = loss_s_class + ds * dw * loss_dann + wd * g_decay

    disc_gradients = tape.gradient(loss_disc, discriminator.trainable_variables)
    main_gradients = tape.gradient(loss_main, classifier.trainable_variables)
    del tape

    # Scale 0.1 to the pre-trained resnet variables.
    if FLAGS.differ_gradients:
        main_gvs = [(10 * g, v) if 'dense' in v.name else (g, v)
                    for g, v in zip(main_gradients, classifier.trainable_variables)]
    else:
        main_gvs = zip(main_gradients, classifier.trainable_variables)

    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    main_optimizer.apply_gradients(main_gvs)

    train_loss_disc.update_state(loss_disc)
    train_loss_main.update_state(loss_main)
    train_loss_s_class.update_state(loss_s_class)
    train_loss_dann.update_state(loss_dann)
    s_train_accuracy.update_state(s_labels, s_predictions)

    if main_optimizer.iterations % LOG_FREQUENCY == 0:
        tf.summary.scalar('loss_disc', train_loss_disc.result(), step=main_optimizer.iterations)
        tf.summary.scalar('loss_main', train_loss_main.result(), step=main_optimizer.iterations)
        tf.summary.scalar('loss_s_class', train_loss_s_class.result(), step=main_optimizer.iterations)
        tf.summary.scalar('loss_dann', train_loss_dann.result(), step=main_optimizer.iterations)
        tf.summary.scalar('s_train_accuracy', s_train_accuracy.result(), step=main_optimizer.iterations)
        if main_optimizer.iterations % FLAGS.steps != 0:
            train_loss_disc.reset_states()
            train_loss_main.reset_states()
            train_loss_s_class.reset_states()
            train_loss_dann.reset_states()
            s_train_accuracy.reset_states()


def train_epoch(FLAGS, s_train_ds, t_train_ds, classifier, discriminator, main_optimizer, disc_optimizer,
                dann_schedule, train_loss_main, train_loss_s_class, train_loss_dann, train_loss_disc,
                s_train_accuracy, summary_writer):
    s_train_it = iter(s_train_ds)
    t_train_it = iter(t_train_ds)

    # Make a progress bar
    steps = tqdm(range(FLAGS.steps), leave=False)

    # Training
    for _ in steps:
        s_images, s_labels = next(s_train_it)
        t_images = next(t_train_it)
        current_lr = lr_scheduler(main_optimizer.iterations / (FLAGS.steps * FLAGS.epochs), FLAGS.lr)
        tf.keras.backend.set_value(main_optimizer.lr, current_lr)
        tf.keras.backend.set_value(disc_optimizer.lr, current_lr)
        train_step(FLAGS, s_images, s_labels, t_images, classifier, discriminator,
                   main_optimizer, disc_optimizer,
                   train_loss_main, train_loss_s_class, train_loss_dann, train_loss_disc,
                   s_train_accuracy, dann_schedule)
        summary_writer.flush()
    steps.close()


@tf.function
def test_step(t_images, t_labels, classifier, t_test_accuracy):
    t_predictions, _ = classifier(t_images, training=False)
    t_test_accuracy.update_state(t_labels, t_predictions)


def test_epoch(t_test_ds, classifier, t_test_accuracy):
    # Testing
    for test_images, test_labels in t_test_ds:
        test_step(test_images, test_labels, classifier, t_test_accuracy)
