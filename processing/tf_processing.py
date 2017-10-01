import tensorflow as tf

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()


def _parse_function(example_proto):
    features = {"img_raw": tf.FixedLenFeature((), tf.string, default_value=""),
                "category_id": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)

    # image = tf.decode_raw(parsed_features["img_raw"], tf.float32)
    # image = tf.reshape(image, [180, 180, 3])

    return parsed_features["img_raw"], parsed_features["category_id"]


training_filenames = ["../input/output_file.tfrecords"]
dataset = tf.contrib.data.TFRecordDataset(training_filenames)

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset.map(_parse_function)
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(1)
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
sess.run(iterator.get_next())
