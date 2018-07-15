import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import queues

import datasets
import data_provider
import matplotlib.pyplot as plt


class DataProviderTest(tf.test.TestCase):
    def setUp(self):
        tf.test.TestCase.setUp(self)

    def test_labels_correctly_shuffled(self):
        batch_size = 4
        data = data_provider.get_data(
                dataset = datasets.fsns_test.get_test_split(),
                batch_size = batch_size,
                augment = True,
                central_crop_size = None
        )
        with self.test_session() as sess, queues.QueueRunners(sess):
            images, labels, probs, texts = sess.run([data.images, data.labels, data.probs, data.texts])
            for i in range(batch_size * batch_size):
                plt.imshow(images[i])
                print(texts[i], probs[i], labels[i])


if __name__ == '__main__':
    tf.test.main()
