import numpy as np
import PIL.Image
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.training import monitored_session
import common_flags
import datasets
import data_provider


FLAGS = flags.FLAGS
common_flags.define()

flags.DEFINE_string('image_path_pattern', '',
                    'A file pattern with a placeholder for the image index.')


def dict_to_array(id_to_char, default_character = '?'):
    num_char_classes = max(id_to_char.keys()) + 1
    array = [default_character] * num_char_classes
    for k, v in id_to_char.items():
        array[k] = v
    return array


def get_dataset_image_size(dataset_name):
    ds_module = getattr(datasets, dataset_name)
    height, width, _ = ds_module.DEFAULT_CONFIG['image_shape']
    return width, height


def load_images(file_pattern, batch_size, dataset_name):
    width, height = get_dataset_image_size(dataset_name)
    images_actual_data = np.ndarray(shape = (batch_size, height, width, 3), dtype = 'uint8')
    for i in range(batch_size):
        path = file_pattern % i
        print("Reading %s" % path)
        pil_image = PIL.Image.open(open(path, 'rb'))
        images_actual_data[i, ...] = np.asarray(pil_image)
    return images_actual_data


def string_to_int64(raw_labels):
    raw_labels = [
        list(raw_labels[i])
        for i in range(32)
    ]

    dataset = common_flags.create_dataset(split_name = FLAGS.split_name)
    inv_charset = {v: k for k, v in dataset.charset.items()}

    return [[
            inv_charset[raw_labels[i][j]] for j in range(37)
            ] for i in range(32)]


def create_model(batch_size, dataset_name):
    width, height = get_dataset_image_size(dataset_name)
    dataset = common_flags.create_dataset(split_name = FLAGS.split_name)
    model = common_flags.create_model(
            num_char_classes = dataset.num_char_classes,
            seq_length = dataset.max_sequence_length,
            num_views = dataset.num_of_views,
            null_code = dataset.null_code,
            charset = dataset.charset)
    raw_images = tf.placeholder(shape = [batch_size, height, width, 3], dtype = tf.uint8)

    images = tf.map_fn(data_provider.preprocess_image, raw_images, dtype = tf.float32)
    labels = tf.placeholder(shape = [batch_size, dataset.max_sequence_length], dtype = tf.int64)

    endpoints = model.create_base(images, labels)
    return raw_images, labels, endpoints


def run(checkpoint, batch_size, dataset_name, images_data, labels):
    images_placeholder, labels_placeholder, endpoints = create_model(batch_size, dataset_name)
    session_creator = monitored_session.ChiefSessionCreator(checkpoint_filename_with_path = checkpoint)
    with monitored_session.MonitoredSession(session_creator = session_creator) as sess:
        prob = sess.run(endpoints, feed_dict = {images_placeholder: images_data, labels_placeholder: labels})
    return prob


_CHECKPOINT = '/tmp/attention_ocr/train/model.ckpt-0'
image_path_pattern = 'test-images/fsns_train_%02d.png'
raw_labels = [
            'Boulevard de Lunel░░░░░░░░░░░░░░░░░░░',
            'Rue de Provence░░░░░░░░░░░░░░░░░░░░░░',
            'Rue de Port Maria░░░░░░░░░░░░░░░░░░░░',
            'Avenue Charles Gounod░░░░░░░░░░░░░░░░',
            'Rue de l‘Aurore░░░░░░░░░░░░░░░░░░░░░░',
            'Rue de Beuzeville░░░░░░░░░░░░░░░░░░░░',
            'Rue d‘Orbey░░░░░░░░░░░░░░░░░░░░░░░░░░',
            'Rue Victor Schoulcher░░░░░░░░░░░░░░░░',
            'Rue de la Gare░░░░░░░░░░░░░░░░░░░░░░░',
            'Rue des Tulipes░░░░░░░░░░░░░░░░░░░░░░',
            'Rue André Maginot░░░░░░░░░░░░░░░░░░░░',
            'Route de Pringy░░░░░░░░░░░░░░░░░░░░░░',
            'Rue des Landelles░░░░░░░░░░░░░░░░░░░░',
            'Rue des Ilettes░░░░░░░░░░░░░░░░░░░░░░',
            'Avenue de Maurin░░░░░░░░░░░░░░░░░░░░░',
            'Rue Théresa░░░░░░░░░░░░░░░░░░░░░░░░░░',
            'Route de la Balme░░░░░░░░░░░░░░░░░░░░',
            'Rue Hélène Roederer░░░░░░░░░░░░░░░░░░',
            'Rue Emile Bernard░░░░░░░░░░░░░░░░░░░░',
            'Place de la Mairie░░░░░░░░░░░░░░░░░░░',
            'Rue des Perrots░░░░░░░░░░░░░░░░░░░░░░',
            'Rue de la Libération░░░░░░░░░░░░░░░░░',
            'Impasse du Capcir░░░░░░░░░░░░░░░░░░░░',
            'Avenue de la Grand Mare░░░░░░░░░░░░░░',
            'Rue Pierre Brossolette░░░░░░░░░░░░░░░',
            'Rue de Provence░░░░░░░░░░░░░░░░░░░░░░',
            'Rue du Docteur Mourre░░░░░░░░░░░░░░░░',
            'Rue d‘Ortheuil░░░░░░░░░░░░░░░░░░░░░░░',
            'Rue des Sarments░░░░░░░░░░░░░░░░░░░░░',
            'Rue du Centre░░░░░░░░░░░░░░░░░░░░░░░░',
            'Impasse Pierre Mourgues░░░░░░░░░░░░░░',
            'Rue Marcel Dassault░░░░░░░░░░░░░░░░░░'
        ]

images_data = load_images(image_path_pattern, 32, 'fsns')

result = np.ndarray(shape = (32, 32), dtype = 'float')

for i in range(32):
    tf.reset_default_graph()
    result[i, ...] = run(_CHECKPOINT, 32, 'fsns', images_data, np.array(string_to_int64(raw_labels)))
    first_item = raw_labels.pop(0)
    raw_labels.append(first_item)

for i in range(32):
    print('Image', i)
    for j in range(32):
        print('Text:', raw_labels[j], 'Prob:', result[(32 + j - i) % 32][i], 'truth' if i == j else None)
