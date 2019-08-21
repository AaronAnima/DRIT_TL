import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags


def get_Y2X_train():
    cat_images_path = tl.files.load_file_list(path='/home/asus/Workspace/new_code/cat2dog/trainA/',
                                                            regx='.*.jpg', keep_prefix=True, printable=False)
    dog_images_path = tl.files.load_file_list(path='/home/asus/Workspace/new_code/cat2dog/trainB/',
                                                            regx='.*.jpg', keep_prefix=True, printable=False)
    len_cat = len(cat_images_path)
    len_dog = len(dog_images_path)
    dataset_len = min(len_cat, len_dog)
    flags.len_dataset = dataset_len
    cat_images_path = cat_images_path[0:dataset_len]
    dog_images_path = dog_images_path[0:dataset_len]

    def generator_train():
        for cat_path, dog_path in zip(cat_images_path, dog_images_path):
            yield cat_path.encode('utf-8'), dog_path.encode('utf-8')

    def _map_fn(image_path, image_path_2):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = image[20:198, :, :]  # central crop
        image = tf.image.resize(image, [256, 256])  # how to do BICUBIC?
        image = image * 2 - 1

        image2 = tf.io.read_file(image_path_2)
        image2 = tf.image.decode_jpeg(image2, channels=3)  # get RGB with 0~1
        image2 = tf.image.convert_image_dtype(image2, dtype=tf.float32)
        image2 = image2[20:198, :, :]  # central crop
        image2 = tf.image.resize(image2, [256, 256])  # how to do BICUBIC?
        image2 = image2 * 2 - 1

        return image, image2

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.string, tf.string))
    ds = train_ds.shuffle(buffer_size=4096)
    # ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
    n_step_epoch = int(dataset_len // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)
    ds = ds.repeat(n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(flags.batch_size_train)
    ds = ds.prefetch(buffer_size=2)
    return ds


def get_cat2dog_eval(batch_num=1, datatype='X'):
    cat_images_path = tl.files.load_file_list(path='/home/asus/Workspace/dataset/cat2dog/testA/',
                                                            regx='.*.jpg', keep_prefix=True, printable=False)
    dog_images_path = tl.files.load_file_list(path='/home/asus/Workspace/dataset/cat2dog/testB/',
                                                            regx='.*.jpg', keep_prefix=True, printable=False)
    len_cat = len(cat_images_path)
    len_dog = len(dog_images_path)
    dataset_len = min(len_cat, len_dog)
    total_num = flags.batch_size_eval * batch_num
    if total_num > dataset_len:
        print("The total num of the eval data is larger than dataset length.")

    index_cat = np.random.choice(dataset_len, total_num)
    index_dog = np.random.choice(dataset_len, total_num)
    cat_images_path = cat_images_path[index_cat]
    dog_images_path = dog_images_path[index_dog]

    if datatype is 'X':
        images_path = cat_images_path
    else:
        images_path = dog_images_path

    eval_images = []
    for image_path in images_path:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = image[20:198, :, :]  # central crop
        image = tf.image.resize(image, [256, 256])  # how to do BICUBIC?
        image = image * 2 - 1
        eval_images.append(image)

    return eval_images





