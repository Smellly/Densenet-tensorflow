import tempfile
import os
import random

import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from .base_provider import ImagesDataSet, DataProvider
from .downloader import download_data_url

import tensorflow as tf


def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    flip = random.getrandbits(1)
    if flip:
        image = image[:, ::-1, :]
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    # randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]
    return cropped


def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=4)
    return new_images


class LivenessDataSet(ImagesDataSet):
    def __init__(self, images, labels, shuffle, normalization, num_examples,
                n_classes=2,
                augmentation=False):
        """
        Args:
            images: 4D numpy array
            labels: 2D or 1D numpy array
            n_classes: `int`, number of cifar classes - 10 or 100
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            augmentation: `bool`
        """
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == 'once_prior_train':
            self.shuffle_every_epoch = False
            images, labels = self.shuffle_images_and_labels(images, labels)
        elif shuffle == 'every_epoch':
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")

        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self._num_examples = num_examples
        self.augmentation = augmentation
        self.normalization = normalization
        # self.images = self.normalize_images(images, self.normalization)
        # self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            images, labels = self.shuffle_images_and_labels(
                self.images, self.labels)
        else:
            images, labels = self.images, self.labels
        if self.augmentation:
            images = augment_all_images(images, pad=4)
        self.epoch_images = images
        self.epoch_labels = labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice


class LivenessDataProvider(DataProvider):
    """Abstract class for cifar readers"""

    def __init__(self, 
                 shuffle=None, normalization=None,
                 train_save_path=None, val_save_path=None, test_save_path=None,
                 one_hot=True, **kwargs):
        """
        Args:
            validation_set: `bool`.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            save_path: `str` or None
                /path/to/filelist.txt
                e.g. in train_save_path.txt
                    .../.../.../a.jpg 1
                    .../.../.../b.jpg 2
            one_hot: `bool`, return lasels one hot encoded
        """
        assert train_save_path, "train datalist is none"
        assert test_save_path, "test datalist is none"

        self._train_save_path = Path(train_save_path) 
        self._test_save_path = Path(test_save_path) 
        self._val_save_path = Path(val_save_path) \
                    if val_save_path is not None else None
        self.one_hot = one_hot
        self._n_classes = 2

        # add train and validations datasets
        trainTFRecords = Path("data_providers/LivenessTFRecordData/train.tfrecords")

        if not trainTFRecords.is_file():
            self.ToTFRecords("train")
        else:
            self._num_examples = sum(1 for _ in
                    tf.python_io.tf_record_iterator(str(trainTFRecords)))
        images, labels = self.FromTFRecords(trainTFRecords)


        self.train = LivenessDataSet(
            images=images, labels=labels,
            n_classes=self.n_classes, shuffle=shuffle,
            num_examples=self._num_examples,
            normalization=normalization)

        # add test set
        testTFRecords = Path("data_providers/LivenessTFRecordData/test.tfrecords")
        if not testTFRecords.is_file():
            self.ToTFRecords("test")
        images, labels = self.FromTFRecords(testTFRecords)

        self.test = LivenessDataSet(
            images=images, labels=labels,
            shuffle=None, n_classes=self.n_classes,
            num_examples=sum(1 for _ in
                tf.python_io.tf_record_iterator(str(testTFRecords))),
            normalization=normalization)

        # add val set
        if self._val_save_path is not None :
            valTFRecords = Path("data_providers/LivenessTFRecordData/val.tfrecords")
            if not valTFRecords.is_file():
                self.ToTFRecords("val")
            images, labels = self.FromTFRecords(valTFRecords)
            self.validation = LivenessDataSet(
                images=images, labels=labels,
                num_examples=sum(1 for _ in 
                    tf.python_io.tf_record_iterator(str(valTFRecords))),
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization)
        else:
            self.validation = self.test

    @property
    def data_shape(self):
        return (256,256, 3)

    @property
    def n_classes(self):
        return self._n_classes

    # from imglist to tfrecords
    # filetype : train, val, test
    def ToTFRecords(self, filetype):
        if filetype == "train":
            savepath = self._train_save_path
        elif filetype == "val":
            savepath = self._val_save_path
        elif filetype == "test":
            savepath = self._test_save_path
        else:
            print("Wrong with filetype: train, val, test")
            exit(0)
        with savepath.open() as f:
            filenames = f.readlines()
            self._num_examples = len(filenames)

        print("Saving dataset %s to TFRecords"%filetype)
        folder = Path("data_providers/LivenessTFRecordData") 
        if not folder.exists():
            folder.mkdir()
        writer = tf.python_io.TFRecordWriter(str(folder / (filetype +
                                                            ".tfrecords")))
        for item in tqdm(filenames):
            img_path, label = item.split()
            img = Image.open(img_path)
            # print(img.size)
            # print(type(img.size))
            if img.size != (256, 256):
                img = img.resize((256, 256))
            img_raw = img.tobytes()              #将图片转化为原生bytes
            example = tf.train.Example(
                    features=tf.train.Features(
                                feature={
                    "label": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[eval(label)])),
                    'img_raw': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())  #序列化为字符串
        writer.close()
        
        # images_res = []
        # labels_res = []
        # for fname in filenames:
        #     with open(fname, 'rb') as f:
        #         images_and_labels = pickle.load(f, encoding='bytes')
        #     images = images_and_labels[b'data']
        #     images = images.reshape(-1, 3, 32, 32)
        #     images = images.swapaxes(1, 3).swapaxes(1, 2)
        #     images_res.append(images)
        #     labels_res.append(images_and_labels[labels_key])
        # images_res = np.vstack(images_res)
        # labels_res = np.hstack(labels_res)
        # if self.one_hot:
        #     labels_res = self.labels_to_one_hot(labels_res)
        # return images_res, labels_res

    # read and decode from tfrecords
    def FromTFRecords(self, filename):
        #根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([str(filename)])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                                   'label': tf.FixedLenFeature([], tf.int64),
                                                   'img_raw' : tf.FixedLenFeature([], tf.string),
                                           })

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [256, 256, 3])
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        label = tf.cast(features['label'], tf.int32)

        return img, label        





