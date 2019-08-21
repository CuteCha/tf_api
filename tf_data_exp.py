# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


def data_test1():
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))


def data_test2():
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element))
        except tf.errors.OutOfRangeError:
            print("end!")


def data_test3():
    tf.enable_eager_execution()
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    dataset = dataset.repeat(5).batch(3)
    for one_element in dataset:
        print(one_element)


def data_test4():
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    dataset = dataset.repeat(5).batch(3)
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element))
        except tf.errors.OutOfRangeError:
            print("end!")


def data_test5():
    tf.enable_eager_execution()
    np.random.seed(97)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2)))
    dataset = dataset.repeat(5).batch(3)
    for one_element in dataset:
        print(one_element)


def data_test6():
    tf.enable_eager_execution()
    np.random.seed(97)
    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "y": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "x": np.random.uniform(size=(5, 2))
        }
    )
    dataset = dataset.repeat(5).batch(3)
    for one_element in dataset:
        print("-" * 72)
        print(one_element)
        print(one_element["y"])
        print(one_element["x"])
        print("-"*72)


if __name__ == '__main__':
    data_test6()
