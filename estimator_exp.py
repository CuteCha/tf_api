# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf


def linear_regression():
    feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir='./output/d1')

    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])

    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7., 0.])

    input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train},
                                                  y_train,
                                                  batch_size=4,
                                                  num_epochs=None,
                                                  shuffle=True)
    estimator.train(input_fn=input_fn, steps=1000)
    train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train},
                                                        y_train,
                                                        batch_size=4,
                                                        num_epochs=1000,
                                                        shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval},
                                                       y_eval,
                                                       batch_size=4,
                                                       num_epochs=1000,
                                                       shuffle=False)

    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

    print("train metrics: %r" % train_metrics)
    print("eval metrics: %r" % eval_metrics)


def lr_def():
    feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
    estimator = tf.estimator.Estimator(model_fn=lr_model_fn, model_dir="./output/d2")

    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])

    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7., 0.])

    input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train},
                                                  y_train,
                                                  batch_size=4,
                                                  num_epochs=None,
                                                  shuffle=True)
    estimator.train(input_fn=input_fn, steps=1000)
    train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train},
                                                        y_train,
                                                        batch_size=4,
                                                        num_epochs=1000,
                                                        shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval},
                                                       y_eval,
                                                       batch_size=4,
                                                       num_epochs=1000,
                                                       shuffle=False)

    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

    print("train metrics: %r" % train_metrics)
    print("eval metrics: %r" % eval_metrics)


def lr_model_fn(features, labels, mode):
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b

    loss = tf.reduce_sum(tf.square(y - labels))

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train)


def my_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def app1():
    import iris_data

    batch_size = 100
    train_steps = 1000

    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    my_feature_columns = [tf.feature_column.numeric_column(key=key) for key in train_x.keys()]

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3},
        model_dir='./output/d1')

    classifier.train(
        input_fn=lambda: iris_data.train_input_fn(train_x, train_y, batch_size),
        steps=train_steps)

    eval_result = classifier.evaluate(
        input_fn=lambda: iris_data.eval_input_fn(test_x, test_y, batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: iris_data.eval_input_fn(predict_x,
                                                 labels=None,
                                                 batch_size=batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id], 100 * probability, expec))


def app2():
    abalone_train = "./data/abalone/train.csv"
    abalone_test = "./data/abalone/test.csv"
    abalone_predict = "./data/abalone/predict.csv"

    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_train, target_dtype=np.int, features_dtype=np.float64)

    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)

    prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)

    print(prediction_set)
    print(type(prediction_set))

    model_params = {"learning_rate": 0.001}
    nn = tf.estimator.Estimator(model_fn=model_fn2, params=model_params, model_dir='./output/d2')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    nn.train(input_fn=train_input_fn, steps=5000)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    ev = nn.evaluate(input_fn=test_input_fn)
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": prediction_set.data},
        num_epochs=1,
        shuffle=False)
    predictions = nn.predict(input_fn=predict_input_fn)
    t = np.array(prediction_set.target)
    for i, p in enumerate(predictions):
        print("Prediction {}\t t:{}\tp{}".format(i + 1, t[i], p["ages"]))


def model_fn2(features, labels, mode, params):
    first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)
    second_hidden_layer = tf.layers.dense(first_hidden_layer, 10, activation=tf.nn.relu)
    output_layer = tf.layers.dense(second_hidden_layer, 1)

    predictions = tf.reshape(output_layer, [-1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"ages": predictions})

    loss = tf.losses.mean_squared_error(labels, predictions)

    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float64), predictions)
    }

    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    app2()
