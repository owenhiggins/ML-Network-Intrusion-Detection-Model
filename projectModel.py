import s3fs
from s3fs.core import S3FileSystem
import pickle
import json
import pandas as pd
import numpy as np
import requests
from zipfile import ZipFile
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
#import seaborn as sns
import sklearn.metrics as sk_metrics
import tempfile
import os

def build_model():
    # Preset matplotlib figure sizes.
    matplotlib.rcParams['figure.figsize'] = [9, 6]

    # To make the results reproducible, set the random seed value.
    tf.random.set_seed(22)


    # >>>GET DATA<<<
    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_wh = 's3://ece5984-bucket-caseygary/project/model_data/'  # Insert here
    # Get data from S3 bucket as a pickle file

    # Retrieve data from DataWarehouse
    x_train = np.load(s3.open('{}/{}'.format(DIR_wh, 'X_train_prj.pkl')), allow_pickle=True)
    x_test = np.load(s3.open('{}/{}'.format(DIR_wh, 'X_test_prj.pkl')), allow_pickle=True)
    y_train = np.load(s3.open('{}/{}'.format(DIR_wh, 'y_train_prj.pkl')), allow_pickle=True)
    y_test = np.load(s3.open('{}/{}'.format(DIR_wh, 'y_test_prj.pkl')), allow_pickle=True)

    # convert to tensor
    x_train, y_train = tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_test, y_test = tf.convert_to_tensor(x_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.float32)


    # >>> Setup Logistic Regression Model <<<
    # Adaped from https://www.tensorflow.org/guide/core/logistic_regression_core

    class Normalize(tf.Module):
        def __init__(self, x):
            # Initialize the mean and standard deviation for normalization
            self.mean = tf.Variable(tf.math.reduce_mean(x, axis=0))
            self.std = tf.Variable(tf.math.reduce_std(x, axis=0))

        def norm(self, x):
            # Normalize the input
            return (x - self.mean) / self.std

        def unnorm(self, x):
            # Unnormalize the input
            return (x * self.std) + self.mean

    norm_x = Normalize(x_train)
    x_train_norm, x_test_norm = norm_x.norm(x_train), norm_x.norm(x_test)


    def log_loss(y_pred, y):
        # Compute the log loss function
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
        return tf.reduce_mean(ce)

    class LogisticRegression(tf.Module):

        def __init__(self):
            self.built = False

        def __call__(self, x, train=True):
            # Initialize the model parameters on the first call
            if not self.built:
                # Randomly generate the weights and the bias term
                rand_w = tf.random.uniform(shape=[x.shape[-1], 1], seed=22)
                rand_b = tf.random.uniform(shape=[], seed=22)
                self.w = tf.Variable(rand_w)
                self.b = tf.Variable(rand_b)
                self.built = True
            # Compute the model output
            z = tf.add(tf.matmul(x, self.w), self.b)
            #z = tf.squeeze(z, axis=1)
            if train:
                return z
            return tf.sigmoid(z)


    #build initial model
    log_reg = LogisticRegression()


    def predict_class(y_pred, thresh=0.5):
        # Return a tensor with  `1` if `y_pred` > `0.5`, and `0` otherwise
        return tf.cast(y_pred > thresh, tf.float32)

    def accuracy(y_pred, y):
        # Return the proportion of matches between `y_pred` and `y`
        y_pred = tf.math.sigmoid(y_pred)
        y_pred_class = predict_class(y_pred)
        check_equal = tf.cast(y_pred_class == y, tf.float32)
        acc_val = tf.reduce_mean(check_equal)
        return acc_val

    # >>> Train Logistic Regression Model <<<
    # Train in batches
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_norm, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_norm, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=x_test.shape[0]).batch(batch_size)

    # Set training parameters
    epochs = 20
    learning_rate = 0.01
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    # Set up the training loop and begin training
    for epoch in range(epochs):
        batch_losses_train, batch_accs_train = [], []
        batch_losses_test, batch_accs_test = [], []
        print (f"Epoch: {epoch}")
        # Iterate over the training data
        batchNum = 0
        for x_batch, y_batch in train_dataset:

            batchNum += 1
            if batchNum % 100 ==0:
                print(batchNum)
            with tf.GradientTape() as tape:
                y_pred_batch = log_reg(x_batch)
                batch_loss = log_loss(y_pred_batch, y_batch)
            batch_acc = accuracy(y_pred_batch, y_batch)
            # Update the parameters with respect to the gradient calculations
            grads = tape.gradient(batch_loss, log_reg.variables)
            for g, v in zip(grads, log_reg.variables):
                v.assign_sub(learning_rate * g)
            # Keep track of batch-level training performance
            batch_losses_train.append(batch_loss)
            batch_accs_train.append(batch_acc)

        # Iterate over the testing data
        for x_batch, y_batch in test_dataset:
            y_pred_batch = log_reg(x_batch)
            batch_loss = log_loss(y_pred_batch, y_batch)
            batch_acc = accuracy(y_pred_batch, y_batch)
            # Keep track of batch-level testing performance
            batch_losses_test.append(batch_loss)
            batch_accs_test.append(batch_acc)

        # Keep track of epoch-level model performance
        # NOTE WE ARE NOT COLLECTING THIS DATA FOR THE PROJECT - CG
        train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
        test_loss, test_acc = tf.reduce_mean(batch_losses_test), tf.reduce_mean(batch_accs_test)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch: {epoch}, Training log loss: {train_loss:.3f}")


    # Save performance

    y_pred_train, y_pred_test = log_reg(x_train, train=False), log_reg(x_test, train=False)
    train_classes, test_classes = predict_class(y_pred_train), predict_class(y_pred_test)

    cm_train = sk_metrics.confusion_matrix(y_train, train_classes)
    cm_train_df = pd.DataFrame(cm_train)

    cm_test = sk_metrics.confusion_matrix(y_test, test_classes)
    cm_test_df = pd.DataFrame(cm_test)

    DIR = 's3://ece5984-bucket-caseygary/project/model/'  # insert here
    # Push confusion matrix data to S3 bucket as a pickle file
    with s3.open('{}/{}'.format(DIR, 'cm_train.pkl'), 'wb') as f:
        f.write(pickle.dumps(cm_train_df))

    with s3.open('{}/{}'.format(DIR, 'cm_test.pkl'), 'wb') as f:
        f.write(pickle.dumps(cm_test_df))

    # Save model temporarily
    class ExportModule(tf.Module):
        def __init__(self, model, norm_x, class_pred):
            # Initialize pre- and post-processing functions
            self.model = model
            self.norm_x = norm_x
            self.class_pred = class_pred

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
        def __call__(self, x):
            # Run the `ExportModule` for new data points
            x = self.norm_x.norm(x)
            y = self.model(x, train=False)
            y = self.class_pred(y)
            return y

    log_reg_export = ExportModule(model=log_reg,
                                  norm_x=norm_x,
                                  class_pred=predict_class)





    with tempfile.TemporaryDirectory() as tempdir:
        tf.saved_model.save(log_reg_export, f"{tempdir}/log_reg_export")
        DIR_ml = 's3://ece5984-bucket-caseygary/project/model/'
        #Push saved model to S3
        s3.put(f"{tempdir}/log_reg_export", f"{DIR_ml}/log_reg_export", recursive=True)

