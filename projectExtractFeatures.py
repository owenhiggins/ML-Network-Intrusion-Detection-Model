import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
#START TUTORIAL IMPORTS
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
#teimport seaborn as sns
import sklearn.metrics as sk_metrics
import tempfile
import os


def feature_extract():
    #>>>GET DATA<<<
    s3 = S3FileSystem()
    # S3 bucket directory (data warehouse)
    DIR_wh = ''  # Insert here
    # Get data from S3 bucket as a pickle file
    prj_np = np.load(s3.open('{}/{}'.format(DIR_wh, 'clean_project_data.pkl')), allow_pickle=True)
    #read column names from numpy array
    features = prj_np.columns.to_list()
    #convert to dataframe
    project_data = pd.DataFrame(columns=features, data=prj_np, index=prj_np.index)

    #Split into training and test datasets
    train_dataset = project_data.sample(frac=0.75, random_state=1)
    test_dataset = project_data.drop(train_dataset.index)

    #split features from target columns
    x_train, y_train = train_dataset.iloc[:, 1:], train_dataset.iloc[:, :1]
    x_test, y_test = test_dataset.iloc[:, 1:], test_dataset.iloc[:, :1]


    # Push extracted features to data warehouse
    DIR_prj = '' # Insert here
    with s3.open('{}/{}'.format(DIR_prj, 'X_train_prj.pkl'), 'wb') as f:
        f.write(pickle.dumps(x_train))
    with s3.open('{}/{}'.format(DIR_prj, 'X_test_prj.pkl'), 'wb') as f:
        f.write(pickle.dumps(x_test))
    with s3.open('{}/{}'.format(DIR_prj, 'y_train_prj.pkl'), 'wb') as f:
        f.write(pickle.dumps(y_train))
    with s3.open('{}/{}'.format(DIR_prj, 'y_test_prj.pkl'), 'wb') as f:
        f.write(pickle.dumps(y_test))
