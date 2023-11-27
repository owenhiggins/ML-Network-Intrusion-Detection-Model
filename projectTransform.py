import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pandas as pd
import pickle

def transform_data():

    s3 = S3FileSystem()
    # S3 bucket directory (data lake)
    DIR = 's3://ece5984-bucket-caseygary/project/data/'                                    # Insert here
    # Get data from S3 bucket as a pickle file
    data = np.load(s3.open('{}/{}'.format(DIR, 'project_data.pkl')), allow_pickle=True)

    # Split the IP Address columns into individual columns
    data[['IPV4_SRC_ADDR1', 'IPV4_SRC_ADDR2', 'IPV4_SRC_ADDR3', 'IPV4_SRC_ADDR4']] = (
        data['IPV4_SRC_ADDR'].str.split('.', expand=True))

    data[['IPV4_DST_ADDR1', 'IPV4_DST_ADDR2', 'IPV4_DST_ADDR3', 'IPV4_DST_ADDR4']] = (
        data['IPV4_DST_ADDR'].str.split('.', expand=True))


    # Drop string columns of IP addresses
    data.drop(columns=['IPV4_DST_ADDR', 'IPV4_SRC_ADDR', 'Attack'], axis=1, inplace=True)

    #Move the Target columns to the end of the dataframe
    move_me = data.pop("Label")
    #move_me_too = data.pop("Attack")

    #data.insert(len(data.columns), "Attack", move_me_too)
    data.insert(0, "Label", move_me)

    # S3 bucket directory
    DIR = 's3://ece5984-bucket-caseygary/project/data/'  # insert here
    # Push data to S3 bucket as a pickle file
    with s3.open('{}/{}'.format(DIR, 'clean_project_data.pkl'), 'wb') as f:
        f.write(pickle.dumps(data))