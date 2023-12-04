import s3fs
from s3fs.core import S3FileSystem
import pickle
import json
import pandas as pd
import requests
from zipfile import ZipFile

def ingest_data():

    #DOWNLOAD URL OF PROJECT BASE DATASET
    url = (
        "https://api.rdm.uq.edu.au/production/files/"
        "c31a9f50-ef99-11ed-ab7b-c7846b13c8a9/download")

    #PATH TO SAVE DOWNLOADED FILE IN CURRET DIR
    output = r'downloaded_file.zip'

    
    zipDIR = '' #Input Directory Here

    s3 = S3FileSystem()

    r = requests.get(url)
    with s3.open('{}/{}'.format(zipDIR, output), 'wb') as f:
        f.write(r.content)

    with ZipFile(s3.open('{}/{}'.format(zipDIR, output), 'rb')) as myzip:
        with myzip.open('88695f0f620eb568_MOHANAD_A4706/data/NF-UNSW-NB15.csv') as myfile:
            downloaded_file = pd.read_csv(myfile)


    # S3 bucket directory
    DIR = ''  # insert here
    # Push data to S3 bucket as a pickle file
    with s3.open('{}/{}'.format(DIR, 'project_data.pkl'), 'wb') as f:
        f.write(pickle.dumps(downloaded_file))
