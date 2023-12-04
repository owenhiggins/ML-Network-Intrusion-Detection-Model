import s3fs
from s3fs.core import S3FileSystem
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timezone

def perf_data():

    s3 = S3FileSystem()
    # S3 bucket directory (data lake)
    DIR = '' #Insert Here
    DIRp = '' # Insert here
    # Get data from S3 bucket as a pickle file
    # Load as DataFrame
    with s3.open('{}/{}'.format(DIR, 'cm_train.pkl'), 'rb') as f:
        cm_train_df = pickle.load(f)

    with s3.open('{}/{}'.format(DIR, 'cm_test.pkl'), 'rb') as f:
        cm_test_df = pickle.load(f)

    def perf_report(df, setname):
        dfnp = df.to_numpy()
        true_negative = dfnp[0][0]
        false_negative = dfnp[1][0]
        true_positive = dfnp[1][1]
        false_positive = dfnp[0][1]

        true_negative_rate = true_negative / (true_negative + false_negative)
        true_positive_rate = true_positive / (true_positive + false_positive)
        false_negative_rate = false_negative / (false_negative + true_negative)
        false_positive_rate = false_positive / (false_positive + true_positive)

        column_names = ['True Negative', 'False Negative', 'True Positive', 'False Positive']
        data = [['{:.2%}'.format(true_negative_rate), '{:.2%}'.format(false_negative_rate),
                 '{:.2%}'.format(true_positive_rate), '{:.2%}'.format(false_positive_rate)]]

        results = pd.DataFrame(data, columns=column_names)

        str_date = datetime.now(timezone.utc)
        results['Performance Data Date'] = str(str_date)
        results['Set Name'] = setname

        return results

    testset_results = perf_report(cm_test_df, 'Test')
    trainset_results = perf_report(cm_train_df, 'Train')

    # alternatively should be loaded as an entry in a database
    perf_report = pd.concat([trainset_results, testset_results])


    # Convert DataFrames to CSV and Push to S3 bucket
    with s3.open('{}/{}'.format(DIR, 'cm_train.csv'), 'w') as f:
        cm_train_df.to_csv(f, index=False)

    with s3.open('{}/{}'.format(DIR, 'cm_test.csv'), 'w') as f:
        cm_test_df.to_csv(f, index=False)

    with s3.open('{}/{}'.format(DIRp, 'perf_report.csv'), 'w') as f:
        perf_report.to_csv(f, index=False)

