from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from projectBatchIngest import ingest_data
from projectExtractFeatures import feature_extract
from projectModel import build_model
from projectTransform import transform_data
from projectPerformance import perf_data

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 9, 7),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    'projectDAG',
    default_args=default_args,
    description='ingest project data',
    schedule_interval=timedelta(days=1),
)

ingest_etl = PythonOperator(
    task_id='ingestDataset',
    python_callable=ingest_data,
    dag=dag,
)

transform_etl = PythonOperator(
    task_id='transformDataset',
    python_callable=transform_data,
    dag=dag,
)

extract_features = PythonOperator(
    task_id='featureExtraction',
    python_callable=feature_extract,
    dag=dag,
)

build_model = PythonOperator(
    task_id='buildModel',
    python_callable=build_model,
    dag=dag,
)

load_perf = PythonOperator(
    task_id='perfData',
    python_callable=perf_data,
    dag=dag,
)

ingest_etl >> transform_etl >> extract_features >> build_model >> load_perf