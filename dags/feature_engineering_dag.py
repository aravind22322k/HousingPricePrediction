from datetime import datetime, timedelta
import sys
import os

# Optionally, add your project directory to the PYTHONPATH if it's not already included.
# For example, if your project root is two levels up:
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from airflow import DAG
from airflow.operators.python import PythonOperator
from feature_engineering.feature_engineering import preprocess_data

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    dag_id='feature_engineering_dag',
    default_args=default_args,
    description='A DAG to run the feature engineering pipeline',
    schedule_interval='@daily',  # Adjust the schedule as needed
    catchup=False,
) as dag:

    run_feature_engineering = PythonOperator(
        task_id='run_feature_engineering',
        python_callable=preprocess_data,
        op_kwargs={
            'input_path': 'data/raw_housing_data.csv',          # Path to raw data
            'output_path': 'data/processed_housing_data.csv'      # Path to store processed data
        }
    )

    run_feature_engineering
