from datetime import datetime, timedelta
import sys
import os

# Add project root to the Python path so that Airflow can locate your modules
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from airflow import DAG
from airflow.operators.python import PythonOperator
from model_training.train import train_model

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

# Define the DAG for model training
with DAG(
    dag_id='model_training_dag',
    default_args=default_args,
    description='A DAG to run the model training pipeline',
    schedule_interval='@daily',  # Adjust the schedule as needed
    catchup=False,
) as dag:

    run_model_training = PythonOperator(
        task_id='run_model_training',
        python_callable=train_model,
        op_kwargs={
            'data_path': 'data/processed_housing_data.csv',  # Path to processed data
            'model_path': 'model.pkl'                         # Path to save the trained model
        }
    )

    run_model_training
