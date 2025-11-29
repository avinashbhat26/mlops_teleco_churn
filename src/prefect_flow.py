from prefect import flow, task

from . import data_ingestion, preprocess, train, evaluate


@task
def ingest_task():
    data_ingestion.main()


@task
def preprocess_task():
    preprocess.main()


@task
def train_task():
    train.main()


@task
def evaluate_task():
    evaluate.main()


@flow(name="telco-churn-ml-pipeline")
def main_flow():
    ingest_task()
    preprocess_task()
    train_task()
    evaluate_task()


if __name__ == "__main__":
    main_flow()
