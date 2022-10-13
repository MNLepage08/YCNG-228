# https://cloud.google.com/storage/docs/reference/libraries
# https://googleapis.dev/python/storage/latest/client.html

import logging

import joblib
from google.api_core.exceptions import NotFound
from google.cloud import storage


def create_bucket(bucket_name):
    log = logging.getLogger()

    storage_client = storage.Client()
    if bucket_name not in [x.name for x in storage_client.list_buckets()]:
        bucket = storage_client.create_bucket(bucket_name)

        log.info("Bucket {} created".format(bucket.name))
    else:
        log.info("Bucket {} already exists".format(bucket_name))


def upload_file_to_bucket(model_file_name, bucket_name):
    log = logging.getLogger()
    log.warning(f'uploading {model_file_name} to {bucket_name}')
    client = storage.Client()
    b = client.get_bucket(bucket_name)
    blob = storage.Blob(model_file_name, b)
    with open(model_file_name, "rb") as model_file:
        blob.upload_from_file(model_file)

