from __future__ import annotations

import io
import os
import pickle

import requests
import time

from celery import Celery
from typing import Optional, Any, Union

from shared_constants import secret
from pipeline.predict import *
import redis

celery = Celery(__name__,
                borker=os.environ.get("CELERY_BROKER_URL"),
                backend=os.environ.get("CELERY_RESULT_BACKEND"))

redis_client = redis.from_url(os.environ.get('REDIS_JOB_DB'))

backend_host = os.environ.get('FASTAPI_HOST')

available_models = [
    "./tests/trained_epoch50_trial.pt"
]


def write(file_id: str, file_name: str, content: bytes | str) -> bool:
    """
    Upload file to the backend

    :param content: Content
    :param file_id: File ID
    :param file_name: File name under the ID
    :return: Success or not
    """
    if isinstance(content, str):
        content = content.encode('utf-8')

    r = requests.post(f'{backend_host}/file/internal/write',
                      params={'file_id': file_id, 'file_name': file_name}, headers={'auth': secret},
                      files={'upload': content})

    if not r.ok:
        print(r.text)

    return r.ok


def write_pickle(file_id: str, file_name: str, obj: Any) -> bool:
    bytes_io = io.BytesIO()
    pickle.dump(obj, bytes_io)
    bytes_io.seek(0)
    return write(file_id, file_name, bytes_io.read())


def read(file_id: str, file_name: Optional[str]) -> Union[bytes, None]:
    """
    Read a file previously uploaded to the backend by any worker

    :param file_id: File ID
    :param file_name: File name under the ID
    :return: File content (bytes)
    """
    p = {'job_id': file_id, 'file_name': file_name}
    r = requests.get(f'{backend_host}/file/internal/read', params=p, headers={'auth': secret})
    if not r.ok:
        return None
    return r.content


def read_text(file_id: str, file_name: Optional[str]) -> str:
    return read(file_id, file_name).decode('utf-8')


@celery.task(name="sleep")
def sleep(delay: int):
    print("sleeping")
    time.sleep(delay)
    return delay


@celery.task(name="predict")
def predict(job_id: str, model: int):
    # TODO: check if the files are properly uploaded
    seq_file = read(f'seq_{job_id}', None)
    sub_file = read(f'sub_{job_id}', None)
    while seq_file is None or sub_file is None:
        time.sleep(2)
        seq_file = read(f'seq_{job_id}', None)
        sub_file = read(f'sub_{job_id}', None)
    result, _, _ = predict_from_byte_input(available_models[model], seq_file, sub_file, params={'job_id': job_id})
    print(result)

    # update job status
    redis_client.hset(job_id, 'status', 1)
