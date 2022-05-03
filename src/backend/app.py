import datetime
import uuid
import os
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, Header
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

import worker
from shared_constants import secret

import redis

app = FastAPI()

# CORS For local development (TODO: Remove this in production)
app.add_middleware(CORSMiddleware, allow_origins="*")

# Create upload folder
UPLOAD_FOLDER = Path('jobs')
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

redis_client = redis.from_url(os.environ.get('REDIS_JOB_DB'))


@app.get("/")
def hello_world():
    return "Hello, World"


@app.post("/file/upload")
async def upload_file(job_id: str, upload: UploadFile, job_type: Literal['seq', 'sub']):
    """
    Upload a csv file containing sequences to the server for preprocessing

    :param job_id: Job ID
    :param upload: File to upload
    :param job_type: Type of file, either 'seq' or 'sub'
    :return File ID
    """
    if not job_type in ['seq', 'sub']:
        raise HTTPException(400)
    if not job_id:
        raise HTTPException(400)
    file_id = job_type + '_' + job_id

    # Write file
    (UPLOAD_FOLDER / file_id).write_bytes(await upload.read())

    return {'file_id': file_id}


def internal_validate(job_id: str, file_name: Optional[str], auth: str) -> Path:
    if not auth == secret:
        raise HTTPException(401)

    return UPLOAD_FOLDER / job_id if not file_name else UPLOAD_FOLDER / (job_id + '.dir') / file_name


@app.get("/file/internal/read")
async def file_internal_read(job_id: str, file_name: Optional[str] = None, auth: str = Header(None)):
    """
    Read internal temporary file stored by a worker

    File Structure:

    UPLOAD_FOLDER
    - job_id: File uploaded by the user
    - file_id.dir
        - file_name: File uploaded by a worker

    :param job_id: Sequences file ID
    :param file_name: File name under the ID
    :param auth: Secret string to verify that the request comes from a worker
                 (must be the same as the text in the secret variable)
    :return: File
    """
    path = internal_validate(job_id, file_name, auth)

    if not path.is_file():
        raise HTTPException(404)

    return FileResponse(path)


@app.post("/file/internal/write")
async def file_internal_write(upload: UploadFile, job_id: str, file_name: Optional[str] = None,
                              auth: str = Header(None)):
    # Even though it's validated by an auth token, we should still check for path injections
    if '..' in file_name or '..' in job_id:
        raise HTTPException(400)

    path = internal_validate(job_id, file_name, auth)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(await upload.read())
    return 'success'


@app.get("/create_job")
def create_job(job_id: str = "", submitter: str = "", job_name: str = "", submitter_email: str = ""):
    """
    Create a job

    :param job_id: Job ID
    :param submitter: Submitter name
    :param job_name: Job name
    :param submitter_email: Submitter email

    :return: Job ID
    """
    if not job_id:
        job_id = str(uuid.uuid4())
    redis_client.hmset(job_id, {'job_id': job_id, 'submitter': submitter, 'job_name': job_name, 'submitter_email': submitter_email,
                                'status': 0, 'time_created': str(datetime.datetime.now())})
    return job_id


@app.get("/get_job_status")
def job_status(job_id: str):
    """
    Get job status

    :param job_id: Job ID
    :return: Job status
    """
    if not redis_client.exists(job_id):
        raise HTTPException(404)
    return redis_client.hgetall(job_id)


@app.get("/get_all_jobs")
def get_all_jobs():
    """
    Get all jobs

    :return: List of jobs
    """
    keys = redis_client.keys()
    return [redis_client.hgetall(key) for key in keys]


@app.get("/run_prediction")
async def run_prediction(job_id: str):
    """
    Run prediction

    :param job_id: Job ID
    :return: Job ID
    """
    print("Running prediction")
    job = worker.predict.delay(job_id, 0)
    return {'job_id': job.id}


@app.get("/get_result")
def get_result(job_id: str):
    """
    Get prediction result

    :param job_id: Job ID
    :return: Prediction result
    """
    import pandas as pd
    if os.path.exists(f"results/{job_id}/predictions.csv"):
        pairs = pd.read_csv(f"results/{job_id}/predictions.csv")
        df = pd.DataFrame(pairs)
        df.columns = ['sub', 'seq', 'val']
        return df.to_json(orient='records')


@app.get("/get_heatmap")
def get_heatmap(job_id: str):
    """
    Get heatmap

    :param job_id: Job ID
    :return: Heatmap
    """
    if os.path.exists(f"results/{job_id}/heatmap.svg"):
        # if exists, return the heatmap image
        return FileResponse(f"results/{job_id}/heatmap.svg")



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=4000)