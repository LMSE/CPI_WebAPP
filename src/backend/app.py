from flask import Flask, flash, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
import os

from utils.job_management import generate_job_id
import pipeline.data_process_new

# np imports
import numpy as np
import pandas as pd

UPLOAD_FOLDER = 'jobs'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "test"


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/preprocess/load_seq", methods=['POST'])
def load_seq():
    """
    Upload a csv file containing sequences to the server for preprocessing
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            abort(400, "No file part")
        file = request.files['file']
        if file:
            filename = generate_job_id('seq')
            upload_folder = os.path.join(app.instance_path, app.config["UPLOAD_FOLDER"])
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            file.save(os.path.join(upload_folder, filename))
            return filename


@app.route("/preprocess/load_sub", methods=['POST'])
def load_sub():
    """
    Upload a csv file containing substrates to the server for preprocessing
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            abort(400, "No file part")
        file = request.files['file']
        if file:
            filename = generate_job_id('sub')
            upload_folder = os.path.join(app.instance_path, app.config["UPLOAD_FOLDER"])
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            file.save(os.path.join(upload_folder, filename))
            return filename


@app.route("/preprocess/seq_stat", methods=['GET'])
def seq_stat():
    """
    Given a uuid associated with an uploaded sequence file, return
    basic statistics about the sequence file
    """
    if request.method == 'GET':
        if 'filename' not in request.args:
            abort(400, "No filename")
        filename = request.args['filename']
        if not os.path.exists(os.path.join(app.instance_path, app.config["UPLOAD_FOLDER"], filename)):
            abort(404, "File not found")
        full_path = os.path.join(app.instance_path, app.config["UPLOAD_FOLDER"], filename)
        full_processed_path = os.path.join(app.instance_path, app.config["UPLOAD_FOLDER"], filename + ".processed")
        arr = pipeline.data_process_new.read_seqs(full_path, full_processed_path)
        stat = {
            "num_seq": arr.size,
            "max_len": int(arr.map(len).max()),
        }
        return stat
