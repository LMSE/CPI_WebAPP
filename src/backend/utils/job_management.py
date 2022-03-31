def generate_job_id(job_type):
    """
    Generates a job id.
    """
    import uuid
    return job_type + '-' + str(uuid.uuid4())