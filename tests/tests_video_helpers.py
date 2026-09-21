import time


def wait_for_job(api,job_id):
    deadline=time.monotonic()+4
    while time.monotonic()<deadline:
        result=api('/api/videos/jobs/'+job_id)[1]
        if result['status'] in {'done','error','cancelled'}:
            return result
        time.sleep(.01)
    raise AssertionError('Video job did not finish')
