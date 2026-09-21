import threading
import time
import pytest
from ui.video_jobs import VideoJobs, QueueFull


def until(check):
    deadline = time.monotonic()+3
    while time.monotonic()<deadline:
        result = check()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError('Condition did not become true')


def test_fifo_cancellation_and_completed_output_retained():
    entered, release = threading.Event(), threading.Event()
    calls=[]
    def render(request, report, cancelled):
        calls.append(request['prompt'])
        entered.set()
        assert release.wait(3)
        return {'file_url':'/api/videos/files/one.mp4'}
    queue = VideoJobs(render)
    first=queue.submit({'engine_id':'v','prompt':'first'})
    assert entered.wait(2)
    second=queue.submit({'engine_id':'v','prompt':'second'})
    third=queue.submit({'engine_id':'v','prompt':'third'})
    assert queue.cancel(second['id'])['status']=='cancelled'
    assert queue.cancel(first['id'])['status']=='cancelling'
    release.set()
    until(lambda:queue.get(third['id'])['status']=='done')
    assert calls==['first','third']
    result=queue.get(first['id'])
    assert result['status']=='cancelled' and result['file_url'].endswith('one.mp4')
    assert queue.cancel(third['id'])['status']=='done'


def test_idempotency_queue_limit_and_failure_isolation():
    release=threading.Event()
    def render(request, *_):
        assert release.wait(3)
        if request['prompt']=='bad':
            raise ValueError('engine offline')
        return {}
    queue=VideoJobs(render,pending_limit=2)
    one=queue.submit({'engine_id':'v','prompt':'bad','request_id':'unique'})
    assert queue.submit({'engine_id':'v','prompt':'bad','request_id':'unique'})['id']==one['id']
    with pytest.raises(ValueError,match='different settings'):
        queue.submit({'engine_id':'v','prompt':'changed','request_id':'unique'})
    two=queue.submit({'engine_id':'v','prompt':'good'})
    with pytest.raises(QueueFull):
        queue.submit({'engine_id':'v','prompt':'overflow'})
    release.set()
    until(lambda:queue.get(two['id'])['status']=='done')
    assert queue.get(one['id'])['status']=='error'
    assert 'offline' in queue.get(one['id'])['error']


def test_cancelled_queue_entries_release_payloads_and_history_is_bounded():
    entered,release=threading.Event(),threading.Event()
    def render(*_):
        entered.set()
        release.wait(3)
        return {}
    queue=VideoJobs(render,pending_limit=2,history_limit=5)
    one=queue.submit({'engine_id':'v','prompt':'first'})
    assert entered.wait(2)
    for _ in range(20):
        item=queue.submit({'engine_id':'v','prompt':'cancel me'})
        queue.cancel(item['id'])
    assert len(queue.queues['v'])==0
    assert len(queue.jobs)<=5
    release.set()
    until(lambda:queue.get(one['id'])['status']=='done')
