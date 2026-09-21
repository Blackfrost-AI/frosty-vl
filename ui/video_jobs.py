"""Small, bounded FIFO video queue. No model imports and no GPU preemption."""
from collections import deque
from copy import deepcopy
import hashlib
import json
import secrets
import threading
import time


class QueueFull(ValueError):
    pass


class Cancelled(Exception):
    pass


class VideoJobs:
    def __init__(self, runner, pending_limit=8, history_limit=100):
        self.runner = runner
        self.pending_limit = pending_limit
        self.history_limit = history_limit
        self.lock = threading.RLock()
        self.jobs = {}
        self.queues = {}
        self.workers = set()
        self.keys = {}

    def submit(self, request):
        request = deepcopy(request)
        engine = request['engine_id']
        key = request.pop('request_id', None)
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key and key in self.keys:
                job_id, previous = self.keys[key]
                if fingerprint != previous:
                    raise ValueError('request_id was already used with different settings')
                return self.get(job_id)
            active = sum(j['status'] not in {'done', 'error', 'cancelled'} for j in self.jobs.values())
            if active >= self.pending_limit:
                raise QueueFull('Video queue is full; wait for a job to finish')
            for job_id in list(self.jobs):
                if len(self.jobs) < self.history_limit:
                    break
                if self.jobs[job_id]['status'] in {'done', 'error', 'cancelled'}:
                    del self.jobs[job_id]
                    self.keys = {k: v for k, v in self.keys.items() if v[0] != job_id}
            job_id = 'vid_' + secrets.token_hex(12)
            self.jobs[job_id] = dict(id=job_id, job=job_id, status='queued', stage='Queued',
                engine_id=engine, kind=request.get('kind', 'clip'), created_at=time.time(),
                updated_at=time.time(), cancel_requested=False,
                cancellation='Queued jobs stop immediately; a running render finishes before stopping.',
                prompt=request.get('prompt') or request.get('story', ''),
                total_scenes=len(request.get('scenes', [])), completed_scenes=0, scene_urls=[])
            if key:
                self.keys[key] = (job_id, fingerprint)
            self.queues.setdefault(engine, deque()).append((job_id, request))
            if engine not in self.workers:
                self.workers.add(engine)
                threading.Thread(target=self._run, args=(engine,), daemon=True).start()
            return self.get(job_id)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError('Unknown video job; Studio restart clears job history')
            job = deepcopy(self.jobs[job_id])
            job['ok'] = job['status'] != 'error'
            return job

    def list(self):
        with self.lock:
            return [self.get(key) for key in reversed(self.jobs)]

    def cancel(self, job_id):
        with self.lock:
            self.get(job_id)
            job = self.jobs[job_id]
            if job['status'] not in {'done', 'error', 'cancelled'}:
                queued = job['status'] == 'queued'
                job.update(cancel_requested=True, updated_at=time.time(),
                           status='cancelled' if queued else 'cancelling',
                           stage='Cancelled before rendering' if queued else 'Stopping after the current render')
                if queued:
                    engine = job['engine_id']
                    self.queues[engine] = deque((i, r) for i, r in self.queues[engine] if i != job_id)
            return self.get(job_id)

    def _run(self, engine):
        while True:
            with self.lock:
                if not self.queues[engine]:
                    self.workers.remove(engine)
                    return
                job_id, request = self.queues[engine].popleft()
                job = self.jobs[job_id]
                if job['cancel_requested']:
                    continue
                job.update(status='running', stage='Preparing render', started_at=time.time())
            def cancelled():
                with self.lock:
                    return self.jobs[job_id]['cancel_requested']
            def report(**fields):
                with self.lock:
                    current = self.jobs[job_id]
                    if current['cancel_requested']:
                        fields.pop('stage', None)
                    current.update(fields, updated_at=time.time())
            started = time.time()
            try:
                result = self.runner(request, report, cancelled)
                with self.lock:
                    job.update(result, status='cancelled' if cancelled() else 'done',
                               stage='Stopped; completed clips were kept' if cancelled() else 'Complete')
            except Cancelled:
                with self.lock:
                    job.update(status='cancelled', stage='Stopped; completed clips were kept')
            except Exception as exc:
                with self.lock:
                    job.update(status='error', stage='Render failed', error=str(exc))
            finally:
                with self.lock:
                    job.update(seconds=round(time.time() - started, 1), finished_at=time.time(), updated_at=time.time())
