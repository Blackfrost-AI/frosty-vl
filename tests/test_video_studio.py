import base64
import importlib.util
import io
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
import pytest
from PIL import Image
from server.video_library import VideoLibrary
from tests_video_helpers import wait_for_job


@pytest.fixture
def video_studio(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('tested_video_ui',Path(__file__).resolve().parents[1]/'ui/webui.py')
    ui=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    monkeypatch.setattr(ui,'GALLERY_DIR',tmp_path)
    engines=[dict(id='image',label='Image',description='Image fixture',base='http://image.invalid',capabilities=['text_to_image'],controls={}),
             dict(id='video',label='Video',description='Video fixture',base='http://video.invalid',
                  capabilities=['text_to_video','image_to_video','same_face','scene_lab'],
                  controls={'durations':[5,8],'steps':{'min':4,'max':60,'default':30},'resolutions':['480p'],'aspect_ratios':['16:9'],'guidance':{'min':1,'max':10,'default':5},'negative_prompt':True})]
    monkeypatch.setattr(ui,'ENGINES',engines)
    monkeypatch.setattr(ui,'ENGINES_BY_ID',{e['id']:e for e in engines})
    monkeypatch.setattr(ui,'DEFAULT_ENGINE_ID','image')
    calls=[]
    def upstream(method,path,payload=None,timeout=None,base=None):
        calls.append((method,path,payload,base))
        if path=='/health':return 200,json.dumps({'ready':True}).encode()
        if path=='/generate':
            name='clip_'+str(len(calls))+'.mp4'
            (tmp_path/name).write_bytes(b'fixture-video-content'*100)
            return 200,json.dumps({'filename':name,'mode':'video','duration_seconds':5}).encode()
        raise AssertionError(path)
    monkeypatch.setattr(ui,'http_json',upstream)
    server=ui.ThreadingHTTPServer(('127.0.0.1',0),ui.Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    def api(path,payload=None):
        request=urllib.request.Request(base+path,data=json.dumps(payload).encode() if payload is not None else None,
                                       headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request) as response:
            return response.status,json.load(response)
    yield ui,api,base,calls
    server.shutdown();server.server_close();thread.join()


def test_video_job_uses_video_engine_with_image_default_and_reuses_request_id(video_studio):
    ui,api,base,calls=video_studio
    body={'prompt':'A small red boat','seed':None,'request_id':'one'}
    _,job=api('/api/videos/jobs',body)
    result=wait_for_job(api,job['id'])
    assert result['status']=='done' and result['engine_id']=='video'
    assert result['used_seed'] is not None
    assert api('/api/videos/jobs',body)[1]['id']==job['id']
    renders=[c for c in calls if c[1]=='/generate']
    assert len(renders)==1 and renders[0][3]=='http://video.invalid'
    assert renders[0][2]['prompt']=='A small red boat'
    assert len(api('/api/videos/jobs')[1]['items'])==1
    assert urllib.request.urlopen(base+result['file_url']).read().startswith(b'fixture-video-content')


@pytest.mark.parametrize('payload',[
    {'prompt':'x','engine_id':'image'}, {'prompt':'x','engine_id':'missing'},
    {'prompt':'x','duration_seconds':9}, {'prompt':'x','num_inference_steps':999},
    {'prompt':'x','guidance_scale':float('nan')}, {'prompt':'x','seed':'invalid'},
    {'prompt':'x','image_b64':'not an image'}, {'prompt':'x','identity_lock':True},
    {'prompt':'x','resolution':'9999p'}, {'prompt':'x','unknown_setting':1},
    {'kind':'scenes','scenes':[{'prompt':'one','duration':5}]},
    {'prompt_format':'json','prompt':'[]'},
])
def test_invalid_jobs_never_reach_backend(video_studio,payload):
    _,api,_,calls=video_studio
    with pytest.raises(urllib.error.HTTPError) as error:api('/api/videos/jobs',payload)
    assert error.value.code==400
    assert not calls


def test_video_trash_restore_pair_and_all_file_routes(video_studio):
    ui,api,base,_=video_studio
    _,job=api('/api/videos/jobs',{'prompt':'Keep this prompt','seed':42})
    wait_for_job(api,job['id'])
    item=api('/api/videos/gallery')[1]['items'][0]
    original=(ui.GALLERY_DIR/item['name']).read_bytes()
    metadata=(ui.GALLERY_DIR/(item['name']+'.json')).read_bytes()
    assert item['prompt']=='Keep this prompt' and item['seed']==42 and item['id'].startswith('video_')
    result=api('/api/videos/gallery/trash',{'ids':[item['id']]})[1]
    token=result['results'][0]['trash_id']
    assert not api('/api/videos/gallery')[1]['items']
    for route in [item['file_url'],'/api/gallery/file?name='+item['name'],'/api/file?engine=video&name='+item['name']]:
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(base+route)
        assert error.value.code==404
    assert api('/api/videos/gallery/trash')[1]['items'][0]['id']==token
    assert api('/api/videos/gallery/restore',{'ids':[token]})[1]['ok']
    assert (ui.GALLERY_DIR/item['name']).read_bytes()==original
    assert (ui.GALLERY_DIR/(item['name']+'.json')).read_bytes()==metadata
    request=urllib.request.Request(base+item['file_url'],headers={'Range':'bytes=1-9'})
    with urllib.request.urlopen(request) as response:
        assert response.status==206 and response.read()==original[1:10]


def test_scene_cancel_keeps_first_clip_and_never_starts_next_scene(video_studio,monkeypatch):
    ui,api,_,calls=video_studio
    entered,release=threading.Event(),threading.Event()
    original=ui._remote_generate
    def slow(*args):
        entered.set();assert release.wait(3);return original(*args)
    monkeypatch.setattr(ui,'_remote_generate',slow)
    _,job=api('/api/videos/jobs',{'kind':'scenes','scenes':[{'prompt':'one','duration':5},{'prompt':'two','duration':5}]})
    assert entered.wait(2)
    assert api('/api/videos/jobs/'+job['id']+'/cancel',{})[1]['status']=='cancelling'
    release.set()
    result=wait_for_job(api,job['id'])
    assert result['status']=='cancelled' and result['completed_scenes']==1
    assert len(result['scene_urls'])==1 and len(api('/api/videos/gallery')[1]['items'])==1
    assert len([c for c in calls if c[1]=='/generate'])==1


def test_offline_backend_is_a_failed_job_not_a_fake_result(video_studio,monkeypatch):
    ui,api,_,_=video_studio
    def unavailable(*args,**kwargs):raise OSError('offline fixture')
    monkeypatch.setattr(ui,'http_json',unavailable)
    _,job=api('/api/videos/jobs',{'prompt':'test'})
    result=wait_for_job(api,job['id'])
    assert result['status']=='error' and 'offline fixture' in result['error']
    assert 'file_url' not in result


def test_video_library_recovery_conflict_and_media_isolation(tmp_path,monkeypatch):
    (tmp_path/'image.png').write_bytes(b'image unchanged')
    (tmp_path/'video.mp4').write_bytes(b'video unchanged')
    (tmp_path/'video.mp4.json').write_text('{"prompt":"saved"}')
    library=VideoLibrary(tmp_path)
    item=library.gallery()['items'][0]
    rename=Path.rename
    def fail(source,target):
        if source.name=='video.mp4':raise PermissionError('temporary file lock')
        return rename(source,target)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'rename',fail)
        with pytest.raises(PermissionError):library.move_to_trash(item['id'])
    restarted=VideoLibrary(tmp_path)
    assert restarted.gallery()['count']==0
    token=restarted.trash_items()['items'][0]['id']
    (tmp_path/'video.mp4').write_bytes(b'new version')
    with pytest.raises(FileExistsError):restarted.restore(token)
    assert (tmp_path/'video.mp4').read_bytes()==b'new version'
    assert (tmp_path/'image.png').read_bytes()==b'image unchanged'


def test_scene_continuity_and_movie_assembly_with_real_ffmpeg(video_studio,monkeypatch,tmp_path):
    import shutil
    import subprocess
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:pytest.skip('ffmpeg is required for the media assembly check')
    ui,api,base,calls=video_studio
    clip=tmp_path/'synthetic-source.mp4'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-f','lavfi','-i',
        'testsrc2=size=160x96:rate=24','-t','0.5','-c:v','libx264','-pix_fmt','yuv420p',str(clip)],check=True)
    source=clip.read_bytes();clip.unlink()
    monkeypatch.setattr(ui,'FFMPEG',ffmpeg)
    original=ui.http_json
    def upstream(method,path,payload=None,timeout=None,base=None):
        status,body=original(method,path,payload,timeout,base)
        if path=='/generate':(tmp_path/json.loads(body)['filename']).write_bytes(source)
        return status,body
    monkeypatch.setattr(ui,'http_json',upstream)
    monkeypatch.setattr(ui,'http_bytes',lambda path,**kwargs:(200,'video/mp4',(tmp_path/Path(path).name).read_bytes()))
    _,job=api('/api/videos/jobs',{'kind':'scenes','seed':12,'story':'A test pattern',
        'scenes':[{'prompt':'first','duration':5},{'prompt':'second','duration':8}]})
    result=wait_for_job(api,job['id'])
    assert result['status']=='done',result
    rendered=[call[2] for call in calls if call[1]=='/generate']
    assert 'image_b64' not in rendered[0]
    assert rendered[1]['image_b64'].startswith('data:image/png;base64,')
    assert rendered[1]['seed']==13 and rendered[1]['duration_seconds']==8
    image=Image.open(io.BytesIO(base64.b64decode(rendered[1]['image_b64'].split(',')[1])))
    assert image.size==(160,96)
    movie=urllib.request.urlopen(base+result['file_url']).read()
    assert b'ftyp' in movie[:20] and len(result['scene_urls'])==2
    gallery=api('/api/videos/gallery')[1]['items']
    assert len(gallery)==3
    assembled=next(item for item in gallery if item['mode']=='scene lab')
    assert len(assembled['scenes'])==2 and assembled['seed']==12
