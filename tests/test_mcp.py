import asyncio
import base64
import io
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest
from PIL import Image

# Optional companion dependencies should not prevent the render-only test suite.
pytest.importorskip('mcp')
from importlib.metadata import version
from packaging.version import Version
if Version(version('mcp')) < Version('2.2.0'):
    pytest.skip('Install requirements-mcp.txt for companion tests', allow_module_level=True)
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from frosty_mcp.client import StudioClient
from frosty_mcp.models import ImageSpec
from frosty_mcp.server import create_server
from frosty_mcp.__main__ import http_app
from server.image_library import ImageLibrary
from test_video_studio import video_studio

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def studio_bridge(video_studio,tmp_path,monkeypatch):
    ui,api,url,calls=video_studio
    library=ImageLibrary(tmp_path/'images')
    item=library.publish(Image.new('RGBA',(24,24),(50,100,180,255)),'test.png',{'prompt':'blue fixture'})
    upstream=ui.http_json
    image_calls=[]
    def image_api(method,path,payload=None,timeout=None,base=None):
        if base!='http://image.invalid':return upstream(method,path,payload,timeout,base)
        image_calls.append((method,path,payload))
        if path=='/health':data={'ready':True}
        elif path=='/gallery':data=library.gallery()
        elif path=='/gallery/trash' and method=='GET':data=library.trash_items()
        elif path in {'/gallery/trash','/gallery/restore'}:
            action=library.restore if path.endswith('restore') else library.move_to_trash
            results=[]
            for identifier in payload['ids']:
                try:
                    entry=action(identifier);results.append({'ok':True,'id':identifier,'trash_id':entry['id']})
                except (ValueError,OSError) as exc:results.append({'ok':False,'id':identifier,'error':str(exc)})
            data={'ok':all(r['ok'] for r in results),'results':results}
        elif method=='POST' and path in {'/jobs','/enhance'}:
            data={'id':'img_'+'a'*24,'status':'queued'}
        elif path.endswith('/cancel'):data={'id':'img_'+'a'*24,'status':'cancelled'}
        elif path.startswith('/jobs/'):
            data={'id':'img_'+'a'*24,'status':'done','outputs':[{'name':'test.png','file_url':'/files/test.png'}]}
        else:raise AssertionError(path)
        return 200,json.dumps(data).encode()
    monkeypatch.setattr(ui,'http_json',image_api)
    monkeypatch.setattr(ui,'http_bytes',lambda path,**kwargs:(200,'image/png',library.file(Path(path).name).read_bytes()))
    return StudioClient(url),item,image_calls


def test_protocol_image_reference_tools_and_recoverable_gallery(studio_bridge):
    bridge,item,calls=studio_bridge
    async def run():
        async with Client(create_server(bridge)) as client:
            tools={tool.name:tool for tool in (await client.list_tools()).tools}
            assert len(tools)==12
            assert tools['frosty_trash'].annotations.destructive_hint
            assert tools['frosty_status'].annotations.read_only_hint
            assert not tools['frosty_generate_image'].annotations.idempotent_hint
            resource=await client.read_resource('frosty://guide')
            assert 'already finished clips are retained' in resource.contents[0].text
            status=await client.call_tool('frosty_status')
            assert status.structured_content['workspaces']=={'image':True,'video':True}
            spec={'prompt':'compose','images':[item['id']]*10,'n':2,'width':512,'height':512,'dwm_scale':0.75,
                  'reference_resolution':256,'use_kv_cache':False,'num_inference_steps':20}
            result=await client.call_tool('frosty_generate_image',{'request':spec})
            assert not result.is_error
            payload=calls[-1][2]
            assert len(payload['images_b64'])==10 and payload['dwm_scale']==.75 and payload['n']==2
            assert payload['images_b64'][0].startswith('data:image/png;base64,')
            too_many=await client.call_tool('frosty_generate_image',{'request':dict(spec,images=[item['id']]*11)})
            assert too_many.is_error
            # Omitting DWM keeps the user's backend default, rather than sending zero.
            await client.call_tool('frosty_enhance_prompt',{'request':{'prompt':'expand this'}})
            assert calls[-1][1]=='/enhance' and 'dwm_scale' not in calls[-1][2]
            result=await client.call_tool('frosty_job',{'workspace':'image','job_id':'img_'+'a'*24})
            assert result.structured_content['outputs'][0]['file_url']==bridge.url+'/api/images/files/test.png'
            preview=await client.call_tool('frosty_preview_image',{'asset_id':item['id']})
            assert preview.content[0].type=='image'
            assert Image.open(io.BytesIO(base64.b64decode(preview.content[0].data))).size==(24,24)
            asset=await client.call_tool('frosty_asset',{'workspace':'image','asset_id':item['id']})
            assert asset.content[1].type=='resource_link'
            trash=await client.call_tool('frosty_trash',{'workspace':'image','asset_ids':[item['id']]})
            token=trash.structured_content['results'][0]['trash_id']
            assert (await client.call_tool('frosty_asset',{'workspace':'image','asset_id':item['id']})).is_error
            restored=await client.call_tool('frosty_restore',{'workspace':'image','trash_ids':[token]})
            assert not restored.is_error
            gallery=await client.call_tool('frosty_gallery',{'workspace':'image','search':'blue','limit':1})
            assert gallery.structured_content['total']==1
    asyncio.run(run())


def test_real_stdio_legacy_client_and_video_lifecycle(video_studio):
    _,_,url,_=video_studio
    async def run():
        env=dict(os.environ,FROSTY_STUDIO_URL=url)
        params=StdioServerParameters(command=sys.executable,args=['-m','frosty_mcp'],cwd=ROOT,env=env)
        async with Client(params,mode='legacy') as client:
            result=await client.call_tool('frosty_generate_video',{'request':{'prompt':'A sailing boat','request_id':'mcp-test'}})
            assert not result.is_error
            job_id=result.structured_content['id']
            for _ in range(100):
                job=await client.call_tool('frosty_job',{'workspace':'video','job_id':job_id})
                data=job.structured_content
                if data['status']=='done':break
                await asyncio.sleep(.02)
            assert data['status']=='done' and data['file_url'].startswith(url+'/api/')
            gallery=await client.call_tool('frosty_gallery',{'workspace':'video'})
            assert len(gallery.structured_content['items'])==1
            bad=await client.call_tool('frosty_job',{'workspace':'image','job_id':job_id})
            assert bad.is_error
            duplicate=await client.call_tool('frosty_generate_video',{'request':{'prompt':'A sailing boat','request_id':'mcp-test'}})
            assert duplicate.structured_content['id']==job_id
    asyncio.run(run())


def test_streamable_http_authentication_and_real_client(studio_bridge):
    import httpx2
    import uvicorn
    bridge,_,_=studio_bridge
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    token='fixture-token-with-at-least-24-chars'
    app=http_app(create_server(bridge),'127.0.0.1',port,token)
    server=uvicorn.Server(uvicorn.Config(app,log_level='critical'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    try:
        deadline=time.monotonic()+5
        while not server.started and time.monotonic()<deadline:time.sleep(.01)
        assert server.started
        url=f'http://127.0.0.1:{port}/mcp'
        for headers in [{},{'Authorization':'Bearer wrong-token'}]:
            with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(urllib.request.Request(url,headers=headers))
            assert error.value.code==401
        request=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Host':'unrelated.invalid'})
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
        assert error.value.code in {400,421}
        async def run():
            async with httpx2.AsyncClient(headers={'Authorization':'Bearer '+token}) as http:
                async with Client(streamable_http_client(url,http_client=http)) as client:
                    result=await client.call_tool('frosty_status')
                    assert not result.is_error and result.structured_content['workspaces']['video']
        asyncio.run(run())
    finally:
        server.should_exit=True;thread.join(timeout=5);sock.close()


def test_input_directories_and_reference_boundaries(tmp_path):
    allowed=tmp_path/'allowed';allowed.mkdir()
    inside=allowed/'ok.png';outside=tmp_path/'outside.png'
    Image.new('RGB',(8,8)).save(inside);Image.new('RGB',(8,8)).save(outside)
    async def run():
        with pytest.raises(ValueError,match='disabled'):await StudioClient().image(str(inside))
        client=StudioClient(input_dirs=[allowed])
        assert (await client.image(str(inside))).startswith('data:image/png;base64,')
        with pytest.raises(ValueError,match='outside'):await client.image(str(outside))
        link=allowed/'redirect.png'
        try:link.symlink_to(outside)
        except OSError:pass
        else:
            with pytest.raises(ValueError,match='outside'):await client.image(str(link))
        with pytest.raises(ValueError):await client.image('data:image/png;base64,bad')
    asyncio.run(run())


def test_network_listener_requires_auth_and_specific_private_bind():
    server=create_server()
    with pytest.raises(ValueError,match='requires'):http_app(server,'10.0.0.2',8891)
    with pytest.raises(ValueError,match='specific'):http_app(server,'0.0.0.0',8891,'x'*32)
    with pytest.raises(ValueError,match='specific'):http_app(server,'8.8.8.8',8891,'x'*32)
