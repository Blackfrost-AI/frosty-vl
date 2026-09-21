"""Bounded HTTP calls to one operator-configured Studio; never a general URL proxy."""
import asyncio
import base64
import io
import json
from pathlib import Path
import re
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit
from PIL import Image


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Studio redirects are not followed; configure its final URL')


class StudioClient:
    def __init__(self, url='http://127.0.0.1:8890', input_dirs=(), token=None):
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Studio URL must be an HTTP(S) origin or base path without embedded credentials')
        self.url = url.rstrip('/')
        self.input_dirs = tuple(Path(p).expanduser().resolve(strict=True) for p in input_dirs)
        if any(not p.is_dir() for p in self.input_dirs):
            raise ValueError('Every configured input directory must be a directory')
        self.token = token

    def _read(self, path, payload=None, binary=False):
        if not path.startswith('/api/') or path.startswith('//'):
            raise ValueError('Expected a Studio API path')
        data = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        if data and len(data)>36_000_000:
            raise ValueError('Combined request exceeds 36 MB')
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        request = urllib.request.Request(self.url+path, data=data, headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        limit = 12_000_000 if binary else 4_000_000
        try:
            with opener.open(request, timeout=30) as response:
                raw = response.read(limit+1)
                if len(raw)>limit:
                    raise ValueError('Studio response exceeds the transfer limit')
                if binary:
                    return raw
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ValueError('Studio returned an unexpected response')
                return result
        except urllib.error.HTTPError as exc:
            detail = exc.read(4000).decode(errors='replace')
            raise ValueError(f'Studio returned HTTP {exc.code}: {detail}') from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ValueError('Studio is unavailable; check its URL and readiness') from exc

    async def request(self, path, payload=None):
        return await asyncio.to_thread(self._read, path, payload)

    def output_links(self, workspace, data):
        """Translate engine-relative results into links reachable through Studio."""
        def link(value):
            if workspace=='image' and value.startswith('/files/'):
                value = '/api/images/files/' + value.removeprefix('/files/')
            if not value.startswith('/api/'):
                raise ValueError('Studio returned an unexpected output URL')
            return self.url + value
        if isinstance(data, list):
            return [self.output_links(workspace, item) for item in data]
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key=='file_url' and value:
                    result[key] = link(value)
                elif key=='scene_urls':
                    result[key] = [link(v) for v in value]
                else:
                    result[key] = self.output_links(workspace, value)
            return result
        return data

    @staticmethod
    def prefix(workspace):
        if workspace not in {'image', 'video'}:
            raise ValueError('Workspace must be image or video')
        return '/api/images' if workspace=='image' else '/api/videos'

    @staticmethod
    def job_id(workspace, value):
        prefix = 'img' if workspace=='image' else 'vid'
        if not re.fullmatch(prefix+r'_[a-f0-9]{24}', value):
            raise ValueError('Job ID does not belong to this workspace')
        return value

    async def asset(self, workspace, identifier):
        if not re.fullmatch(workspace+r'_[a-f0-9]{32}', identifier):
            raise ValueError('Invalid gallery asset ID')
        gallery = await self.request(self.prefix(workspace)+'/gallery')
        item = next((x for x in gallery.get('items', []) if x.get('id')==identifier), None)
        if not item:
            raise ValueError('Asset is unavailable or in Trash')
        name = item.get('name', '')
        if not name or any(c in name for c in '/\\:\x00') or name.rstrip(' .')!=name:
            raise ValueError('Studio returned an invalid asset name')
        path = self.prefix(workspace)+'/files/'+quote(name, safe='')
        return dict(item, file_url=self.url+path, api_path=path)

    @staticmethod
    def encode_image(raw):
        if len(raw)>9_000_000:
            raise ValueError('Reference image exceeds 9 MB; resize it first')
        try:
            with Image.open(io.BytesIO(raw)) as image:
                fmt = image.format
                if fmt not in {'PNG', 'JPEG', 'WEBP'} or image.width*image.height>16_000_000:
                    raise ValueError('Use PNG/JPEG/WebP images up to 16 megapixels')
                image.verify()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError('Invalid reference image') from exc
        mime = {'PNG':'image/png','JPEG':'image/jpeg','WEBP':'image/webp'}[fmt]
        return 'data:'+mime+';base64,'+base64.b64encode(raw).decode()

    async def image(self, value):
        if value.startswith('image_'):
            item = await self.asset('image', value)
            raw = await asyncio.to_thread(self._read, item['api_path'], None, True)
        elif value.startswith('data:image/'):
            if len(value)>12_000_000 or ';base64,' not in value:
                raise ValueError('Invalid or oversized image data URL')
            try:
                raw = base64.b64decode(value.split(',',1)[1], validate=True)
            except ValueError as exc:
                raise ValueError('Invalid image base64') from exc
        else:
            if not self.input_dirs:
                raise ValueError('Local file inputs are disabled. Set FROSTY_MCP_INPUT_DIRS or use a gallery ID/data URL.')
            path = Path(value).expanduser().resolve(strict=True)
            if not any(path.is_relative_to(root) for root in self.input_dirs):
                raise ValueError('Image is outside configured input directories')
            if not path.is_file() or path.stat().st_size>9_000_000:
                raise ValueError('Expected an image file up to 9 MB')
            with path.open('rb') as stream:
                raw = stream.read(9_000_001)
        return self.encode_image(raw)

    async def image_payload(self, request):
        values = request.model_dump(exclude_none=True)
        images = values.pop('images')
        mask = values.pop('mask', None)
        if mask and len(images)>9:
            raise ValueError('A mask allows at most nine reference images')
        values['images_b64'] = [await self.image(value) for value in images]
        if mask:
            values['mask_b64'] = await self.image(mask)
        return values

    async def video_payload(self, request):
        values = request.model_dump(exclude_none=True)
        image = values.pop('image', None)
        if image:
            values['image_b64'] = await self.image(image)
        return values
