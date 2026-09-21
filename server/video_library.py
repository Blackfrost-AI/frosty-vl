"""Persistent video gallery using the same recoverable journal as Image Studio."""
import os
import secrets
import shutil
from .image_library import ImageLibrary, _write


class VideoLibrary(ImageLibrary):
    media_types = {'.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime',
                   '.m4v': 'video/mp4', '.gif': 'image/gif'}
    id_prefix = 'video'
    trash_name = '.frosty-video-trash'
    file_prefix = '/api/videos/files/'
    metadata_fields = ('prompt', 'prompt_format', 'seed', 'mode', 'duration_seconds',
                       'scenes', 'engine_id', 'engine_label', 'resolution', 'aspect_ratio',
                       'num_inference_steps', 'guidance_scale', 'negative_prompt', 'width', 'height')

    def metadata(self, name, values):
        with self.lock:
            path = self.file(name)
            _write(self._path(path.name + '.json'),
                   {k: v for k, v in values.items() if k in self.metadata_fields})
            return self._item(path)

    def publish_file(self, source, name, metadata):
        with self.lock:
            target = self._path(self._name(name))
            sidecar = self._path(name + '.json')
            if target.exists() or sidecar.exists():
                raise FileExistsError('Output name already exists')
            temporary = self._path(name + '.partial-' + secrets.token_hex(6))
            try:
                with open(source, 'rb') as src, temporary.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
                _write(sidecar, {k: v for k, v in metadata.items() if k in self.metadata_fields})
                temporary.rename(target)
            finally:
                temporary.unlink(missing_ok=True)
            return self._item(target)
