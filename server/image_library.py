"""Image publication and recoverable Trash; owned by the image-engine process.

The journal is written before moving either member of an image/sidecar pair.
Interrupted moves resume at startup; collisions are never overwritten. This
module deliberately has no purge operation and no GPU dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import threading
import time
from pathlib import Path
from urllib.parse import quote

TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def asset_id(name, prefix="image"):
    return prefix + "_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]


def _plain(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Linked files and Windows reparse points are not supported")
    return info


def _name(name, types=TYPES):
    if not isinstance(name, str) or not name or name in {".", ".."} or any(c in name for c in '/\\:\x00'):
        raise ValueError("Invalid image name")
    if name.rstrip(" .") != name or Path(name).suffix.lower() not in types:
        raise ValueError("Unsupported image name")
    return name


def _digest(path):
    _plain(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path, data):
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(6))
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ImageLibrary:
    media_types = TYPES
    id_prefix = "image"
    trash_name = ".frosty-trash"
    file_prefix = "/api/images/files/"
    metadata_fields = ("prompt", "effective_prompt", "seed", "mode", "width", "height",
                       "num_inference_steps", "dwm_scale", "reference_count")

    def _name(self, name):
        return _name(name, self.media_types)

    def _id(self, name):
        return asset_id(name, self.id_prefix)

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.trash = self.root / self.trash_name
        if self.trash.exists() or self.trash.is_symlink():
            _plain(self.trash)
        self.trash.mkdir(exist_ok=True)
        self.recovery_errors = []
        for entry in self._entries():
            if entry["state"] in {"moving", "restoring"}:
                try:
                    self._finish(entry)
                except (OSError, ValueError) as exc:
                    self.recovery_errors.append({"id": entry["id"], "error": str(exc)})

    def _path(self, name, parent=None):
        parent = parent or self.root
        _plain(parent)
        candidate = parent / name
        if candidate.exists() or candidate.is_symlink():
            if not stat.S_ISREG(_plain(candidate).st_mode):
                raise ValueError("Expected a regular file")
        if candidate.resolve().parent != parent:
            raise ValueError("File is outside image storage")
        return candidate

    def _folder(self, token):
        if not re.fullmatch(r"[a-f0-9]{32}", token):
            raise ValueError("Invalid Trash ID")
        _plain(self.trash)
        folder = self.trash / token
        if folder.exists() or folder.is_symlink():
            _plain(folder)
        if folder.resolve().parent != self.trash:
            raise ValueError("Invalid Trash location")
        return folder

    def _read(self, folder):
        path = self._path("entry.json", folder)
        entry = json.loads(path.read_text(encoding="utf-8"))
        name = self._name(entry["name"])
        if entry["id"] != folder.name or entry["asset_id"] != self._id(name):
            raise ValueError("Invalid Trash journal identity")
        if entry["state"] not in {"moving", "trashed", "restoring", "restored"}:
            raise ValueError("Invalid Trash journal state")
        if set(entry["files"]) not in ({name}, {name, name + ".json"}):
            raise ValueError("Invalid Trash file pair")
        if any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in entry["files"].values()):
            raise ValueError("Invalid Trash file digest")
        return entry

    def _entries(self):
        _plain(self.trash)
        rows = []
        for folder in self.trash.iterdir():
            if not re.fullmatch(r"[a-f0-9]{32}", folder.name):
                continue
            try:
                rows.append(self._read(self._folder(folder.name)))
            except (OSError, ValueError, KeyError, TypeError):
                continue  # Never act on a malformed or redirected journal.
        return rows

    def _hidden(self):
        return {entry["name"] for entry in self._entries() if entry["state"] != "restored"}

    def _finish(self, entry):
        folder = self._folder(entry["id"])
        restoring = entry["state"] == "restoring"
        source, destination = (folder, self.root) if restoring else (self.root, folder)
        # Sidecar first, image last: the visible image is the completion marker.
        names = sorted(entry["files"], key=lambda name: not name.endswith(".json"))
        for name in names:
            src, dst = self._path(name, source), self._path(name, destination)
            if src.exists() and dst.exists():
                raise FileExistsError("Restore/move conflict; existing files were kept")
            present = src if src.exists() else dst
            if not present.is_file() or _digest(present) != entry["files"][name]:
                raise ValueError("Image changed or is missing; files were kept for recovery")
        for name in names:
            src, dst = self._path(name, source), self._path(name, destination)
            if src.exists():
                # Single owning process + library lock; no overwrite by this app.
                if dst.exists():
                    raise FileExistsError("Destination already exists")
                src.rename(dst)
        entry["state"] = "restored" if restoring else "trashed"
        _write(self._path("entry.json", folder), entry)
        return entry

    def _item(self, path):
        info = _plain(path)
        metadata = {}
        try:
            sidecar = self._path(path.name + ".json")
            if sidecar.exists():
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    metadata = data
        except (OSError, ValueError):
            pass
        fields = {key: metadata.get(key) for key in self.metadata_fields}
        return dict(fields, id=self._id(path.name), name=path.name,
                    content_type=self.media_types[path.suffix.lower()], size=info.st_size,
                    created_at=info.st_mtime,
                    file_url=self.file_prefix + quote(path.name))

    def gallery(self):
        with self.lock:
            hidden = self._hidden()
            items = []
            for path in self.root.iterdir():
                if path.name in hidden or path.suffix.lower() not in self.media_types:
                    continue
                try:
                    self._name(path.name)
                    if stat.S_ISREG(_plain(path).st_mode):
                        items.append(self._item(path))
                except (OSError, ValueError):
                    continue
            items.sort(key=lambda item: (item["created_at"], item["name"]), reverse=True)
            return dict(ok=True, count=len(items), items=items,
                        trash_count=sum(e["state"] != "restored" for e in self._entries()))

    def file(self, name):
        with self.lock:
            name = self._name(name)
            if name in self._hidden():
                raise FileNotFoundError("Image is in Trash")
            path = self._path(name)
            if not path.is_file():
                raise FileNotFoundError("Image not found")
            return path

    def trash_items(self):
        with self.lock:
            items = []
            for entry in self._entries():
                if entry["state"] == "restored":
                    continue
                items.append({key: entry[key] for key in ("id", "asset_id", "name", "state", "deleted_at")})
            return dict(ok=True, items=sorted(items, key=lambda item: item["deleted_at"], reverse=True))

    def move_to_trash(self, identifier):
        with self.lock:
            if not re.fullmatch(self.id_prefix + r"_[a-f0-9]{32}", identifier):
                raise ValueError("Invalid image ID")
            previous = next((e for e in self._entries() if e["asset_id"] == identifier and e["state"] != "restored"), None)
            if previous:
                if previous["state"] == "restoring":
                    raise ValueError("Restore is pending; finish restoring this item first")
                return self._finish(previous) if previous["state"] == "moving" else previous
            item = next((i for i in self.gallery()["items"] if i["id"] == identifier), None)
            if not item:
                raise FileNotFoundError("Image not found")
            path = self.file(item["name"])
            files = {path.name: _digest(path)}
            sidecar = self._path(path.name + ".json")
            if sidecar.exists():
                files[sidecar.name] = _digest(sidecar)
            entry = dict(version=1, id=secrets.token_hex(16), asset_id=identifier, name=path.name,
                         deleted_at=time.time(), state="moving", files=files)
            folder = self._folder(entry["id"])
            folder.mkdir()
            _write(self._path("entry.json", folder), entry)
            return self._finish(entry)

    def restore(self, token):
        with self.lock:
            folder = self._folder(token)
            entry = self._read(folder)
            if entry["state"] == "restored":
                return entry
            entry["state"] = "restoring"
            _write(self._path("entry.json", folder), entry)
            return self._finish(entry)

    def publish(self, image, name, metadata):
        with self.lock:
            target = self._path(self._name(name))
            sidecar = self._path(name + ".json")
            if target.exists() or sidecar.exists():
                raise FileExistsError("Output name already exists")
            temporary = self._path(name + ".partial-" + secrets.token_hex(6))
            try:
                image.save(temporary, format="PNG")
                _write(sidecar, metadata)
                temporary.rename(target)
            finally:
                temporary.unlink(missing_ok=True)
            return self._item(target)
