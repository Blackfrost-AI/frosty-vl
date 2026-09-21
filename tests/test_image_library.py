import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.image_library import ImageLibrary, asset_id


def sample(library, name="one.png"):
    return library.publish(Image.new("RGBA", (16, 16), (10, 20, 30, 40)), name,
                           {"prompt": "A blue square", "seed": 42, "dwm_scale": 0.5})


def test_trash_restore_pair_survives_restart(tmp_path):
    library = ImageLibrary(tmp_path)
    item = sample(library)
    before = {name: (tmp_path / name).read_bytes() for name in ["one.png", "one.png.json"]}
    moved = library.move_to_trash(item["id"])
    assert library.gallery()["items"] == []
    assert library.move_to_trash(item["id"])["id"] == moved["id"]
    with pytest.raises(FileNotFoundError):
        library.file("one.png")
    restarted = ImageLibrary(tmp_path)
    assert restarted.trash_items()["items"][0]["state"] == "trashed"
    assert restarted.restore(moved["id"])["state"] == "restored"
    assert restarted.restore(moved["id"])["state"] == "restored"
    assert restarted.gallery()["items"][0]["dwm_scale"] == 0.5
    for name, data in before.items():
        assert (tmp_path / name).read_bytes() == data


def test_legacy_without_sidecar_and_video_untouched(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "legacy.jpg")
    (tmp_path / "clip.mp4").write_bytes(b"video")
    library = ImageLibrary(tmp_path)
    assert library.gallery()["count"] == 1
    entry = library.move_to_trash(asset_id("legacy.jpg"))
    library.restore(entry["id"])
    assert (tmp_path / "clip.mp4").read_bytes() == b"video"
    assert not (tmp_path / "legacy.jpg.json").exists()


@pytest.mark.parametrize("phase", ["moving", "restoring"])
def test_partial_move_recovers_on_restart(tmp_path, monkeypatch, phase):
    library = ImageLibrary(tmp_path)
    item = sample(library)
    entry = library.move_to_trash(item["id"]) if phase == "restoring" else None
    real_rename = Path.rename
    def fail_media(source, target):
        if source.name == "one.png":
            raise PermissionError("simulated Windows file lock")
        return real_rename(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_media)
        with pytest.raises(PermissionError):
            library.restore(entry["id"]) if entry else library.move_to_trash(item["id"])
        assert library.gallery()["items"] == []
        assert library.trash_items()["items"][0]["state"] == phase
    recovered = ImageLibrary(tmp_path)
    assert recovered.recovery_errors == []
    assert recovered.gallery()["count"] == (1 if phase == "restoring" else 0)
    if phase == "moving":
        recovered.restore(recovered.trash_items()["items"][0]["id"])
    assert (tmp_path / "one.png").is_file() and (tmp_path / "one.png.json").is_file()


def test_restore_conflict_keeps_both_versions(tmp_path):
    library = ImageLibrary(tmp_path)
    entry = library.move_to_trash(sample(library)["id"])
    (tmp_path / "one.png").write_bytes(b"new file must survive")
    with pytest.raises(FileExistsError):
        library.restore(entry["id"])
    assert (tmp_path / "one.png").read_bytes() == b"new file must survive"
    assert (library.trash / entry["id"] / "one.png").is_file()


def test_redirected_sidecar_is_not_moved(tmp_path):
    library = ImageLibrary(tmp_path)
    Image.new("RGB", (8, 8)).save(tmp_path / "legacy.png")
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_text('{}')
    try:
        (tmp_path / "legacy.png.json").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this host")
    with pytest.raises(ValueError):
        library.move_to_trash(asset_id("legacy.png"))
    assert outside.read_text() == '{}'
    assert (tmp_path / "legacy.png").exists()


def test_arbitrary_paths_and_invalid_ids_rejected(tmp_path):
    library = ImageLibrary(tmp_path)
    sample(library)
    for name in ["../one.png", "..\\one.png", "C:one.png", "one.png "]:
        with pytest.raises(ValueError):
            library.file(name)
    with pytest.raises(ValueError):
        library.restore("../outside")
    with pytest.raises(ValueError):
        library.move_to_trash("one.png")
    assert library.gallery()["count"] == 1


def test_publish_does_not_expose_half_pair(tmp_path, monkeypatch):
    library = ImageLibrary(tmp_path)
    real_rename = Path.rename
    def inspect_final(source, target):
        if target.name == "one.png":
            assert library.gallery()["count"] == 0
            assert json.loads((tmp_path / "one.png.json").read_text())["seed"] == 42
        return real_rename(source, target)
    monkeypatch.setattr(Path, "rename", inspect_final)
    sample(library)
    assert library.gallery()["count"] == 1


def test_double_publication_never_overwrites(tmp_path):
    library = ImageLibrary(tmp_path)
    sample(library)
    before = (tmp_path / "one.png").read_bytes()
    with pytest.raises(FileExistsError):
        sample(library)
    assert (tmp_path / "one.png").read_bytes() == before
