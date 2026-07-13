import hashlib
import json
from pathlib import Path

from build_final_manifest import materialize_manifest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_hashes_without_image_decoding(tmp_path: Path):
    data = tmp_path / "data"
    images = data / "images_largescale" / "ninco" / "class_a"
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"not-an-image-a")
    (images / "b.jpg").write_bytes(b"not-an-image-b")
    imglist = data / "benchmark_imglist" / "imagenet" / "test_ninco.txt"
    imglist.parent.mkdir(parents=True)
    imglist.write_text("ninco/class_a/a.jpg -1\nninco/class_a/b.jpg -1\n", encoding="utf-8")
    prereg = tmp_path / "prereg.json"
    prereg.write_text(
        json.dumps(
            {
                "final_evaluation_executed": False,
                "datasets": [{"name": "NINCO", "openood_alias": "ninco", "expected_images": 2}],
            }
        ),
        encoding="utf-8",
    )
    prereg.with_suffix(".sha256").write_text(f"{digest(prereg)}  prereg.json\n", encoding="utf-8")

    summary = materialize_manifest(data, prereg, tmp_path / "manifest")

    assert summary["total_images"] == 2
    assert summary["image_decoding_executed"] is False
    assert summary["score_computation_executed"] is False
    assert len((tmp_path / "manifest" / "final_dataset_manifest.jsonl").read_text().splitlines()) == 2
