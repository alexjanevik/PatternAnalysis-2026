"""
Download the ISIC 2017 images, metadata, and binary segmentation masks.
---------
Lesion images and metadata are downloaded via ISIC's `isic-cli` tool. This is
super convenient as the files in this archive are compressed and thus much smaller
in file-size than the original images. It also doesn't contain the superpixel
masks which aren't needed for this project anyway. This download comes with the
CSV metadata file used for classification.

Unfortunately, `isic-cli` doesn't contain the binary segmentation masks. These
are downloaded from the ISIC 2017 challenge website as ZIP archives. The archives
are extracted into the dataset directory and the archive is deleted afterwards.
The archive is only downloaded again if the existing masks are missing or if the
`--overwrite` flag is used.
"""

import argparse
import csv
import http.client
import importlib.util
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SPLITS = {
    "train": (60, "ISIC-2017_Training_Part1_GroundTruth.zip"),
    "test": (69, "ISIC-2017_Test_v2_Part1_GroundTruth.zip"),
    "validation": (71, "ISIC-2017_Validation_Part1_GroundTruth.zip"),
}
BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2017/"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def download_images(split_dir: Path, collection: int, overwrite: bool) -> None:
    raw_dir = split_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in raw_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                path.unlink()

    # Use the same environment as this script, without relying on uv or PATH.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from isic_cli.cli import main; main()",
            "image",
            "download",
            "--collections",
            str(collection),
            str(raw_dir),
        ],
        check=True,
    )
    metadata = raw_dir / "metadata.csv"
    if not metadata.is_file():
        raise RuntimeError(f"isic-cli did not produce {metadata}")
    metadata.replace(split_dir / "metadata.csv")


def image_ids(split_dir: Path) -> set[str]:
    with (split_dir / "metadata.csv").open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if "isic_id" not in (reader.fieldnames or []):
            raise RuntimeError("Metadata has no isic_id column")
        rows = [row["isic_id"] for row in reader]
    if not rows or any(not value for value in rows) or len(rows) != len(set(rows)):
        raise RuntimeError(
            "Metadata contains empty or duplicate image IDs, or no images"
        )
    expected = set(rows)
    images = [
        path.stem
        for path in (split_dir / "raw").iterdir()
        if path.is_file()
        and path.stat().st_size > 0
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    check_ids("images", expected, set(images))
    if len(images) != len(set(images)):
        raise RuntimeError("Multiple image files have the same ISIC ID")
    return expected


def check_ids(kind: str, expected: set[str], actual: set[str]) -> None:
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise RuntimeError(
            f"{kind} do not match metadata: "
            f"{len(missing)} missing ({', '.join(sorted(missing)[:5])}); "
            f"{len(extra)} extra ({', '.join(sorted(extra)[:5])})"
        )


def download_masks(
    split_dir: Path,
    cache_dir: Path,
    archive_name: str,
    expected: set[str],
    overwrite: bool,
) -> None:
    mask_dir = split_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / archive_name
    existing = {
        path.name.removesuffix("_segmentation.png")
        for path in mask_dir.glob("*_segmentation.png")
        if path.is_file() and path.stat().st_size > 0
    }
    if existing != expected or overwrite:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if overwrite or not archive.is_file():
            partial = archive.with_suffix(".zip.part")
            print(f"Downloading {archive_name}", flush=True)
            with (
                urllib.request.urlopen(BASE_URL + archive_name, timeout=60) as response,
                partial.open("wb") as destination,
            ):
                shutil.copyfileobj(response, destination)
            partial.replace(archive)

        try:
            with zipfile.ZipFile(archive) as source:
                members = [
                    member
                    for member in source.infolist()
                    if not member.is_dir()
                    and Path(member.filename).name.endswith("_segmentation.png")
                    and "__MACOSX" not in Path(member.filename).parts
                ]
                names = [Path(member.filename).name for member in members]
                ids = {name.removesuffix("_segmentation.png") for name in names}
                check_ids("archived masks", expected, ids)
                if len(names) != len(set(names)):
                    raise RuntimeError("Mask archive contains duplicate filenames")
                for member, name in zip(members, names):
                    # Flatten only recognised mask filenames; never extract archive paths.
                    target = mask_dir / name
                    if (
                        not overwrite
                        and target.is_file()
                        and target.stat().st_size == member.file_size
                    ):
                        continue
                    partial = target.with_suffix(".png.part")
                    with (
                        source.open(member) as contents,
                        partial.open("wb") as destination,
                    ):
                        shutil.copyfileobj(contents, destination)
                    partial.replace(target)
        except zipfile.BadZipFile:
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                "Corrupt mask ZIP removed; rerun to download it again"
            ) from None

    actual = {
        path.name.removesuffix("_segmentation.png")
        for path in mask_dir.glob("*_segmentation.png")
        if path.is_file() and path.stat().st_size > 0
    }
    check_ids("masks", expected, actual)
    archive.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download images and masks again, replacing existing files",
    )
    args = parser.parse_args()
    if importlib.util.find_spec("isic_cli") is None:
        parser.error(
            "isic-cli is missing; install project dependencies with pip install . or uv sync"
        )
    try:
        for split in dict.fromkeys(args.splits):
            print(f"Preparing {split}", flush=True)
            collection, archive_name = SPLITS[split]
            split_dir = args.dataset_dir / split
            download_images(split_dir, collection, args.overwrite)
            expected = image_ids(split_dir)
            download_masks(
                split_dir,
                args.dataset_dir / ".downloads",
                archive_name,
                expected,
                args.overwrite,
            )
            print(f"Verified {split}: {len(expected)} image–mask pairs", flush=True)
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        EOFError,
        http.client.HTTPException,
    ) as error:
        print(
            f"Download failed: {error}\nRerun the same command to retry.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
