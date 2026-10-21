"""
ISIC2018 data pipeline: cohort, pair audit, patient split and YOLO export.

Reusable dataset logic is contained here. The stages run in order (build_cohort,
audit_pairs, build_independence_groups, make_grouped_splits, validate_split_manifest,
export_yolo_dataset) and are driven by scripts/audit_dataset.py, which also decides the
split seed and mask threshold. Detailed reasoning of the data decisions can be found in
the project's README.
"""

import csv
import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# type aliases for the diagnoses
Diagnosis = Literal["Benign", "Malignant"]
ClassId = Literal[0, 1]
CLASS_MAPPING: Mapping[Diagnosis, ClassId] = {
    "Benign": 0,
    "Malignant": 1,
}


@dataclass(frozen=True)
class CohortRecord:
    """An included image with its mask, identifiers and class.

    Attributes:
        isic_id (str): ISIC image identifier.
        source_set (str): Name of the original set the image came from.
        image_path (Path): Path to the image file.
        mask_path (Path): Path to the segmentation mask file.
        patient_id (str): Patient identifier.
        lesion_id (str | None): Lesion identifier, or None if the metadata has none.
        diagnosis (Diagnosis): The record's diagnosis.
        class_id (ClassId): 0 for Benign, 1 for Malignant.
    """

    isic_id: str
    source_set: str
    image_path: Path
    mask_path: Path
    patient_id: str
    lesion_id: str | None
    diagnosis: Diagnosis
    class_id: ClassId


@dataclass(frozen=True)
class ExcludedRecord:
    """A rejected metadata row and all of its exclusion reasons.

    Attributes:
        metadata_row (int): Zero-based index of the row in the raw CSV, excluding the
            header.
        isic_id (str | None): The row's ISIC ID, or None if it has none.
        reasons (tuple[str, ...]): Every rule the row failed.
    """

    metadata_row: int
    isic_id: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CohortResult:
    """The included and excluded records, and the raw metadata they came from.

    Attributes:
        records (tuple[CohortRecord, ...]): Included records, ordered by ISIC ID.
        exclusions (tuple[ExcludedRecord, ...]): Rejected rows, in metadata order.
        raw_metadata_sha256 (str): SHA-256 of the raw metadata file's bytes.
        input_row_count (int): Number of data rows in the raw metadata.
    """

    records: tuple[CohortRecord, ...]
    exclusions: tuple[ExcludedRecord, ...]
    raw_metadata_sha256: str
    input_row_count: int


def _index_files(
    root: Path,
    pattern: str,
    id_suffix: str = "",
) -> dict[str, Path]:
    """Index files in a directory tree by their ISIC identifier.

    Args:
        root (Path): Root directory to search for files.
        pattern (str): Glob pattern to match files.
        id_suffix (str): Suffix to remove from the file stem to obtain the
            ISIC ID (default: "").

    Returns:
        dict[str, Path]: Mapping of ISIC IDs to file paths.

    Raises:
        ValueError: If multiple files are found for the same ISIC ID.
    """
    paths_by_id: dict[str, Path] = {}

    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue

        isic_id = path.stem.removesuffix(id_suffix)

        if isic_id in paths_by_id:
            raise ValueError(
                f"Multiple files for {isic_id}: {paths_by_id[isic_id]} and {path}"
            )

        paths_by_id[isic_id] = path

    return paths_by_id


def build_cohort(
    raw_metadata: Path,
    image_roots: Mapping[str, Path],
    mask_roots: Mapping[str, Path],
) -> CohortResult:
    """Build the cohort of included records from the raw metadata.

    Applies the inclusion and exclusion rules, and pairs each retained record with
    its image and mask. Only Benign (0) and Malignant (1) records with patient IDs
    are retained.

    Args:
        raw_metadata (Path): Path to the raw metadata CSV file.
        image_roots (Mapping[str, Path]): Mapping of source set names to image root
            directories
        mask_roots (Mapping[str, Path]): Mapping of source set names to mask root
            directories

    Returns:
        CohortResult: Included records, exclusions with reasons, and a hash and row
            count of the raw metadata.

    Raises:
        ValueError: If the metadata path is not a file, required columns are missing,
            roots are invalid, or an image ID appears in multiple sources.
    """
    # ensure the raw metadata file and image/mask roots exist
    if not raw_metadata.is_file():
        raise ValueError(f"Raw metadata file not found: {raw_metadata}")

    if not image_roots:
        raise ValueError("No image roots provided")
    if set(image_roots) != set(mask_roots):
        raise ValueError("Image and mask roots must have identical source-set keys")

    for root in image_roots.values():
        if not root.is_dir():
            raise ValueError(f"Image root directory not found: {root}")
    for root in mask_roots.values():
        if not root.is_dir():
            raise ValueError(f"Mask root directory not found: {root}")

    required_columns = {"isic_id", "lesion_id", "patient_id", "diagnosis_1"}

    # utf-8-sig drops any byte-order mark. every field is read as a string
    with raw_metadata.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing_columns = required_columns - set(reader.fieldnames or ())

        if missing_columns:
            raise ValueError(
                f"Raw metadata file is missing required columns: {', '.join(sorted(missing_columns))}"
            )

        rows = list(reader)

    # hash the raw metadata for provenance
    metadata_hash = hashlib.sha256(raw_metadata.read_bytes()).hexdigest()

    def clean(value: str | None) -> str | None:
        # strip whitespace and treat empty strings as missing
        return (value or "").strip() or None

    images_by_source = {
        source: _index_files(root, "*.jpg") for source, root in image_roots.items()
    }
    masks_by_source = {
        source: _index_files(root, "*_segmentation.png", "_segmentation")
        for source, root in mask_roots.items()
    }

    id_counts = Counter(clean(row["isic_id"]) for row in rows)
    duplicate_ids = {
        isic_id for isic_id, count in id_counts.items() if isic_id and count > 1
    }

    records: list[CohortRecord] = []
    exclusions: list[ExcludedRecord] = []

    for metadata_row, row in enumerate(rows):
        reasons: list[str] = []
        source_set: str | None = None
        image_path: Path | None = None
        mask_path: Path | None = None

        isic_id = clean(row["isic_id"])
        patient_id = clean(row["patient_id"])
        lesion_id = clean(row["lesion_id"])
        diagnosis = clean(row["diagnosis_1"])

        # check for missing or invalid fields
        if isic_id is None:
            reasons.append("missing_isic_id")
        elif isic_id in duplicate_ids:
            reasons.append("duplicate_metadata_id")
        if diagnosis not in CLASS_MAPPING:
            reasons.append("unsupported_diagnosis")
        if patient_id is None:
            reasons.append("missing_patient_id")

        if isic_id is not None:
            # find all sources containing the image for this ISIC ID
            matching_sources = [
                source
                for source, images in images_by_source.items()
                if isic_id in images
            ]

            if not matching_sources:
                reasons.append("missing_image")

            elif len(matching_sources) > 1:
                raise ValueError(
                    f"ISIC ID {isic_id} appears in multiple image "
                    f"sources: {sorted(matching_sources)}"
                )

            else:
                source_set = matching_sources[0]
                image_path = images_by_source[source_set][isic_id]
                mask_path = masks_by_source[source_set].get(isic_id)

                if mask_path is None:
                    reasons.append("missing_mask")

        # append to the exclusion list if any reasons were found
        if reasons:
            exclusions.append(
                ExcludedRecord(
                    metadata_row=metadata_row,
                    isic_id=isic_id,
                    reasons=tuple(reasons),
                )
            )
            continue

        # all inclusion checks have passed.
        assert isic_id is not None
        assert patient_id is not None
        assert source_set is not None
        assert image_path is not None
        assert mask_path is not None
        assert diagnosis in ("Benign", "Malignant")

        # append to record list
        records.append(
            CohortRecord(
                isic_id=isic_id,
                source_set=source_set,
                image_path=image_path,
                mask_path=mask_path,
                patient_id=patient_id,
                lesion_id=lesion_id,
                diagnosis=diagnosis,
                class_id=CLASS_MAPPING[diagnosis],
            )
        )

    return CohortResult(
        records=tuple(sorted(records, key=lambda record: record.isic_id)),
        exclusions=tuple(exclusions),
        raw_metadata_sha256=metadata_hash,
        input_row_count=len(rows),
    )
