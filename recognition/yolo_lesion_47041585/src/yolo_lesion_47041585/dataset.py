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
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

# type aliases for the diagnoses
Diagnosis = Literal["Benign", "Malignant"]
ClassId = Literal[0, 1]
CLASS_MAPPING: Mapping[Diagnosis, ClassId] = {
    "Benign": 0,
    "Malignant": 1,
}

# split names matching the keys YOLO dataset YAML files use
Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")
DEFAULT_SPLIT_RATIOS: Mapping[Split, float] = {"train": 0.70, "val": 0.15, "test": 0.15}


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


@dataclass(frozen=True)
class NormalizedBox:
    """A bounding box derived from a binary mask, in centre format.

    Attributes:
        x_center (float): Horizontal centre, as a fraction of the image width.
        y_center (float): Vertical centre, as a fraction of the image height.
        width (float): Box width, as a fraction of the image width.
        height (float): Box height, as a fraction of the image height.
    """

    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class AuditIssue:
    """A blocking issue found when auditing an image/mask pair.

    Attributes:
        isic_id (str): ISIC ID of the pair with the issue.
        code (str): Short identifier of the check that failed.
        detail (str): Explanation of the failure.
    """

    isic_id: str
    code: str
    detail: str


@dataclass(frozen=True)
class AuditedRecord:
    """A cohort record with its image size, bounding box and file hashes.

    Attributes:
        record (CohortRecord): The audited cohort record.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        box (NormalizedBox): Box around the mask's foreground.
        image_sha256 (str): SHA-256 of the image file's bytes.
        mask_sha256 (str): SHA-256 of the mask file's bytes.
    """

    record: CohortRecord
    image_width: int
    image_height: int
    box: NormalizedBox
    image_sha256: str
    mask_sha256: str


@dataclass(frozen=True)
class PairAuditResult:
    """The audited records and any issues found while auditing the pairs.

    Attributes:
        records (tuple[AuditedRecord, ...]): Pairs that passed the audit.
        issues (tuple[AuditIssue, ...]): Issues found, at most one per pair.
    """

    records: tuple[AuditedRecord, ...]
    issues: tuple[AuditIssue, ...]

    @property
    def passed(self) -> bool:
        """Whether the audit found no issues."""
        return not self.issues


@dataclass(frozen=True)
class SplitRecord:
    """An audited record assigned to a split, with its independence group.

    Attributes:
        audited (AuditedRecord): The audited record.
        group_id (str): ID of the independence group the record belongs to.
        split (Split): The split the record is assigned to.
    """

    audited: AuditedRecord
    group_id: str
    split: Split


@dataclass(frozen=True)
class SplitManifest:
    """Every record's split assignment.

    Attributes:
        records (tuple[SplitRecord, ...]): One entry per record, ordered by ISIC ID.
    """

    records: tuple[SplitRecord, ...]


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


def mask_to_box(mask: NDArray[np.bool_]) -> NormalizedBox:
    """Convert a binary mask to a normalized bounding box.

    Args:
        mask (NDArray[np.bool_]): A 2D binary mask where True indicates the lesion.

    Returns:
        NormalizedBox: The bounding box in normalized coordinates
            (x_center, y_center, width, height).

    Raises:
        ValueError: If the mask is not a non-empty 2D boolean array, or has no
            foreground pixels.
    """
    if mask.dtype != np.bool_:
        raise ValueError(f"Mask must be boolean, got dtype={mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={mask.shape}")
    if mask.size == 0:
        raise ValueError(f"Mask has a zero-sized axis, got shape={mask.shape}")

    image_height, image_width = mask.shape
    ys, xs = np.nonzero(mask)

    if xs.size == 0:
        raise ValueError("Mask has no foreground pixels")

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    # calculate normalized center coordinates
    x_center = (x_min + x_max + 1) / (2 * image_width)
    y_center = (y_min + y_max + 1) / (2 * image_height)

    # calculate normalized width and height
    width = (x_max - x_min + 1) / image_width
    height = (y_max - y_min + 1) / image_height

    return NormalizedBox(
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


class _PairIssue(Exception):
    """A blocking problem with one image/mask pair, raised by _audit_pair.

    Args:
        code (str): Short identifier of the check that failed.
        detail (str): Explanation of the failure.
    """

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _audit_pair(record: CohortRecord, threshold: int) -> AuditedRecord:
    """Audit one image/mask pair and derive its bounding box.

    Args:
        record (CohortRecord): The pair to audit.
        threshold (int): Mask intensity at or above which a pixel is foreground.

    Returns:
        AuditedRecord: The pair with its image size, box and file hashes.

    Raises:
        _PairIssue: On the first blocking problem found with the pair.
    """
    # read each file once to avoid repeated I/O
    try:
        image_bytes = record.image_path.read_bytes()
        mask_bytes = record.mask_path.read_bytes()
    except OSError as exc:
        raise _PairIssue("file_read_error", str(exc)) from exc

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image_width, image_height = image.size

        with Image.open(BytesIO(mask_bytes)) as mask_image:
            mask_image.load()
            mask_mode = mask_image.mode
            mask = np.array(mask_image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        # Pillow's message for unreadable data embeds a memory address, which
        # would make the issues file differ between runs
        detail = (
            "Not a recognisable image file"
            if isinstance(exc, UnidentifiedImageError)
            else str(exc)
        )
        raise _PairIssue("decode_error", detail) from exc

    # check if 8-bit greycale 2D mask
    if mask_mode != "L" or mask.ndim != 2:
        raise _PairIssue(
            "unexpected_mask_format",
            f"Expected 2D greyscale mask, got mode={mask_mode}, shape={mask.shape}",
        )

    # check if image and mask dimensions match
    mask_height, mask_width = mask.shape
    if (image_width, image_height) != (mask_width, mask_height):
        raise _PairIssue(
            "dimension_mismatch",
            f"Image={image_width}×{image_height}, mask={mask_width}×{mask_height}.",
        )

    # check if mask contains any foreground pixels
    foreground = mask >= threshold
    if not foreground.any():
        raise _PairIssue("empty_mask", "Mask contains no foreground pixels.")

    return AuditedRecord(
        record=record,
        image_width=image_width,
        image_height=image_height,
        box=mask_to_box(foreground),
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
    )


def audit_pairs(
    records: Sequence[CohortRecord], threshold: int = 128
) -> PairAuditResult:
    """Audit image/mask pairs for consistency and generate bounding boxes.

    Each pair reports at most one issue, the first blocking problem found.
    Unexpected errors (e.g. programming bugs) are not caught and will propagate.

    Args:
        records (Sequence[CohortRecord]): A sequence of CohortRecord objects.
        threshold (int): Pixel intensity threshold for foreground in the mask, default
            value chosen due to interpolated 2x upscaling of the original 8-bit masks
            (default: 128).

    Returns:
        PairAuditResult: Contains audited records and any errors found during the audit.

    Raises:
        ValueError: If the threshold is not in the range [1, 255].
    """
    if not 1 <= threshold <= 255:
        raise ValueError(f"Threshold must be in [1, 255], got threshold={threshold}")

    audited: list[AuditedRecord] = []
    issues: list[AuditIssue] = []

    for record in sorted(records, key=lambda record: record.isic_id):
        try:
            audited.append(_audit_pair(record, threshold))
        except _PairIssue as issue:
            issues.append(
                AuditIssue(
                    isic_id=record.isic_id,
                    code=issue.code,
                    detail=issue.detail,
                )
            )

    return PairAuditResult(records=tuple(audited), issues=tuple(issues))


def build_independence_groups(records: Sequence[AuditedRecord]) -> dict[str, str]:
    """Group records that must stay together in a single split.

    A group is all of one patient's records, identified by the smallest ISIC ID
    among them. Lesion IDs are not used to link patients: a lesion ID recorded
    under more than one patient is contradictory, so it raises for review.

    Args:
        records (Sequence[AuditedRecord]): Audited records to group.

    Returns:
        dict[str, str]: A group ID for every ISIC ID.

    Raises:
        ValueError: If an ISIC ID appears more than once, or a lesion ID is recorded
            under more than one patient.
    """
    group_by_id: dict[str, str] = {}
    first_id_by_patient: dict[str, str] = {}
    patients_by_lesion: dict[str, set[str]] = {}

    # in ISIC ID order, so a patient's first record is its smallest ID
    for audited in sorted(records, key=lambda a: a.record.isic_id):
        record = audited.record

        if record.isic_id in group_by_id:
            raise ValueError(f"Duplicate ISIC ID: {record.isic_id}")

        group_id = first_id_by_patient.setdefault(record.patient_id, record.isic_id)
        group_by_id[record.isic_id] = group_id

        if record.lesion_id is not None:
            patients_by_lesion.setdefault(record.lesion_id, set()).add(
                record.patient_id
            )

    conflicts = {
        lesion_id: sorted(patients)
        for lesion_id, patients in sorted(patients_by_lesion.items())
        if len(patients) > 1
    }
    if conflicts:
        raise ValueError(f"Lesion IDs under more than one patient: {conflicts}")

    return group_by_id


def make_grouped_splits(
    records: Sequence[AuditedRecord],
    groups: Mapping[str, str],
    seed: int,
    ratios: Mapping[Split, float] = DEFAULT_SPLIT_RATIOS,
) -> SplitManifest:
    """Assign whole independence groups to train, val and test splits.

    Groups are placed largest first, each into the split whose image, benign and
    malignant counts it brings closest to their targets. Groups are never divided,
    so group separation takes precedence over exact ratios. Model performance is
    never used.

    Args:
        records (Sequence[AuditedRecord]): Audited records to split.
        groups (Mapping[str, str]): Group ID for every ISIC ID.
        seed (int): Seed for tie-breaking between equally sized groups.
        ratios (Mapping[Split, float]): Target fraction of images per split, which
            must sum to 1 (default: 70/15/15).

    Returns:
        SplitManifest: One split assignment per record, ordered by ISIC ID.

    Raises:
        ValueError: If the ratios are invalid, an ISIC ID is duplicated or has no
            group, the records lack a class, or a split ends up without both classes.
    """
    if set(ratios) != set(SPLITS) or any(r <= 0 for r in ratios.values()):
        raise ValueError(f"Ratios must be positive and cover {SPLITS}: {dict(ratios)}")
    if abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError(f"Ratios must sum to 1, got {sum(ratios.values())}")

    ids = [audited.record.isic_id for audited in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ISIC IDs in records")
    if missing := sorted(set(ids) - set(groups)):
        raise ValueError(f"Records without a group: {missing[:5]}")

    # count images, benign and malignant records per group
    group_counts: dict[str, np.ndarray] = {}
    for audited in records:
        counts = group_counts.setdefault(groups[audited.record.isic_id], np.zeros(3))
        counts += (1, audited.record.class_id == 0, audited.record.class_id == 1)

    # check that both classes are present in the dataset
    # creates empty totals array first to avoid dtype issues
    totals = sum(group_counts.values(), start=np.zeros(3))
    if totals[1] == 0 or totals[2] == 0:
        raise ValueError(f"Records must include both classes, got {totals[1:]}")
    targets = {split: ratios[split] * totals for split in SPLITS}
    current = {split: np.zeros(3) for split in SPLITS}

    # relative squared deviation from target counts for a split
    def deviation(split: Split, counts: np.ndarray) -> float:
        # squared deviation from target, relative so the rarer malignant class counts
        # as much as the images total
        return float((((counts - targets[split]) / targets[split]) ** 2).sum())

    # largest groups first. ties are broken by the seed
    order = sorted(group_counts)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda group_id: -group_counts[group_id][0])

    split_by_group: dict[str, Split] = {}
    for group_id in order:
        # compare how much each split's deviation would change if this group is added
        split_by_group[group_id] = min(
            SPLITS,
            key=lambda split: (
                deviation(split, current[split] + group_counts[group_id])
                - deviation(split, current[split])
            ),
        )
        current[split_by_group[group_id]] += group_counts[group_id]

    # ensure that each split has at least one record from each class
    for split in SPLITS:
        if current[split][1] == 0 or current[split][2] == 0:
            raise ValueError(f"Split '{split}' is missing a class: {current[split]}")

    return SplitManifest(
        records=tuple(
            SplitRecord(
                audited=audited,
                group_id=groups[audited.record.isic_id],
                split=split_by_group[groups[audited.record.isic_id]],
            )
            for audited in sorted(records, key=lambda a: a.record.isic_id)
        ),
    )
