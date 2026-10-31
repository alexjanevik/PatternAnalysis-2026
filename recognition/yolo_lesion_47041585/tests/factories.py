"""Builders for test data."""

from pathlib import Path

from yolo_lesion_47041585.dataset import (
    AuditedRecord,
    CohortRecord,
    NormalizedBox,
    Split,
    SplitRecord,
)


def make_records(
    benign: int, malignant: int, split: Split = "train"
) -> list[SplitRecord]:
    records: list[SplitRecord] = []
    for index in range(benign + malignant):
        is_malignant = index >= benign
        isic_id = f"ISIC_{index:04d}"
        cohort_record = CohortRecord(
            isic_id=isic_id,
            source_set="train",
            image_path=Path(f"{isic_id}.jpg"),
            mask_path=Path(f"{isic_id}.png"),
            patient_id=f"P{index}",
            lesion_id=None,
            diagnosis="Malignant" if is_malignant else "Benign",
            class_id=1 if is_malignant else 0,
        )
        audited = AuditedRecord(
            record=cohort_record,
            image_width=100,
            image_height=100,
            box=NormalizedBox(0.5, 0.5, 0.5, 0.5),
            image_sha256="",
            mask_sha256="",
        )
        records.append(SplitRecord(audited, group_id=f"G{index}", split=split))
    return records
