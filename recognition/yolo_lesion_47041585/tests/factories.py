"""Builders for test data."""


def make_classes(benign: int, malignant: int) -> dict[str, int]:
    """Class IDs by ISIC ID for a training split, with the malignant images last."""
    return {
        f"ISIC_{index:04d}": int(index >= benign) for index in range(benign + malignant)
    }
