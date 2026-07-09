"""Download MSD Task01_BrainTumour dataset.

Source: https://medicaldecathlon.com/
Direct download: https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar

This dataset is freely available without registration.
"""

import argparse
import subprocess
from pathlib import Path


def download_and_extract(output_dir: Path):
    """Download and extract MSD Brain Tumor dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "Task01_BrainTumour"

    if dataset_dir.exists():
        print(f"Dataset already exists at {dataset_dir}")
        return dataset_dir

    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    tar_path = output_dir / "Task01_BrainTumour.tar"

    print("Downloading MSD Task01_BrainTumour (~7.1 GB)...")
    print(f"Source: {url}")

    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(tar_path), url],
        check=True,
    )

    print("Extracting...")
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(output_dir)], check=True)

    tar_path.unlink()
    print(f"Done. Dataset at {dataset_dir}")
    return dataset_dir


def verify_dataset(dataset_dir: Path):
    """Verify the downloaded MSD dataset."""
    import json

    meta_path = dataset_dir / "dataset.json"
    if not meta_path.exists():
        print(f"ERROR: dataset.json not found at {meta_path}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"\nDataset: {meta['name']}")
    print(f"Modalities: {meta['modality']}")
    print(f"Labels: {meta['labels']}")
    print(f"Training subjects: {meta['numTraining']}")
    print(f"Test subjects: {meta['numTest']}")

    # Count actual case files. The MSD tar ships macOS AppleDouble sidecars
    # ("._BRATS_xxx.nii.gz") inside imagesTr/labelsTr; globbing "*.nii.gz" counts
    # those too and inflates the total (e.g. 495 vs the real 484), which used to
    # make a perfectly good download report FAILED. Filter hidden dot-files out.
    def real_niis(subdir):
        return [p for p in (dataset_dir / subdir).glob("*.nii.gz")
                if not p.name.startswith((".", "._"))]

    images_tr = real_niis("imagesTr")
    labels_tr = real_niis("labelsTr")
    images_ts = real_niis("imagesTs")

    print("\nFiles found (excluding hidden sidecar files):")
    print(f"  Training images: {len(images_tr)}")
    print(f"  Training labels: {len(labels_tr)}")
    print(f"  Test images:     {len(images_ts)}")

    n_expected = meta["numTraining"]
    if len(images_tr) == len(labels_tr) == n_expected:
        print(f"\nVerification: PASSED ({n_expected} training image/label pairs)")
    else:
        print(f"\nVerification: FAILED — expected {n_expected} training image/label "
              f"pairs, found {len(images_tr)} images / {len(labels_tr)} labels")


def main():
    parser = argparse.ArgumentParser(description="Download MSD Brain Tumor dataset")
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.verify_only:
        verify_dataset(output_dir / "Task01_BrainTumour")
    else:
        dataset_dir = download_and_extract(output_dir)
        verify_dataset(dataset_dir)


if __name__ == "__main__":
    main()
