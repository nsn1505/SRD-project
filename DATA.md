# Dataset and Data Preparation
This document describes the data actually used by `SRD-project.ipynb` and the accompanying report. The reported experiment uses **300 training images, 50 internal validation images, and 50 held-out test images**, not all 2,700 files in the downloaded package.

## 1. Sources and version

| Field | Value |
| --- | --- |
| Dataset | DIV2K: DIVerse 2K resolution images |
| Original dataset website | https://data.vision.ee.ethz.ch/cvl/DIV2K/ |
| Download source used by this project | https://www.kaggle.com/datasets/takihasan/div2k-dataset-for-super-resolution |
| Kaggle owner / handle | `takihasan/div2k-dataset-for-super-resolution` |
| Kaggle version documented here | **1**, initial release |
| Release timestamp reported by Kaggle | `2024-03-11T12:37:45.813Z` |
| Version-specific page | https://www.kaggle.com/datasets/takihasan/div2k-dataset-for-super-resolution/versions/1 |
| Metadata verification | 5 October 2026, using the public dataset metadata endpoint below |
| Package size reported by Kaggle | 5,305,173,909 bytes, about 5.31 GB decimal |

Metadata endpoint: https://www.kaggle.com/api/v1/datasets/view/takihasan/div2k-dataset-for-super-resolution

The original dataset is associated with the NTIRE 2017/2018 super-resolution challenges. The Kaggle package contains training and validation HR images and bicubic x2/x4 counterparts. It does not supply the official hidden HR test set. The notebook's recorded folder counts are consistent with this release, but the original downloaded archive checksum was not recorded; byte-for-byte provenance of that local copy has not been verified.

The Kaggle metadata displays an MIT license label. The original DIV2K website states that images are provided for academic research and remain copyrighted by their original owners. These are different source notices; this project does not assign a new license to the images or treat the mirror label as overriding the original notice.

## 2. Package contents observed in the notebook

Paths below are relative to the user's `Dataset` directory. An extracted archive may add an outer folder; cell 3 searches recursively.

| Directory | Images | Used by the reported pipeline? |
| --- | ---: | --- |
| `DIV2K_train_HR/` | 800 | Yes, source of training and internal validation |
| `DIV2K_valid_HR/` | 100 | Yes, source of the project test subset |
| `DIV2K_train_LR_bicubic/X2/` | 800 | No |
| `DIV2K_train_LR_bicubic_X4/X4/` | 800 | No |
| `DIV2K_valid_LR_bicubic/X2/` | 100 | No |
| `DIV2K_valid_LR_bicubic_X4/X4/` | 100 | No |
| **Total files in these six image directories** | **2,700** | **900 distinct HR source images** |

LR counterparts are alternate versions of the HR content, not additional independent scenes. The notebook generates LR inputs from HR images, so precomputed LR folders are not required for its calculations. There is no need to supply separate blurred/noisy training folders.

## 3. Obtain the data

### Option A: use the existing Google Drive copy

In notebook cell 1, set `DRIVE_PATH` to the actual extracted folder or ZIP file:

```python
DRIVE_PATH = "/content/drive/MyDrive/Dataset"
```

The supplied notebook mounts Drive and uses this folder directly. If `DRIVE_PATH` ends in `.zip`, it extracts to `/content/div2k` only when that directory does not already exist. If an earlier extraction was interrupted, use a fresh extraction directory or verify completeness before continuing; the existence check alone does not establish a complete dataset.

### Option B: download the documented version with KaggleHub

This is an optional preparation step, not code already present in the supplied notebook.

```python
%pip install kagglehub
```

```python
import kagglehub
downloaded_root = kagglehub.dataset_download(
    "takihasan/div2k-dataset-for-super-resolution/versions/1"
)
print(downloaded_root)
```

Use the printed directory as `DRIVE_PATH` in cell 1, in the same runtime. KaggleHub supports public downloads, although some account or access configurations may require Kaggle authentication; follow its official instructions if prompted. Do not commit credentials to the repository.

A download into Colab's local cache is temporary. Preserve an original copy on Drive if it must survive runtime deletion. Moving data to `/content` for faster reads changes storage location, not the split or experiment settings. Do not download the package again when a verified complete copy already exists.

KaggleHub documentation: https://github.com/Kaggle/kagglehub

## 4. Exact experimental split

The notebook sorts file paths, seeds Python's `random` module with 42, and shuffles the 800 training HR paths. It then slices the shuffled list as follows:

| Project split | Source and selection | Count | Samples constructed |
| --- | --- | ---: | ---: |
| Internal validation | First 50 shuffled training HR paths | 50 | 800 fixed HR patches |
| Training | Next 300 shuffled training HR paths | 300 | 4,800 fixed HR patches |
| Project test | First 50 sorted official validation HR paths | 50 | 50 central HR crops |
| Unused training HR | Remaining training HR paths | 450 | 0 |
| Unused official validation HR | Remaining official validation HR paths | 50 | 0 |

With standard DIV2K filenames, the project test images are `0801.png` through `0850.png`. They are not the official DIV2K challenge test images. The split is performed **before** patch extraction, so training and validation patches come from different source images. The code does not perform a content-based duplicate audit.

Equivalent split logic for standard, directly identified HR folders:

```python
from pathlib import Path
import random

train_dir = Path("/content/drive/MyDrive/Dataset/DIV2K_train_HR")
valid_dir = Path("/content/drive/MyDrive/Dataset/DIV2K_valid_HR")

def image_paths(folder):
    return sorted(str(p) for ext in ("*.png", "*.jpg", "*.jpeg")
                  for p in folder.glob(ext))

all_train = image_paths(train_dir)
all_valid = image_paths(valid_dir)
assert len(all_train) == 800 and len(all_valid) == 100
random.Random(42).shuffle(all_train)
val_files = all_train[:50]
train_files = all_train[50:350]
test_files = all_valid[:50]
assert not (set(train_files) & set(val_files))
assert not (set(train_files) & set(test_files))
assert not (set(val_files) & set(test_files))
```

This reproduces file selection given the same sorted paths and Python shuffle behavior. Run the actual notebook from a fresh runtime to preserve its complete sequence of random operations. Cell 3 also has a fallback that reserves the last 100 training images if no validation directory is found. **That fallback was not used in the reported run** and is not the protocol above.

## 5. HR preprocessing and patches

1. Open each image with Pillow, convert to RGB, convert to `float32`, and divide by 255.
2. From each training and internal validation image, sample 16 random 96 x 96 patches using NumPy's seeded RNG. Coordinates are sampled uniformly from valid top-left positions.
3. Stack patches into RAM arrays: training `(4800, 96, 96, 3)` and validation `(800, 96, 96, 3)`.
4. Take one central 256 x 256 crop from each project test image: `(50, 256, 256, 3)`.
5. Apply random horizontal flipping when constructing training pairs. The current validation pipeline also applies this flip because it reuses the same pair functions.

Patch locations are sampled once when cell 4 runs, not resampled every epoch. Online flips and random degradations can change on later passes. Patches may overlap within the same image. The original run assumes images are large enough for these crops; it does not implement small-image padding.

The three HR arrays occupy approximately 659 MB decimal (628 MiB) before TensorFlow datasets, cached test inputs, activations and other overhead. This is not a peak-RAM measurement. The recorded data preparation time was 93 seconds for the submitted run.

## 6. LR synthesis

All experiments use scale **x2**. The network consumes a bicubic-upsampled LR image at HR spatial size and predicts a correction to it.

For A and B: optional horizontal flip of HR -> bicubic downsampling with antialiasing -> clipping to [0,1] -> bicubic upsampling -> clipping to [0,1]. A 96 x 96 HR patch becomes 48 x 48 LR and then a 96 x 96 network input.

For C, the following operations occur in a fixed order:

| Operation | Probability | Parameters |
| --- | ---: | --- |
| Gaussian blur before downsampling | 0.5 | Isotropic 9 x 9 normalized kernel; sigma sampled in [0.2, 2.0); reflect padding; same kernel for RGB |
| Bicubic downsampling | 1.0 | Scale x2; antialiasing enabled |
| Additive Gaussian noise at LR size | 0.5 | Standard deviation sampled in [1, 15)/255; clip after adding noise |
| JPEG encoding/decoding | 0.5 | `tf.image.random_jpeg_quality(lr, 30, 95)`; integer quality sampled with an exclusive upper bound |
| Bicubic upsampling | 1.0 | Return to HR spatial size; clip to [0,1] |

The HR target is not blurred, downsampled, noised or JPEG-compressed. LR pairs are generated in memory; the notebook does not publish a separate processed dataset. Consequently, there is no separate processed-data download link to supply. The source URLs and notebook functions define how to regenerate the data.

## 7. Fixed test conditions

Cell 6 resets TensorFlow's seed to 42 and generates each set once. All methods then use the same stored test inputs within that run.

| Condition | Blur sigma | Noise standard deviation | JPEG quality |
| --- | ---: | ---: | --- |
| Clean (bicubic only) | 0 | 0 | Disabled |
| Blur | 1.5 | 0 | Disabled |
| Noise | 0 | 10/255 | Disabled |
| JPEG | 0 | 0 | 30 |
| Blur+Noise+JPEG | 1.0 | 5/255 | 50 |

Every row also includes bicubic downsampling and upsampling. Test HR crops are 256 x 256, LR images are 128 x 128, and network inputs return to 256 x 256. These degradation families are absent from A/B training but are represented in C's training distribution. This is not evidence that C generalizes to wholly unseen degradation families.

## 8. Reproduction code and audit records

The required preparation code is in `SRD-project.ipynb`; there are no separate preparation scripts in the supplied project. Physical cells are numbered from 1, including the initial Drive cell:

| Cells | Responsibility |
| --- | --- |
| 1-2 | Locate data, imports, seed and configuration |
| 3 | Discover HR folders and select split |
| 4 | Decode images and create HR patches/crops |
| 5 | Fixed degradation functions |
| 6 | Construct fixed test inputs |
| 8 | Construct online clean/random training and validation pipelines |

After cell 3, the following optional audit cell preserves the actual selected paths without changing training data:

```python
from pathlib import Path
import json

audit_dir = Path("/content/drive/MyDrive/SR_Project/outputs/reported_run")
audit_dir.mkdir(parents=True, exist_ok=True)
root = Path(DATA_ROOT)
splits = {"train": train_files, "validation": val_files, "test": test_files}
manifest = {
    "seed": SEED,
    "kaggle_handle": "takihasan/div2k-dataset-for-super-resolution",
    "documented_kaggle_version": 1,
    "splits": {k: [str(Path(p).relative_to(root)) for p in v]
               for k, v in splits.items()},
}
(audit_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
```

The supplied notebook does not contain a saved split manifest, archive hash or complete package lockfile. It seeds Python, NumPy and TensorFlow, but does not enforce fully deterministic TensorFlow operations. Stateful randomness, parallel mapping, changed library versions and running cells out of order can change results. Reproduction should recover the method and qualitative ranking; identical numerical results are not guaranteed.

## 9. Citation

E. Agustsson and R. Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study," IEEE CVPR Workshops, 2017.

R. Timofte et al., "NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results," IEEE CVPR Workshops, 2017.

Official dataset and citation instructions: https://data.vision.ee.ethz.ch/cvl/DIV2K/

Kaggle redistribution by Taki Hasan, version 1: https://www.kaggle.com/datasets/takihasan/div2k-dataset-for-super-resolution/versions/1
