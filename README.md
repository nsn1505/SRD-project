# Image Super-Resolution under Unknown Degradations

A TensorFlow/Keras course project investigating whether random degradation training improves x2 image super-resolution when inputs contain blur, noise or JPEG compression.

The implementation is the self-contained **`SRD-project.ipynb`** notebook. This README documents its existing experiment and saved outputs. It does not assume a separate `train.py`, PyTorch project or command-line interface.

## Models and research question

Can a residual CNN trained with varied synthetic degradations restore degraded images more reliably than a network trained only on bicubic downsampling, and what clean-image performance is lost?

| Method | Architecture | Training input | Parameters |
| --- | --- | --- | ---: |
| Bicubic | Interpolation; no learned model | None | 0 |
| A | SRCNN-style 3-layer CNN with global residual addition | Clean bicubic pairs | 69,251 |
| B | 8 residual blocks, 64 channels, 19 convolution layers | Clean bicubic pairs | 631,299 |
| C | Same architecture as B | Random blur/noise/JPEG pairs | 631,299 |

All learned models operate on bicubic-upsampled RGB inputs. Model A is a residual SRCNN-style adaptation, not an exact reproduction of the original SRCNN. Model C is a compact synthetic-degradation experiment, not Real-ESRGAN.

## Files

Place these documentation files alongside the supplied notebook in the repository:

| File | Purpose |
| --- | --- |
| `SRD-project.ipynb` | Data preparation, training, evaluation and interactive demo |
| `README.md` | Setup and reproduction instructions |
| `DATA.md` | Dataset source/version, split and preprocessing protocol |
| `GroupID_ProjectID_Report.pdf` | Report; replace group/project identifiers before submission |

Model weights and the CSV below are produced when the notebook runs. They are not embedded in the documentation or supplied as separate files here. Keep the dataset and large training artifacts outside Git. The notebook already contains recorded outputs that can be inspected without retraining.

## Dataset and recorded configuration

Download source: https://www.kaggle.com/datasets/takihasan/div2k-dataset-for-super-resolution

Original source: https://data.vision.ee.ethz.ch/cvl/DIV2K/

See [DATA.md](DATA.md) for version 1, source notices, acquisition options and exact split logic.

| Setting | Recorded value |
| --- | --- |
| Framework | TensorFlow 2.20.0 / `tf.keras` |
| Runtime | Google Colab; GPU detected; notebook metadata identifies T4 |
| Seed | 42 for Python, NumPy and TensorFlow |
| Scale | x2 |
| Training / internal validation / project test | 300 / 50 / 50 source images |
| Training / validation patches | 4,800 / 800 |
| HR patch / LR patch | 96 x 96 / 48 x 48 |
| Test HR crop | Central 256 x 256 crop |
| Batch size / epochs per model | 16 / 10 |
| Optimizer / learning rate | Adam / 0.0001 |
| Loss / checkpoint selection | MSE / minimum validation loss |

The 50 internal validation images come from the original training HR folder. The 50 project test images come from the original validation HR folder. The reported experiment is a subset experiment, not full-DIV2K training or evaluation on the official hidden challenge test set.

## Run in Google Colab

### 1. Open the notebook and select a GPU

Open Google Colab, use **File > Upload notebook**, and select `SRD-project.ipynb`. Choose a GPU in the runtime settings. CPU execution is possible but expected to be slower.

Run a separate setup cell before the original notebook cells:

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

The recorded version is 2.20.0. If TensorFlow or the other packages are absent, or you need that TensorFlow version, install:

```python
%pip install "tensorflow==2.20.0" numpy pandas matplotlib pillow h5py
```

Restart the runtime after changing TensorFlow, then execute from the beginning. Do not reinstall packages during a running training job. The exact versions of Python and the remaining dependencies were not saved in the original run; this installation line is a compatible starting specification, not a recovered lockfile.

### 2. Configure the data location

In original cell 1, edit:

```python
DRIVE_PATH = "/content/drive/MyDrive/Dataset"
```

This should point to a folder containing the extracted data, or an actual ZIP file. Authorize Drive mounting when Colab asks. Original cell 3 searches recursively for the HR folders and should report 800 training HR and 100 validation HR images. A KaggleHub alternative is in `DATA.md`.

### 3. Choose persistent output storage

The original code writes checkpoints and `results_topic17.csv` to the **current working directory**, normally `/content`. Merely mounting Drive does not redirect these outputs.

To preserve a new run, insert this optional cell after mounting Drive and before training:

```python
from pathlib import Path
import os

RUN_DIR = Path("/content/drive/MyDrive/SR_Project/outputs/reported_run")
RUN_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(RUN_DIR)
print("Outputs will be saved to:", Path.cwd())
```

Use a new run directory when you want to retain earlier checkpoints. The notebook uses fixed checkpoint names and can overwrite files from an earlier run in the same directory. This storage adjustment does not change the reported model or data configuration.

### 4. Execute the original cells in order

Cell numbers below refer to the **original notebook**, counting its first Drive cell as 1. Added setup cells do not change these references conceptually.

| Original cells | Action |
| --- | --- |
| 1-4 | Mount Drive, set parameters, create split and HR patches |
| 5-8 | Define degradations, generate test inputs, score bicubic, prepare training data |
| 9-10 | Define and inspect architectures |
| 11 | Train A and reload its best checkpoint |
| 12 | Train B and reload its best checkpoint |
| 13 | Plot the A/B training histories |
| 14 | Evaluate A/B on all five test conditions |
| 15 | Train C, evaluate C, display PSNR/SSIM and export CSV |
| 16-17 | Plot comparisons and show a qualitative test example |
| 18 | Optional interactive upload demo |

For reproduction, keep the original settings: `N_TRAIN_IMAGES=300`, `N_VAL_IMAGES=50`, `N_TEST_IMAGES=50`, `PATCHES_PER_IMAGE=16`, `EPOCHS=10`, `SCALE=2`, and `SEED=42`. Changing them creates a new experiment and requires updating the report's results.

Recorded training times were approximately 1.4 minutes for A, 10.6 for B and 11.0 for C, plus 95 seconds for HR preparation. These are observations from the uploaded notebook, not a runtime guarantee, and exclude some setup/evaluation work.

## Recorded results

Values below are copied from the notebook's saved tables. PSNR is shown to two decimals and SSIM to four. Higher is better for both metrics.

### PSNR (dB)

| Condition | Bicubic | A: SRCNN | B: Residual clean | C: Residual random |
| --- | ---: | ---: | ---: | ---: |
| Clean | 29.75 | 31.37 | **32.21** | 30.47 |
| Blur, sigma 1.5 | 26.30 | 26.73 | 26.73 | **27.17** |
| Noise, sigma 10/255 | 26.16 | 25.62 | 25.36 | **28.55** |
| JPEG, quality 30 | 25.75 | 25.62 | 25.57 | **26.10** |
| Blur + noise + JPEG | 25.62 | 25.74 | 25.70 | **26.34** |

### SSIM

| Condition | Bicubic | A: SRCNN | B: Residual clean | C: Residual random |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.8651 | 0.8976 | **0.9098** | 0.8669 |
| Blur | 0.7394 | 0.7602 | **0.7611** | 0.7564 |
| Noise | 0.6633 | 0.6234 | 0.6040 | **0.8082** |
| JPEG | 0.7114 | 0.7034 | 0.7028 | **0.7358** |
| Blur + noise + JPEG | 0.7005 | 0.7008 | 0.6997 | **0.7317** |

C improves noise-condition PSNR by 2.39 dB over bicubic and 3.19 dB over B. It loses 1.74 dB to B on clean inputs. C leads PSNR in all four degraded conditions, but B has higher SSIM in the blur-only condition. These differences are calculated from the rounded displayed tables.

Metrics use RGB float images in [0,1], prediction clipping, TensorFlow PSNR/SSIM with `max_val=1.0`, and a mean over 50 test crops. There is no explicit border-shaving step or luminance-only conversion. Do not directly compare these numbers with published full-image, Y-channel or border-cropped benchmark results.

## Output files and preservation

| Generated file | Meaning |
| --- | --- |
| `SRCNN.weights.h5` | Best validation-loss weights for A |
| `ResidualSR_clean.weights.h5` | Best validation-loss weights for B |
| `ResidualSR_random.weights.h5` | Best validation-loss weights for C |
| `results_topic17.csv` | 20 rows: four methods x five test conditions; PSNR and SSIM |
| `<stem>_x2_bicubic.png`, `<stem>_x2_A.png`, `<stem>_x2_B.png`, `<stem>_x2_C.png` | Demo outputs from cell 18 |

The weight files do not contain the dataset, model-building source or full epoch history. The notebook uses `save_weights_only=True` and has no exact interrupted-training resume implementation. The restoration procedure below rebuilds models for inference; it does not restore the original epoch, optimizer and random-state context for continuing training.

After cell 15, optionally save the in-memory histories and environment with:

```python
import json
import subprocess
import sys
from pathlib import Path

Path("histories.json").write_text(json.dumps(histories, indent=2))
Path("requirements-resolved.txt").write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
)
```

These files use the current working directory. Save the notebook itself to Drive or download its latest `.ipynb` to preserve outputs. Runtime variables and files under `/content` do not persist after the runtime is deleted.

## Evaluate saved weights without training again

This requires the three actual weight files; they cannot be reconstructed from screenshots or metric tables.

1. In a fresh runtime, run original cells 1-10 to rebuild preprocessing, test inputs, baseline results and architecture functions.
2. Skip training cells 11-13 and restore models with the cell below.
3. Run original cell 14 once to define `test_model`/`show_table` and evaluate A/B.
4. Skip original cell 15, which retrains C. Run the second cell below instead, then run cells 16-18 as needed.

```python
from pathlib import Path

weights_dir = Path("/content/drive/MyDrive/SR_Project/outputs/reported_run")
model_A = build_srcnn()
model_B = build_residual_sr(name="ResidualSR_clean")
model_C = build_residual_sr(name="ResidualSR_random")
for model, filename in [
    (model_A, "SRCNN.weights.h5"),
    (model_B, "ResidualSR_clean.weights.h5"),
    (model_C, "ResidualSR_random.weights.h5"),
]:
    model.load_weights(str(weights_dir / filename))
```

After original cell 14:

```python
test_model(model_C, "C: Residual (random degr.)")
table_psnr = show_table()
table_ssim = pd.DataFrame(results).pivot_table(
    index="condition", columns="model", values="SSIM", sort=False
).round(4)
display(table_psnr)
display(table_ssim)
pd.DataFrame(results).to_csv(weights_dir / "results_topic17_recomputed.csv", index=False)
```

Evaluation appends to the global `results` list. Repeating evaluation cells without resetting it adds duplicate rows; `pivot_table` can silently average them. For a clean repeat, rerun original cell 7 to recreate the bicubic baseline and then evaluate A, B and C exactly once.

## Interactive demo

Run cell 18 after all three models are loaded or trained.

- `TREAT_AS_HR=False`: upload an LR/degraded image; the notebook enlarges it by x2 and shows Bicubic/A/B/C. There is no ground truth, so no valid reference PSNR/SSIM is available.
- `TREAT_AS_HR=True`: upload an HR image; the selected synthetic `CONDITION` is applied before restoration. The demo displays reference PSNR.
- `MAX_SIDE=800`: the uploaded image is reduced first when its longer side exceeds 800 pixels. In LR mode, the resulting output can reach 1,600 pixels on that side; this is not a guarantee against GPU memory exhaustion.

The demo saves four PNGs to the current working directory. In HR simulation mode, the saved `_x2_` tag describes the restoration scale; the restored output matches the preprocessed HR size.

## Limitations and troubleshooting

| Issue | Explanation / action |
| --- | --- |
| No HR folder found | Check `DRIVE_PATH` and the printed recursive folder inventory. Point to the real extracted root. |
| ZIP extraction appears incomplete | The existing `/content/div2k` directory can suppress extraction. Verify the HR counts or extract into a fresh location. |
| GPU list is empty | Select a GPU runtime and rerun setup. GPU availability varies. |
| Outputs disappear after disconnection | They were likely under `/content`; use the persistent output cell before a new run. |
| Results differ slightly | Seeds alone do not guarantee deterministic parallel TensorFlow execution; dependency versions and cell order also matter. |
| Very large demo image fails | Reduce `MAX_SIDE`; the model processes feature maps at the enlarged resolution. |
| C smooths an already clean image | Random-degradation training trades some clean-detail fidelity for robustness. |

Only one seed and one run per model are reported. Validation pairs are stochastic: A/B use random flipping, while C also resamples degradations. C's test degradation families fall within its training family, and all reference measurements are synthetic, central-crop measurements. There is no multi-seed uncertainty estimate, exhaustive degradation ablation or quantified real-world benchmark.

The report and documentation were prepared from the submitted notebook's code and stored outputs; model training was not rerun to create these files.

## References and submission details

The report contains the full reference list. Core sources include [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [SRCNN](https://arxiv.org/abs/1501.00092), [residual learning](https://arxiv.org/abs/1512.03385), and [Real-ESRGAN](https://arxiv.org/abs/2107.10833).

Complete the group/project identifiers and the report's member contribution table before submission. The supplied naming convention is `GroupID_ProjectID_Report.pdf` for the report and `DL2026-GroupID-ProjectID` for the repository. Personal details and contribution claims have deliberately been left for the group to complete.
