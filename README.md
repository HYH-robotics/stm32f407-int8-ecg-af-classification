# Record-Level AF Classification on STM32F407

This repository accompanies a study of record-level atrial fibrillation (AF) classification using an INT8 convolutional neural network on an STM32F407 microcontroller. It includes the study partition, training and evaluation code, selected model parameters, firmware projects, and board timing and resource data.

## Repository structure

| Path | Contents |
| --- | --- |
| `data/split_manifest.csv` | Study-specific record partitions |
| `scripts/` | Data preparation, training, quantization, primary evaluation, R–R features, and board summary |
| `model/` | Selected FP32 weights, INT8 parameters, quantization metadata, and configuration |
| `firmware/` | Original C and CMSIS-NN projects and model headers |
| `board/` | Cycle counts and memory resource summaries |
| `requirements.txt` | Recorded Python dependencies |

## 1. Get started

```sh
git clone https://github.com/HYH-robotics/stm32f407-int8-ecg-af-classification.git
cd stm32f407-int8-ecg-af-classification
```

## 2. Software requirements

A version check of the interpreter path later used for training recorded the following versions before the training run. The training process did not independently print the package versions; a complete training-time environment snapshot is unavailable.

| Software | Recorded version |
| --- | --- |
| Python | 3.13.14 |
| NumPy | 2.4.4 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| PyTorch (CPU) | 2.13.0+cpu |
| PyYAML | 6.0.3 |

For Windows, using an existing Python 3.13 installation:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

On Linux or macOS, create and activate a Python 3.13 environment with `python3.13 -m venv .venv` and `source .venv/bin/activate`, then use the same `python -m pip install` command. The [official PyTorch CPU index](https://download.pytorch.org/whl/cpu/torch/) lists a 2.13.0+cpu wheel for CPython 3.13 on Windows x86-64 and Linux x86-64. Package availability differs by platform; the pinned installation and the complete workflow have **not** been tested in a clean environment. The pins are historical records, not a guarantee of bitwise reproduction.

For the recorded single-thread configuration, set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to `1` before launching Python. The launcher sets the training seed to `20260921`.

## 3. Dataset preparation

Obtain the [PhysioNet/Computing in Cardiology Challenge 2017 dataset](https://physionet.org/content/challenge-2017/1.0.0/) separately. The repository does not distribute ECG waveforms or reference labels. The original record partition is in `data/split_manifest.csv`:

| Partition | Records |
| --- | ---: |
| Training | 5,774 |
| Development | 825 |
| Pilot | 825 |
| Primary test (`locked_test`) | 825 |
| Total | 8,249 |

`locked_test` is the study-specific primary test split, not the official challenge test set. Noisy records are excluded from these partitions.

Study records identify [REFERENCE-v3.csv](https://physionet.org/files/challenge-2017/1.0.0/training/REFERENCE-v3.csv) as the reference version. The official file obtained during this documentation check had SHA-256 `0413324b869b168ba1878abfec96589bda1e9aa5e9fcb82ce9a15afa4c644b0b`. After removing directory prefixes from its record identifiers, its binary-label mapping matched the checksum expected by the preparation script. The checksum verifies the study's binary A-versus-N/O mapping, not every original multiclass label or the provenance of each waveform.

Place the required `.mat` files together in a local directory, with each named by its record identifier, for example `A00001.mat`. If the downloaded dataset is arranged in subdirectories, organize the required files into a flat directory while retaining their original filenames. The script uses this layout and does not search subdirectories. Keep ECG files and reference labels outside the repository.

```sh
python scripts/prepare_dataset.py --ecg-dir "/path/to/ecg" --reference "/path/to/REFERENCE-v3.csv"
```

The script reads the reference labels for all eligible partitions to verify the mapping and partition counts, but reads waveform files only for training and development. It writes local manifests under `data/ecg/views/train_dev_v2/`: `train_development_manifest.yaml` and `train_development_split.yaml`. These portable manifests are not byte-identical copies of the historical manifests.

## 4. Model training

The selected FP32 and INT8 parameters are already provided; training is unnecessary for the selected-model evaluation in section 6. To generate new training runs for the development grid, first prepare the manifests, then run:

```sh
python scripts/run_development_grid.py
```

The launcher runs 12 configurations across two window/hop pairs (10/5 and 5/2.5 seconds), two preprocessing paths, and Tiny, Small, and Medium models, using seed `20260921`. Each run writes `runs/<configuration>/best_checkpoint.pt` and `run_metadata.json`. The trainer refuses to overwrite an existing run directory. These are new training runs; bitwise reproduction of the historical checkpoints is not guaranteed. Training and development waveform processing are restricted to their respective partitions.

The selected configuration is in `model/selected_config.yaml`: Small, 5-second window, 2.5-second hop, robust normalization, and INT8 inference.

## 5. INT8 quantization

After all 12 training runs have created their checkpoints, run:

```sh
python scripts/quantize_evaluate_p6_int8.py
```

The script reads the training/development manifests and the `best_checkpoint.pt` for each configuration. It writes quantization artifacts under `artifacts/p6_int8/` and development predictions under `results/raw/p6_int8_development_predictions/`. These outputs are local and ignored by Git. This command performs new quantization and development evaluation; it is not needed to use the supplied selected-model files:

- `model/fp32_weights.npz`
- `model/int8_parameters.npz`
- `model/quantization.json`

## 6. Record-level evaluation

With the supplied selected weights, compatible reference labels, the split manifest and the required ECG files available locally:

```sh
python scripts/evaluate_primary.py --ecg-dir "/path/to/ecg" --reference "/path/to/REFERENCE-v3.csv"
```

This portable adaptation loads the selected FP32 and INT8 parameters, computes record-level AP and AUROC, applies the fixed threshold for confusion counts, sensitivity, specificity and F1, and reports the paired INT8-minus-FP32 AP bootstrap interval. It also fits the fixed R–R feature comparator on training records and reports its AP, AUROC and paired AP contrast. Aggregate JSON is printed to standard output. Running the command accesses the study's primary-test waveforms and labels; it is a new evaluation, not the historical execution. A complete real-data rerun has not been performed for this release.

## 7. STM32 firmware

The projects target STM32F407ZGT6 / Cortex-M4F at a nominal 168 MHz using Keil MDK 5.35, ARMCC 5.06 update 7 and optimization level O2. The original C project is `firmware/original_c/firmware.uvprojx`; the optimized project is `firmware/cmsis_nn/firmware.uvprojx`. Model headers are included.

Install the separately licensed Keil compiler and [STM32F4xx DFP](https://www.keil.arm.com/packs/stm32f4xx_dfp-keil/) device pack (recorded version 2.15.0), and obtain official [CMSIS 5.8.0](https://github.com/ARM-software/CMSIS_5/tree/5.8.0) and [CMSIS 5.9.0](https://github.com/ARM-software/CMSIS_5/tree/5.9.0) sources. The project files reference this relative layout:

```text
firmware/
  common/
  dependencies/
    STM32F4xx_DFP/Drivers/CMSIS/Device/ST/STM32F4xx/Include/
    CMSIS_5_5.8.0/CMSIS/Core/Include/
    CMSIS_5_5.9.0/CMSIS/Core/Include/
    CMSIS_5_5.9.0/CMSIS/NN/Include/
    CMSIS_5_5.9.0/CMSIS/NN/Source/
    CMSIS_5_5.9.0/CMSIS/DSP/Include/
  original_c/firmware.uvprojx
  cmsis_nn/firmware.uvprojx
```

The CMSIS-NN project references four source files under `CMSIS/NN/Source/` and defines `ARM_MATH_DSP` and `CMSIS_NN_USE_SINGLE_ROUNDING`. If the installed vendor-pack layout differs, adjust the project paths to the corresponding official files. Open the relevant `.uvprojx` in Keil and build its target. These portable project templates have not been rebuilt in a clean environment; no board or energy measurement is performed by the repository scripts.

## 8. Board performance analysis

`board/` contains original C and CMSIS-NN cycle-count batches, model-scale cycle counts, and resource summaries. To summarize the existing timing data without ECG inputs:

```sh
python scripts/summarize_board.py
```

The script converts cycles at the nominal 168 MHz and prints sample count, mean, sample standard deviation, median, p95, p99, maximum and 2.5-second deadline misses as JSON. The same 1,000 input windows were replayed in each CMSIS comparison batch; the batches are not independent patients. Timing covers buffered preprocessing and inference, excluding acquisition and communication. Datasheet-based energy estimates in the paper are not board-level energy measurements.

## 9. Reproducibility scope

The repository provides the development grid, quantization scripts, a portable selected-model primary evaluation, firmware configuration and board timing data. Full Q15 and external-cohort analysis code is not included. The clean-environment installation, complete real-data execution and portable firmware rebuild have not been verified.

## 10. Data and software rights

The ECG dataset is distributed separately under the [PhysioNet dataset license](https://physionet.org/content/challenge-2017/view-license/1.0.0/). Third-party software is governed by its respective license. The authors have not yet specified a repository license for the study code and model parameters.
