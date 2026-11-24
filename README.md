# Record-level AF classification on STM32F407

This repository contains the study's training and evaluation code, selected model parameters, embedded firmware and board measurement results.

## Files

- `scripts/`: training, quantization, evaluation and board-summary code. The primary evaluation wrapper is a later portable adaptation; it is not the original execution script.
- `model/`: selected FP32 weights, INT8 parameters, quantization metadata and selected configuration.
- `firmware/`: original C and CMSIS-NN comparison source and project configurations. Vendor libraries and compiler are not included.
- `board/`: cycle counts and resource summaries for the reported board measurements. The same input sequence was replayed in each CMSIS comparison batch.

## Data and reproducibility limits

The ECG recordings are available from [PhysioNet/CinC 2017 v1.0.0](https://physionet.org/content/challenge-2017/1.0.0/) under the dataset's [ODC Attribution 1.0 license](https://physionet.org/content/challenge-2017/view-license/1.0.0/). The original study used 5,774 training, 825 development, 825 pilot and 825 primary-test records after excluding noisy recordings.

The original record-level partition list is provided in `data/split_manifest.csv`; the `locked_test` partition is the study’s primary test split, not the official challenge test set. `prepare_dataset.py` and `evaluate_primary.py` also require compatible reference labels and ECG files obtained separately. The scripts check that the resulting binary labels match the historical mapping. Earlier study records identify REFERENCE-v3 as the relevant label version; the exact download and file identity still require confirmation. The scripts have not been validated by a complete real-data rerun in a clean environment.

The included code does not cover the full Q15 and external-cohort analyses. No real-data rerun was performed when preparing this package. Generated waveforms, labels, individual predictions and local views should not be added to the repository.

## Environment and board

An interpreter version check before training recorded Python 3.13.14, NumPy 2.4.4, SciPy 1.18.0, scikit-learn 1.9.0, CPU PyTorch 2.13.0+cpu and PyYAML 6.0.3. The subsequent training command used the same interpreter path, but training did not independently print package versions. `requirements.txt` lists these recorded versions; installation in a clean environment has not been verified. CPU PyTorch installation may require the [official CPU wheel index](https://download.pytorch.org/whl/cpu).

The target was STM32F407ZGT6 at 168 MHz with Keil MDK 5.35 / ARMCC 5.06 update 7, O2. The project templates refer to separately obtained STM32F4xx DFP 2.15.0, CMSIS 5.8.0 for original C, and CMSIS 5.9.0 / CMSIS-NN 3.1.0 for the comparison. The CMSIS-NN project uses single rounding. The portable project templates have not been rebuilt in a clean environment. Board timings cover buffered preprocessing and inference and exclude acquisition and communication.

To summarize the included cycle counts without ECG data:

```sh
python scripts/summarize_board.py
```

## Rights and release

Dataset rights follow the linked PhysioNet license. The authors must confirm ownership and choose a license for the code and model before publication. Any third-party material must retain its applicable notices and license.
