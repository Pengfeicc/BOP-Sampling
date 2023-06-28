# BOP-Sampling
Synthetic Dataset Generator using Blenderproc

Modified from some examples in https://github.com/DLR-RM/BlenderProc/tree/main/examples/datasets/bop_challenge

## Installation

### Via pip

The simplest way to install blenderproc is via pip:

```bash
pip install blenderproc
```
## Git clone this repository
```bash
git clone https://github.com/Pengfeicc/BOP-Sampling.git
```
Before you run the main_faps_bop.py, you need to prepare you own RGB-D dataset or download from open-source dataset.

## Usage
```bash
blenderproc run main_faps_bop.py bop_data <bop_dataset_name / your dataset name> images/xxx.jpg output
```
