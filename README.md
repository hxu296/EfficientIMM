# EfficintIMM

EfficientIMM builds on top of the Ripples framework. This repository builds from the Ripples commit version `0a9f3e7c450a7` (Sep 11, 2023) from the official Ripples repository. For now, only single-node version is supported. This README will explain how to build EfficientIMM on Perlmutter and run the experiments on SNAP datasets.

## Abstract
EfficientIMM is an optimized implementation of influence maximization algorithms based on the IMM (Influence Maximization via Martingales) algorithm. It enhances performance for computing influence maximization on large-scale networks using the Independent Cascade (IC) and Linear Threshold (LT) diffusion models. Designed for multi-core shared memory systems, EfficientIMM offers significant speedups over the original Ripples implementation when processing SNAP datasets. This repository provides the source code, build instructions, and scripts necessary to reproduce our experiments on SNAP datasets. 

## Description
**Check-list (artifact meta information).**
- **Algorithm**: Influence Maximization
- **Program**: EfficientIMM
- **Compilation**: C++17
- **Binary**: Binary not included
- **Dataset**: SNAP datasets, generated weights for IC and LT diffusion models
- **Run-time environment**: Our artifact has been developed and tested on Linux environment. The main software dependencies are anaconda and numatcl
- **Hardware**: Our artifact has been developed and tested on Perlmutter, dual-socket AMD EPYC 7713 64-Core Processor for experiments presented in our original paper. Similar hardware should result in similar speedup results
- **Output:** The experiments generate JSON files containing execution logs and the top `k` influential nodes (seed sets), which are stored in the `strong-scaling-logs-*` directories within the project directory. These files are used to evaluate execution times for EfficientIMM and Ripples over SNAP datasets. Running `extract_results.py` will generate a summary performance breakdown and the best performance comparison in the `results` directory in the project root, similar to `Table III` of the paper
- **Publicly available?:** Yes

**Hardware dependencies.** Our artifact has been developed and tested on Perlmutter, dual-socket AMD EPYC 7713 64-Core Processor for experiments presented in our original paper. Similar hardware should result in similar speedup results. It is recommended to run the experiments on a machine with at least 64 cores and 128 GB of memory.

**Software dependencies.** Our artifact has been developed and tested on Linux environment. The main software dependencies are anaconda and numatcl. We will download specific conan, cmake, and gcc versions using conda, and build other dependencies using conan.

**Datasets.** We provide the SNAP datasets, generated weights for IC and LT diffusion models, publicly available at [(link)](https://drive.google.com/file/d/1CRNC2NjSQ5B1_Jngbg_G4uCZzgWbG83Q/view?usp=sharing)

To programmatically download the datasets, you can run the following commands. Please first setup a didicated conda environment. You can do this by running the following commands:
```
conda create -n efficientimm python=3.9
conda activate efficientimm
```

Then, run the following commands to download the datasets:
```
bash download_dataset.sh
```

This will create the following directory structure under the project root. These are 7 datasets for each of the IC and LT models. The Twitter 7 dataset is not included in the above dataset, as it is too computationally intensive too run especially given the 8 hour artifact evaluation time budget. But all other datasets from the paper are included:

- `test-data/`:
  - `as-skitter.IC.tsv`: Skitter dataset, IC model
  - `as-skitter.LT.tsv`: Skitter dataset, LT model
  - `com-amazon.ungraph.IC.tsv`: Amazon dataset, IC model
  - `com-amazon.ungraph.LT.tsv`: Amazon dataset, LT model
  - `com-dblp.ungraph.IC.tsv`: DBLP dataset, IC model
  - `com-dblp.ungraph.LT.tsv`: DBLP dataset, LT model
  - `com-lj.ungraph.IC.tsv`: LiveJournal dataset, IC model
  - `com-lj.ungraph.LT.tsv`: LiveJournal dataset, LT model
  - `com-youtube.ungraph.IC.tsv`: YouTube dataset, IC model
  - `com-youtube.ungraph.LT.tsv`: YouTube dataset, LT model
  - `soc-pokec-relationship.IC.tsv`: Pokec dataset, IC model
  - `soc-pokec-relationship.LT.tsv`: Pokec dataset, LT model
  - `web-Google.IC.tsv`: Google dataset, IC model
  - `web-Google.LT.tsv`: Google dataset, LT model

## Build Instructions

The following commands will build EfficientIMM at `$PROJECT_DIR/build/release/src_efficient_imm/tools/imm`. And build Ripples at `$PROJECT_DIR/build/release/src_ripples/tools/imm`:

```
conda activate efficientimm # if not already activated
bash setup_conan.sh
bash run_build.sh
```

## Run SNAP dataset experiments

The following commands will run EfficientIMM and Ripples experiments, starting with 4 threads and continuing with powers of 2 until they reach the system limit. All experiments use k=50 and epsilon=0.5. On Perlmutter, Ripples experiments will take about 2 hours to finish, and EfficientIMM will take about 1 hour to finish.

```
bash run_efficient_imm.sh # takes around 2 hours to finish
bash run_ripples.sh # takes around 3 to 4 hours to finish
```

## Expected Results
After experiments finish, new directories will be created: 
- `strong-scaling-logs-ic-eimm`: Efficient IMM, IC model
- `strong-scaling-logs_lt-eimm`: Efficient IMM, LT model
- `strong-scaling-logs_ic-ripples`: Ripples, IC model
- `strong-scaling-logs_lt-ripples`: Ripples, LT model

Each JSON file in the directory will contain the raw experiment results. To summarize and compare results, run the following command to print a human readable speedup summary table to stdout. This will also generate a summary performance breakdown and the best performance comparison in the `results` directory in the project root, similar to `Table III` of the paper.

```
python3 extract_results.py
```

The CSV files in the `results` directory will contain the following columns:
- `Dataset`: Name of the dataset
- `Speedup`: Speedup of EfficientIMM over Ripples
- `EfficientIMM Time (s)`: Execution time of EfficientIMM in seconds
- `Ripples Time (s)`: Execution time of Ripples in seconds
- `Ripples Best #Threads`: Best performing number of threads for Ripples
- `EfficientIMM Best #Threads`: Best performing number of threads for EfficientIMM

