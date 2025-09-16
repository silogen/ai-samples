# MONAI-SwinUNETR training on AMD MI300X

Training of the MONAI [SwinUNETR](https://arxiv.org/abs/2201.01266) model on the [NSCLCL-Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) dataset for lung cancer tumor segmentation.

## Dataset

The [nsclc-radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) is loaded using the MONAI [TciaDataset](https://github.com/Project-MONAI/MONAI/blob/b58e883c887e0f99d382807550654c44d94f47bd/monai/apps/datasets.py#L404)

If running the code for the first time, add the flag `--download_data` to the main command in the [docker-compose.yml](docker-compose.yml) file. E.g.

```yaml
    command: ["/bin/bash",  "-c", "cd src/ && python main.py --json_list=data.json --download_data --data_root_dir=/workspace/data/datasets ...
```

Downloading the dataset takes a while, but only needs to be done once. After downloading it, it will be saved to the `./data/datasets` directory, which is mounted to the docker image.
That way you can remove the `--download_data` and skip downloading it again.

## Instructions to choosing the GPU to run on
 1. check with `amd-smi process` which GPU is free
 2. check with `rocm-smi` what is the node id of the free GPU (Note: node id is not the same as the device id and is displayed in the second column of the rocm-smi output)
 3. If say, the node id 2 gpu is free, the device to be added to docker run is given by `cat /sys/class/kfd/kfd/topology/nodes/2/properties | grep drm_render_minor`
 4. Add the GPU id to the `devices:` section on the [docker-compose.yml](docker-compose.yml). E.g. /dev/dri/renderD136


## Source code

The code is based on the original [MONAI scripts](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR), adapted with the following key modifications:

- Updating deprecated packages like MONAI and nibabel.
- Replacing outdated code references.
- Incorporating specific data and data loading functions for the MONAI TciaDataset dataset (NSCLC-Radiomics).
- Profiling and optimization

### Project Structure

```
life-science-swinunetr/
├── docker-compose.yml          # Docker Compose configuration for containerized training
├── Dockerfile                  # Docker image definition with ROCm/PyTorch base
├── requirements.txt            # Core Python dependencies for training
├── requirements-dev.txt        # Development dependencies for profiling and analysis
├── README.md                   # Project documentation and usage instructions
└── src/                        # Main source code directory
    ├── __init__.py             # Python package initialization
    ├── main.py                 # Main training script with argument parsing and orchestration
    ├── trainer.py              # Core training loop with epoch management and validation
    ├── test.py                 # Model evaluation and testing script
    ├── optimizers/             # Custom optimizer implementations
    │   ├── __init__.py         # Package initialization
    │   └── lr_scheduler.py     # Learning rate scheduling utilities (LinearWarmupCosineAnnealingLR)
    └── utils/                  # Utility functions and helpers
        ├── __init__.py         # Package initialization
        ├── data_utils.py       # Data loading, preprocessing, and dataset configuration
        ├── summarize_trace.py  # PyTorch profiler trace analysis and summarization
        └── utils.py            # General utilities (metrics, logging, image processing)
```


## Run training

Single GPU:

```bash
docker compose -f docker-compose.yml up --build -d
```

Check the logs:

```bash
docker compose -f docker-compose.yml logs -f
```

Stop the container:

```bash
docker compose -f docker-compose.yml down
```

## Tensorboard

Run `tensorboard --logdir <logdir>` to see the tensorboard logs.


## Optimizations

Optimizations can be enabled/disabled from the `environment:` section of the [dpcker-compose.yml](docker-compose.yml) file.

### MIOPEN

Performance improvement of >5x gain in the model fwd + bwd pass, and >25% GPU memory efficiency. The following environment variables and ROCm base image must be used: (already enabled by default)

```
# Dockerfile
FROM rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0 AS base

# Dockerfile or docker-compose.yml
MIOPEN_FIND_MODE=1
MIOPEN_FIND_ENFORCE=3
```

To reproduce the MIOPEN pre-optimization results, use the following ROCm base image, and remove or unset the `MIOPEN_FIND_MODE`/`MIOPEN_FIND_ENFORCE` environment variables:

```
# Dockerfile
FROM rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.4.1 AS base

# Dockerfile or docker-compose.yml
# Comment-out MIOPEN variables. Let MIOPEN use default behaviour.
# MIOPEN_FIND_MODE=1
# MIOPEN_FIND_ENFORCE=3
```

### Data Loading Enhancements

The data loading pipeline can often be a bottleneck in training deep learning models. We implemented two key optimizations to mitigate this:

- **Increased Number of Workers:** Utilizing a sufficient number of workers (around 32) for the data loader helped to reduce the time spent waiting for data.
- **Persistent Workers:** Setting `persistent_workers=True` in the PyTorch DataLoader caches the data loaders in memory, avoiding re-instantiation between epochs and saving approximately 14 seconds per epoch.

### Investigated Optimizations with Limited Impact

Not all attempted optimizations yielded significant improvements for this specific workload. These included:

- **PyTorch Compile:** Using torch.compile with and without max-autotune did not provide additional speedups beyond what MIOpen's auto-tuning delivered.
- **TunableOps:** This feature offered only minor gains (less than 1%) while adding a considerable amount of tuning time.
- **Mixed Precision Datatype:** Changing the automatic mixed precision (AMP) datatype from float16 to bfloat16 resulted in worse performance, likely due to the lack of optimized MIOpen kernels for bfloat16.

## Profiling

To run with the PyTorch profiler, pass the `--profile` option to `main.py`.
The schedule can be specified using `--profiler_schedule=<WAIT>,<WARMUP>,<ACTIVE>,<REPEAT>`, 
e.g. `--profiler_schedule=1,1,1,1`.
Refer to the [PyTorch Profiler documentation](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs) for details on what each field means.

Profiling results will be available under `./runs/<LOGDIR>/<RUNID>/profiler`. 
For each cycle of the profiler, 2 files will be available:

* Trace file: `<PROF_STEP>-chrome_trace.json.gz`
* Trace summary with key averages: `<PROF_STEP>-trace_summary.txt`

Additionally, memory is also recorded up until the end of the first cycle of the profiler and dumped as the pickle file `memory.pickle`.

The trace can be visualized in <https://ui.perfetto.dev/>.
The memory profile can be visualized in <https://docs.pytorch.org/memory_viz>.

Note that the trace files can be become quite large (Gb+) when recording many epochs/steps.

### Summarizing the trace

The [summarize_trace.py](src/utils/summarize_trace.py) script can be used to get timing statistics of key sections. 
Currently it reports on:

1. the dataloader first load in an epoch;
2. the dataloader batch load per step;
3. the forward pass (first-to-last kernel/GPU time);
4. the backward pass (first-to-last kernel/GPU time);
5. total step time.

Usage:

Install the dev requirements with `pip install -r requirements-dev.txt`. Run

```bash
python src/utils/summarize_trace.py PATH/TO/TRACE_FILE
```

The trace must be a JSON file as exported by the PyTorch profiler, either uncompressed or compressed (`.json.zip` or `.json.gz`).
