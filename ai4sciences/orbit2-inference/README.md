# Support code for blog: https://rocm.blogs.amd.com/artificial-intelligence/orbit2-inference/README.html

## Quickstart


## Preparation
- Copy the data from OLCF to `OLCF_DATA_PATH` following the instructions as described in the blog and the link. `OLCF_DATA_PATH` should contain at least two directories `ERA5_IMERG_input` and `ERA5_IMERG_output`. 
- If you want to run the gencast step: 
    - Edit the `./env_fle` with your CDS API key if you want to run the gen 
    - In some location, download the grib outputs from GenCast that you wish to downscale. Let us assume these files are `gencast-0.25.grib` and `gencast-1.0.grib`

- Build the Docker image from the DockerFile
```bash
bash build.sh # clone ORBIT-2 repo and build Docker image
              # → produces: orbit2 Docker image, ORBIT-2/ directory
```

-  Edit the `setup.sh` file as described in the blog to mount the `OLCF_DATA_PATH` to `/workspace/Data`.
- Start the docker container and prepare for inference:
```bash
bash setup.sh              # launch container and configure environment
                           # → produces: running container with /workspace mounted
                           # Download checkpoints and configs in /workspace/checkpoints/
```
- Enter the container:
```
docker exec -it orbit2 bash
```
and  inside the container:
- Edit the yaml file  (`/workspace/checkpoints/global-finetune/global-9.5m_precipitation.yaml` ) to  ensure that the data paths are correct as explained in the blog.
- Now run the inference :

```bash
bash run_infer.sh

```

- Followed by a plotting script and evaluation of metrics script.

```bash
python plot_comparison.py  --pred-dir results/
python evaluate_metrics.py --pred-dir results/
```
----------------------------------------------------------------
Let us assume you have the files `gencast_1.0.grib` and `gencast_0.25.grib` in some directory. Edit the `run_gencast.sh` file to set `datapath` to that directory. Also setup the lowres and highres directory paths to the inputs and outputs of ORBIT2 as setup before.  
Also make sure that the `env_file` has the right information.

Then you can run the file from inside the container after sourcing the `env_file`:
```bash 
source env_file            # Sets up CDS authorization to obtain data from ERA5 when needed.
bash run_gencast.sh        # convert GenCast GRIB outputs to NPZ format compatible with ORBIT2, obtaining missing variables from ERA5 data
                           # → produces: low-res and high-res NPZ files consumed by run_infer.sh

                           # run ORBIT-2 downscaling inference using `run_infer.sh` 
                           # → produces: prediction .npy files in results/
```

