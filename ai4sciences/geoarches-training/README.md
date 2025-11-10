# Supporting Code for blog

Following code samples are supporting the rocm blog about weather model training on AMD Instinct. For more context refer to the blog or [geoarches documentation.](https://geoarches.readthedocs.io/en/latest/archesweather/)

These scripts closely reproduce the end-to-end training pipeline for weather models described in [ArchesWeather & ArchesWeatherGen](https://arxiv.org/abs/2412.12971) paper on AMD Instinct™ MI300X GPUs.

## Build and run the container

``` bash
docker build -t pytorch_training_geoarches:latest .

docker run -it --rm \
--device=/dev/kfd \
--device=/dev/dri \
--group-add video \
--name geoarches_training \
--shm-size=16g \
pytorch_training_geoarches bash
```

## Execute the training pipeline
Run the numbered scripts in order (numeration is included for simplicity):

``` bash
bash 1_download_assets.sh
...
bash 9_plot.sh
```
