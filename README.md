# LitData Benchmarking Starter Kit

This starter kit will guide you through the process of benchmarking LitData in **Lightning Studio**. 
Follow the steps below to get started.
> This starter kit has been prepared with reference from the [LitData Benchmarking Guide](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries?view=org&section=featured).

## Step 1: Download Data (Optional)

You can download the data from Kaggle if you prefer. However, there is already imagenet data available in `/teamspace/s3_connections/imagenet-1m-template/raw/` that you can use.

### Download from Kaggle
1. Add the `kaggle.json` file to `~/.kaggle/` directory.
3. Run the script `sh download.sh` to download the data from Kaggle.

**ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 dataset**.  

### Breakdown of ImageNet (ILSVRC 2012) dataset:
- **Training images**: **1,281,167**
- **Validation images**: **50,000** (50 per class)
- **Test images**: **100,000** (100 per class)
- **Total images**: **1,431,167**


## Step 2: Optimize the Data

Once you have the data, the next step is to optimize it for benchmarking.
```sh
python run_optimize.py
```

## Step 3: Stream the Data to Record the Benchmark

Finally, stream the optimized data to record the benchmark.
```sh
python run_benchmark.py # executes in a separate A10G studio
```
or
```sh
python stream_imagenet.py # runs in the current studio
```

Follow these steps to successfully benchmark LitData. Happy benchmarking!
