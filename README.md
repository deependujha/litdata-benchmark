# LitData Benchmarking Starter Kit

This starter kit will guide you through the process of benchmarking LitData in **Lightning Studio**. 
Follow the steps below to get started.
> This starter kit has been prepared with reference from the [LitData Benchmarking Guide](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries?view=org&section=featured).

## Step 1: Download Data (Optional)

You can download the data from Kaggle if you prefer. However, there is already imagenet data available in `/teamspace/s3_connections/imagenet-1m-template/raw/` that you can use.

### Download from Kaggle
1. Add the `kaggle.json` file to `~/.kaggle/` directory.
3. Run the script `sh scripts/download.sh` to download the data from Kaggle.

**ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 dataset**.  

### Breakdown of ImageNet (ILSVRC 2012) dataset:
- **Training images**: **1,281,167**
- **Validation images**: **50,000** (50 per class)
- **Test images**: **100,000** (100 per class)
- **Total images**: **1,431,167**


## Step 2: Optimize the Data

Once you have the data, the next step is to optimize it for benchmarking.

1. Load the data into your environment.
2. Perform any necessary preprocessing steps (e.g., cleaning, normalization).
3. Save the optimized data for benchmarking.

## Step 3: Stream the Data to Record the Benchmark

Finally, stream the optimized data to record the benchmark.

1. Set up your benchmarking environment.
2. Stream the data to the benchmarking tool.
3. Record and analyze the benchmark results.

Follow these steps to successfully benchmark LitData. Happy benchmarking!