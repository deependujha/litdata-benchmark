# LitData Benchmarking Starter Kit

This starter kit will guide you through the process of benchmarking LitData in **Lightning Studio**. 
Follow the steps below to get started.
> This starter kit has been prepared with reference from the [LitData Benchmarking Guide](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries?view=org&section=featured).

## Prerequisites

1. Clone the GitHub repository:
    ```sh
    git clone https://github.com/bhimrazy/litdata-benchmark
    ```

2. Navigate to the project directory:
    ```sh
    cd litdata-benchmark
    ```

3. Install the required dependencies:
    ```sh
    pip install -U -r requirements.txt
    ```

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

python stream_imagenet.py | tee logs/benchmark-run-0.log
```

Note: The log files follow a specific naming convention to help identify the details of each benchmark run. The format is as follows:

```
benchmark-MM-DD-v<version>-<downloader>-<dtype>-run-<run_number>.log
```

Where:
- `MM-DD` is the month and day of the benchmark run.
- `<version>` is the version of the litdata used to benchmark.
- `<downloader>` is the tool used for downloading the data (e.g., `s5cmd` or `boto3`)..
- `<dtype>` indicates the data type used (e.g., `f32` or `f16`).
- `<run_number>` is the sequential number of the run.

Example:
```
benchmark-03-15-v0.2.41-s5cmd-f32-run-0.log
````

## Benchmark Hugging Face Datasets

> **Note**: This section is currently under active development and may be updated frequently.

To benchmark Hugging Face datasets, you can use the following command:

```sh
DATASET=0 SHUFFLE=1 PRELOAD=1 LOW_MEMORY=0 python stream_hf_dataset.py
```

### Explanation of Parameters:
- `DATASET`: Specifies the dataset to be used. Replace `0` with the desired dataset index or identifier.
- `SHUFFLE`: Set to `1` to enable shuffling of the dataset, or `0` to disable it.
- `PRELOAD`: Set to `1` to enable preloading the chunk, or `0` to disable it.
- `LOW_MEMORY`: Set to `1` to enable low-memory mode, which reduces memory usage at the cost of performance, or `0` to disable it.


Follow these steps to successfully benchmark LitData. Happy benchmarking! 🎉
