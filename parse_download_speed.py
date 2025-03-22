import argparse
import re
import statistics
import numpy as np
import matplotlib.pyplot as plt


def smooth_data(data, window_size=5):
    """Smooth the data using a moving average with the specified window size."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def main():
    parser = argparse.ArgumentParser(
        description="Process a log file to extract download speeds."
    )
    parser.add_argument(
        "--log_file_path", type=str, help="Path to the log file", required=True
    )
    parser.add_argument(
        "--save_image", action="store_true", help="Flag to save the plot as an image"
    )
    args = parser.parse_args()

    speeds = []
    times = []
    with open(args.log_file_path, "r") as file:
        for line in file:
            # Process only lines that contain "chunk-" and ".bin"
            if "chunk-" in line and ".bin" in line:
                speed_match = re.search(r"at ([\d\.]+) MB/s", line)
                if speed_match:
                    speeds.append(float(speed_match.group(1)))
                time_match = re.search(r"in ([\d\.]+)s at", line)
                if time_match:
                    times.append(float(time_match.group(1)))

    if speeds:
        print("Number of samples:", len(speeds))
        # Remove outliers using the IQR method
        # q1, q3 = np.percentile(speeds, [25, 75])
        # iqr = q3 - q1
        # lower_bound = q1 - 1.5 * iqr
        # upper_bound = q3 + 1.5 * iqr
        # filtered_speeds = [s for s in speeds if lower_bound <= s <= upper_bound]

        # if len(filtered_speeds) == 0:
        #     print("All speeds considered outliers!")
        #     return

        filtered_speeds = speeds
        print("Avg time: {:.2f} s".format(statistics.mean(times)))
        print("Median time: {:.2f} s".format(statistics.median(times)))
        print("Min speed: {:.2f} MB/s".format(min(filtered_speeds)))
        print("Avg speed: {:.2f} MB/s".format(statistics.mean(filtered_speeds)))
        print("Median speed: {:.2f} MB/s".format(statistics.median(filtered_speeds)))
        print("Max speed: {:.2f} MB/s".format(max(filtered_speeds)))
        try:
            mode_value = statistics.mode(filtered_speeds)
            print("Mode speed: {:.2f} MB/s".format(mode_value))
        except statistics.StatisticsError:
            modes = statistics.multimode(filtered_speeds)
            print("Mode speeds: " + ", ".join("{:.2f} MB/s".format(m) for m in modes))

        # Smooth the filtered speeds using a moving average
        smoothed_speeds = smooth_data(filtered_speeds, window_size=5)

        if args.save_image:
            # Plot the smoothed line graph of filtered speeds
            plt.figure(figsize=(10, 6))
            plt.plot(smoothed_speeds, marker="o", linestyle="-", color="blue")
            plt.title("Smoothed Download Speeds Line Graph")
            plt.xlabel("Entry Index")
            plt.ylabel("Speed (MB/s)")
            plt.grid(True)

            # Save the figure. Create an image file with added ".img" extension.
            out_img_path = args.log_file_path + ".jpg"
            plt.savefig(out_img_path)
            print("Diagram saved to:", out_img_path)
    else:
        print("No download speed entries found.")


if __name__ == "__main__":
    main()
