import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
DATAFILE = r"D:\Science and Music Double DCS Program\Science\Programming in Science\Project\ECG_DATA.csv" # Replace this with the filename from your computer
FS = 400 # Sampling frequency in Hz

def load_ecg(path):
    df = pd.read_csv(path)
    return df
path = DATAFILE

WINDOW_SIZE = 15 # For moving average calculation
def filter_signal(df, window = WINDOW_SIZE):
    kernel = np.ones(window) / window
    df['Filtered Voltage'] = (
        df.groupby('Patient ID')['Voltage']
        .transform(lambda x: np.convolve(x, kernel, mode='same'))
    )
    return df
    
def detect_r_peaks(df, threshold_percentage=0.5):
    # Creating new empty columns in the dataframe
    df["R-peak time"] = np.nan
    df["R-peak voltage"] = np.nan
    all_peaks = {}
    for patient_id, group in df.groupby('Patient ID'):
        group = group[group['Time'] > 0.2] # Excluding the first 0.2 seconds of data since it is not representative
        ecg = group['Voltage'].values
        time = group['Time'].values
        idx = group.index.values
        
        baseline = np.median(ecg) # Computing baseline for patient
        peak_height = np.max(ecg) - baseline # Computing peak height of ecg
        threshold = threshold_percentage * peak_height # Computing threshold for patient
        
        peak_times = [] # Initializing empty arrays to store peak information
        peak_voltages = []
        peak_indices = []
        
        for i in range(1, len(ecg) - 1):
            if (ecg[i] - baseline) > threshold and ecg[i] >= ecg[i-1] and ecg[i] > ecg[i+1]: # Condition for peak detection
                peak_times.append(float(time[i]))                                            # Making lists of the peak times, voltages, and indices
                peak_voltages.append(float(ecg[i]))
                peak_indices.append(idx[i])
        df.loc[peak_indices, "R-peak time"] = peak_times
        df.loc[peak_indices, "R-peak voltage"] = peak_voltages
        all_peaks[patient_id] = peak_times
    return df, all_peaks

def compute_metrics(all_peaks):
    rows = []
    metrics = {}
    for patient_id, peak_times in all_peaks.items():
        rr = np.diff(peak_times)
        for rr_value in rr:
            rows.append({"Patient ID": patient_id, "RR-Interval": rr_value})
        # Calculating BPM and HRV
        bpm = 60 * (len(peak_times) - 1) / (peak_times[-1] - peak_times[0])
        hrv = np.std(rr)
        metrics[patient_id] = {"BPM": bpm, "RR_intervals": rr.tolist(), "HRV": hrv}
    rr_df = pd.DataFrame(rows)
     # Add RR_n+1 for scatter plot
    rr_df['RR_n+1'] = rr_df.groupby("Patient ID")['RR-Interval'].shift(-1)
    return rr_df, metrics

def create_plots(df, rr_df):
    # Raw ECG Plot
    plt.figure(figsize=(14,8))
    sns.lineplot(data=df,  x= 'Time', y="Voltage", hue="Patient ID", palette = 'gist_rainbow')
    sns.hls_palette()
    plt.title("Raw ECG Signals")
    plt.savefig(r"D:\Science and Music Double DCS Program\Science\Programming in Science\Project\raw_ecg.png") # Replace this with the filename from your computer
    plt.xlim(0, 10)
    plt.show()
    plt.close()

    # Filtered ecg
    plt.figure(figsize=(14,8))
    sns.lineplot(data=df, x='Time', y="Filtered Voltage", hue="Patient ID", palette='gist_rainbow')
    plt.title("Filtered ECG Signals (Moving Average Smoothing)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.xlim(df['Time'].min(), df['Time'].min() + 10)  # safer limit
    plt.savefig(r"D:\Science and Music Double DCS Program\Science\Programming in Science\Project\filtered_ecg.png") # Replace this with the filename from your computer
    plt.show()
    plt.close()

    # RR Scatter plot 
    rr_df_clean = rr_df.dropna(subset=['RR_n+1'])  # drop NaNs
    plt.figure(figsize=(10,6))
    rr_df["RR_n+1"] = rr_df_clean.groupby("Patient ID")["RR-Interval"].shift(-1)
    sns.scatterplot(data=rr_df, x="RR-Interval", y="RR_n+1", hue="Patient ID", palette='gist_rainbow')
    plt.title("RR Interval Scatter Plot")
    plt.xlabel("RR Interval (s)")
    plt.ylabel("Next RR Interval (s)")
    plt.savefig(r"D:\Science and Music Double DCS Program\Science\Programming in Science\Project\rr_scatter.png")
    plt.show()
    plt.close()

    # HR histogram TODO "hr_hist.png"
    bpm_values = [info["BPM"] for info in metrics.values()]
    plt.figure(figsize=(10,6))
    sns.histplot(bpm_values, bins=10, color='skyblue', edgecolor='black')
    plt.title("Histogram of Heart Rate (BPM)")
    plt.xlabel("Beats Per Minute (BPM)")
    plt.ylabel("Frequency")
    plt.savefig(r"D:\Science and Music Double DCS Program\Science\Programming in Science\Project\hr_hist.png")
    plt.show()
    plt.close()

def make_animation(df):
    ecg_data = df.groupby("Time")["Filtered Voltage"].mean().reset_index()
    times = ecg_data["Time"].values
    voltage = ecg_data["Filtered Voltage"].values

    fig, ax = plt.subplots(figsize=(14,8))
    ax.set_facecolor('black')
    ax.grid(True)
    ax.set_xlim(min(times), max(times))
    ax.set_ylim(min(voltage), max(voltage) + 2)

    line, = ax.plot([], [], '-', color = 'lime', lw = 2)
    print(line)

    def animate(i):
        window = 400 # Sampling frequency
        start = max(0, i - window)
        x = times[:i]
        y = voltage[:i]
        line.set_data(x, y)
        ax.set_xlim(times[start], times[i])
        ax.set_title(f"Time: {times[i]:.2f} s")
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=len(times), interval = 1)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.show()
    anim.save(r"D:\Science and Music Double DCS Program\Science\Programming in Science\Project\ecg_scrolling.gif")
    plt.close()


def export_results(df):
# TODO ecg_summary.csv and rr_intervals.csv
    rr_df.to_csv(r"D:\Science and Music Double DCS Program\Science\Programming in Science\Project\RR_Intervals.csv", index=False)



if __name__ == "__main__":
    df = load_ecg(path)
    df = filter_signal(df) # Apply filtering
    df, all_peaks = detect_r_peaks(df)
    rr_df, metrics = compute_metrics(all_peaks)
    create_plots(df, rr_df) 
    print(make_animation(df))
    export_results(rr_df)
    df.to_csv(r"D:\Science and Music Double DCS Program\Science\Programming in Science\Project\Project5.csv", index=False)
