import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
DATAFILE = r"D:\Science and Music Double DCS Program\Science\Programming in Science\Project\ECG_DATA.csv" # Replace this with the filename from your computer
FS = 400 # Sampling frequency in Hz
WINDOW_SIZE = 15 # For moving average calculation

def load_ecg(path):
    df = pd.read_csv(path)
    return df
path = DATAFILE

def filter_signal(df, window = WINDOW_SIZE):
    kernel = np.ones(window) / window
    df['Filtered Voltage'] = (
        df.groupby('Patient ID')['Voltage']
        .transform(lambda x: np.convolve(x, kernel, mode='same'))
        )
    return df

def compute_metrics(all_peaks):
    #metrics = []
    for patient, peak_times in all_peaks.items():
        bpm = 60 * (len(peak_times) - 1) / (peak_times[-1] - peak_times[0]) # 60*(number of beats in 10 sec - 1)/(time of last peak - time of first peak in seconds)
        rr = (np.diff(peak_times)) # Computes RR-intervals as the time difference between each pair of R-peaks
        hrv = np.std(rr) # Standard deviation of RR-intervals
        #patient_metrics = [patient, bpm, rr, hrv]
        #metrics.append(patient_metrics)
        #print(f"Patient {patient}: BPM = {bpm:.2f}, HRV = {hrv:.4f}")
    return rr #metrics
    
def detect_r_peaks(df, threshold_percentage=0.5):
    # Creating new empty columns in the dataframe
    df["R-peak time"] = np.nan
    df["R-peak voltage"] = np.nan

    for patient, group in df.groupby('Patient ID'):
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
            if (ecg[i] - baseline) > threshold and ecg[i] >= ecg[i-1] and ecg[i] > ecg[i+1]:
                peak_times.append(float(time[i]))
                peak_voltages.append(float(ecg[i]))
                peak_indices.append(idx[i])
        df.loc[peak_indices, "R-peak time"] = peak_times
        df.loc[peak_indices, "R-peak time"] = peak_voltages
    return df


def create_plots(df):
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

    # RR Scatter TODO "rr_scatter.png"

    # HR histogram TODO "hr_hist.png"

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

# NOT SURE IF THIS IS THE ANIMATION HE WANTS; WE CAN ASK HIM ON TUESDAY

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
    # I think this function needs to take arrays from the compute_metrics function and convert them to csv files (ecg_summary.csv and rr_intervals.csv)
    pass

if __name__ == "__main__":
    df = load_ecg(path)
    df = filter_signal(df) # Apply filtering
    detect_r_peaks(df) # Detect R-peaks
    all_peaks = detect_r_peaks(df) # Makes all_peaks accessible for following functions
    #metrics = compute_metrics(all_peaks)
    create_plots(df) # Visualizations
    print(make_animation(df))
    print("Complete all TODOs!")
