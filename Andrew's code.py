import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation

DATAFILE = r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\ECG_DATA.csv" # Replace this with the filename from your computer
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

def detect_r_peaks(df, threshold_percentage=0.1):
    r_peak_results = []
    for patient, group in df.groupby('Patient ID'):
        ecg = group['Voltage'].values
        time = group['Time'].values
        threshold = threshold_percentage * np.max(ecg)
        peak_times = []
        for i in range(1, len(ecg) - 1):
            if ecg[i] > threshold and ecg[i] > ecg[i-1] and ecg[i] > ecg[i+1]:
                peak_times.append(time[i])
        print(f"Patient {patient} R-Peaks: {peak_times}")
        # Something is wrong here since the peak times are displayed as empty arrays...
 
def compute_metrics(df): # Based on the detect_r_peaks(df, threshold_percentage=0.1) function
    time = 1 / FS
    BPM_heart_rate = '''60*(number of beats in 10 sec - 1)/(time of last peak - time of first peak in seconds)'''
    rr = '''an array containing the time between each pair of r-peaks'''
    hrv = np.stdv([rr])
    print(f"The BPM heart rate is: {BPM_heart_rate}, \n, The heart rate variability: {hrv}")

def create_plots(df):
    # Raw ECG Plot
    plt.figure(figsize=(14,8))
    sns.lineplot(data=df,  x= 'Time', y="Voltage", hue="Patient ID", palette = 'gist_rainbow')
    sns.hls_palette()
    plt.title("Raw ECG Signals")
    plt.savefig(r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\raw_ecg.png") # Replace this with the filename from your computer
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
    plt.savefig(r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\filtered_ecg.png") # Replace this with the filename from your computer
    plt.show()
    plt.close()

    # RR Scatter TODO "rr_scatter.png"

    # HR histogram TODO "hr_hist.png"

def make_animation(df):
# TODO horizontal scrolling ECG animation
    pass

def export_results(df):
# TODO ecg_summary.csv and rr_intervals.csv
    # I think this function needs to take arrays from the compute_metrics function and convert them to csv files (ecg_summary.csv and rr_intervals.csv)
    pass

if __name__ == "__main__":
    df = load_ecg(path)
    df = filter_signal(df) # Apply filtering
    detect_r_peaks(df) # Detect R-peaks
    create_plots(df) # Visualizations
    # Animation
    print("Complete all TODOs!")