# ECG_Project_starter.py
# Starter File — Complete all TODOs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation

DATAFILE = r"C:\Users\gabri\Downloads\ECG_data_patient1_10seconds.csv" # Replace this with the filename from your computer
FS = 400 # sampling frequency

def load_ecg(path):
    df = (pd.read_csv(path, header=None)).transpose()  # .transpose() converts the data from a row to a column
    df.columns = ['Voltage']                           # Labeling the voltage data column
    df['Voltage'] = pd.to_numeric(df['Voltage'])       # Cleaning the data
    df['Time'] = df.index / FS                         # Using the frequency to find the time intervals
    return df
path = DATAFILE
print(load_ecg(path)) # Temporarily print to confirm data loading success

def filter_signal(values):
# TODO moving average smoothing
    pass
def detect_r_peaks(values):
# TODO simple threshold-based R-peak detection
    pass
def compute_metrics(df):
    time = np.diff(df['Time'])
    BPM_heart_rate = time / 60 # THIS IS WRONG, BUT I THINK ITS ON THE RIGHT TRACK
# TODO compute BPM and HRV per patient
    return BPM_heart_rate
print(compute_metrics(load_ecg(path)))

def create_plots(df):
# TODO raw_ecg.png
    plt.figure(figsize=(8,8))
    sns.lineplot(data=df, x = df['Time'] , y = df['Voltage'])
    plt.title("Voltage vs. Time")
    plt.savefig(r"C:\Users\gabri\OneDrive\Desktop\raw_ecg.png") # Replace this with the filename from your computer
    plt.xlim(0, 4)
    plt.show()
    plt.close()
create_plots(load_ecg(path))
# TODO filtered_ecg.png
# TODO rr_scatter.png
# TODO hr_hist.png


def make_animation(df):
# TODO horizontal scrolling ECG animation
    pass
def export_results(df):
# TODO ecg_summary.csv and rr_intervals.csv
    pass
#if __name__ == "__main__":
    #df = load_ecg(DATAFILE)
    #df = filter_signal(df)
    #df = detect_r_peaks(df)
    #create_plots(df)
    # Animation
    #print("Complete all TODOs!")
