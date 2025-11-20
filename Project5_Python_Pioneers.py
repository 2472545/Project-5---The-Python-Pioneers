# ECG_Project_starter.py
# Starter File — Complete all TODOs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
DATAFILE = "ECG_sample.csv"
FS = 250 # sampling frequency
def load_ecg(path):
    with open(path, 'r') as ecg_csv_file:
        rows = []
        for line in ecg_csv_file:
            row_values = [value for value in line.strip().split(',')]
            rows.append(row_values)
    return list(rows)
    # TODO load CSV, convert to dataframe
path = r"C:\Users\2472545\Desktop\ECG_sample.csv"
load_ecg(path)
def filter_signal(values):
# TODO moving average smoothing
    pass
def detect_r_peaks(values):
# TODO simple threshold-based R-peak detection
    pass
def compute_metrics(df):
# TODO compute BPM and HRV per patient
    pass
def create_plots(df):
# TODO raw_ecg.png
# TODO filtered_ecg.png
# TODO rr_scatter.png
# TODO hr_hist.png
    pass
def make_animation(df):
# TODO horizontal scrolling ECG animation
    pass
def export_results(df):
# TODO ecg_summary.csv and rr_intervals.csv
    pass
if __name__ == "__main__":
    df = load_ecg(DATAFILE)
# Apply filtering
# Detect R-peaks
# Visualizations
# Animation
print("Complete all TODOs!")

# Animation
print("Complete all TODOs!")

