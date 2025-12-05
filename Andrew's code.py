import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation


DATAFILE = r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\Final Submission files\ECG_DATA.csv" # Replace this with the filename from your computer
FS = 400 # Sampling frequency in Hz

def load_ecg(path): # We define the function
    df = pd.read_csv(path) # Converting the CSV file to our dataframe
    return df # Excel file with one row per value type

path = DATAFILE


WINDOW_SIZE = 15 # For moving average calculation

def filter_signal(df, window = WINDOW_SIZE): # We define the function
    kernel = np.ones(window) / window
    df['Filtered Voltage'] = (df.groupby('Patient ID')['Voltage'].transform(lambda x: np.convolve(x, kernel, mode='same')))
    return df


def detect_r_peaks(df, threshold_percentage = 0.5): # The threshold will aid in computing the R peaks
    df["R-peak time"] = np.nan # We add an empty row to the already existing dataframe for the r peak timetime
    df["R-peak voltage"] = np.nan # We add an empty row to the already existing dataframe for the r peak voltage
    all_peaks = {} # We set an empty list for the r peaks
    for patient_id, group in df.groupby('Patient ID'): # We group by Patient ID so that we have the peaks for every patient
        group = group[group['Time'] > 0.2] # Excluding the first 0.2 seconds of data since it is not representative
        ecg = group['Voltage'].values # Adding the voltage values
        time = group['Time'].values # Adding the time values
        idx = group.index.values # # Index the values
        
        baseline = np.median(ecg) # Computing baseline for patient
        peak_height = np.max(ecg) - baseline # Computing peak height of ecg
        threshold = threshold_percentage * peak_height # Computing threshold for patient
        
        peak_times = [] # Initializing empty arrays to store peak times information
        peak_voltages = [] # Initializing empty arrays to store peak voltages information
        peak_indices = [] # Initializing empty arrays to store peak indices information
        
        for i in range(1, len(ecg) - 1): # We put the start as 1 so that the range function indeed indexes starting at 1
            if (ecg[i] - baseline) > threshold and ecg[i] >= ecg[i-1] and ecg[i] > ecg[i+1]: # Condition for peak detection
                peak_times.append(float(time[i]))                                            # Making lists of the peak times, voltages, and indices
                peak_voltages.append(float(ecg[i]))
                peak_indices.append(idx[i])
        df.loc[peak_indices, "R-peak time"] = peak_times
        df.loc[peak_indices, "R-peak voltage"] = peak_voltages
        all_peaks[patient_id] = peak_times
    return df, all_peaks


def compute_metrics(all_peaks): # We define the function
    rows = [] # We set an empty list for the rows
    metrics = {} # We set an empty list for the metrics
    
    for patient_id, peak_times in all_peaks.items(): # This will loop for every patient
        rr = np.diff(peak_times) # Calculating the difference between r peaks
        for rr_value in rr:
            rows.append({"Patient ID": patient_id, "RR-Interval": rr_value}) # Assigning the rr interval to the corresponding patient in a list
        # Calculating BPM and HRV
        bpm = 60 * (len(peak_times) - 1) / (peak_times[-1] - peak_times[0]) # The formula for heart rate in beats per minute
        hrv = np.std(rr)  # Calculating heart rate variability as the standard deviation of the rr intervals
        
        metrics[patient_id] = {"BPM": bpm, "RR_intervals": rr.tolist(), "HRV": hrv} 
    
    rr_df = pd.DataFrame(rows) # We add the rr intervals to the dataframe using pandas
    rr_df['RR_n+1'] = rr_df.groupby("Patient ID")['RR-Interval'].shift(-1) # We Add RR_n+1 for scatter plot, as this is the case for many rr plots in clinical setting
    
    summary_df = pd.DataFrame(metrics)
    
    return rr_df, summary_df, metrics

def create_plots(df, rr_df): # We define the function to create all plots

    # Raw ECG Plot
    plt.figure(figsize=(14,8)) # This sets the dimensions of the plot
    sns.lineplot(data=df,  x= 'Time', y="Voltage", hue="Patient ID", palette = 'gist_rainbow') # We set our parameters
    sns.hls_palette()
    plt.title("Raw ECG Signals") # We set the title
    plt.savefig(r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\Final Submission files\raw_ecg.png") # Replace this with the filename from your computer
    plt.xlabel("Time (s)") # We label the x axis
    plt.ylabel("Voltage (mV)") # We label the y axis
    plt.xlim(0, 10) # We set the x limit from 0 to 10 seconds, as is the domain for our time values
    plt.show() # This will make the plot appear
    plt.close() # This will close the plot


    # Filtered ECG
    plt.figure(figsize=(14,8)) # This sets the dimensions of the plot
    sns.lineplot(data=df, x='Time', y="Filtered Voltage", hue="Patient ID", palette='rainbow') # We set our parameters
    plt.title("Filtered ECG Signals (Moving Average Smoothing)") # We set the title
    plt.xlabel("Time (s)") # We label the x axis
    plt.ylabel("Voltage (mV)") # We label the y axis
    plt.xlim(df['Time'].min(), df['Time'].min() + 10)  # safer limit
    plt.savefig(r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\Final Submission files\filtered_ecg.png") # Replace this with the filename from your computer
    plt.show() # This will make the plot appear
    plt.close() # This will close the plot


    # RR Scatter plot 
    rr_df_clean = rr_df.dropna(subset=['RR_n+1'])  # drop NaNs
    plt.figure(figsize=(10,6)) # This sets the dimensions of the plot
    rr_df["RR_n+1"] = rr_df_clean.groupby("Patient ID")["RR-Interval"].shift(-1)
    sns.scatterplot(data=rr_df, x="RR-Interval", y="RR_n+1", hue="Patient ID", palette='cool') # We set our parameters
    plt.title("RR Interval Scatter Plot") # We set the title
    plt.xlabel("RR Interval (s)") # We label the x axis
    plt.ylabel("Next RR Interval (s)") # We label the y axis
    plt.savefig(r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\Final Submission files\rr_scatter.png") # Replace this with the filename from your computer
    plt.show() # This will make the plot appear
    plt.close() # This will close the plot


    # HR histogram 
    bpm_values = [info["BPM"] for info in metrics.values()]
    plt.figure(figsize=(10,6)) # This sets the dimensions of the plot
    sns.histplot(bpm_values, bins=10, color='skyblue', edgecolor='black') # We set our parameters
    plt.title("Histogram of Heart Rate (BPM)") # We set the title
    plt.xlabel("Beats Per Minute (BPM)") # We label the x axis
    plt.ylabel("Frequency") # We label the y axis
    plt.savefig(r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\Final Submission files\hr_hist.png") # Replace this with the filename from your computer
    plt.show() # This will make the plot appear
    plt.close() # This will close the plot


def make_animation(df): # We define the function
    ecg_data = df.groupby("Time")["Filtered Voltage"].mean().reset_index() # We use time and voltage as our values and we take the MEAN of the voltage
    times = ecg_data["Time"].values # Our time values from our dataframe
    voltage = ecg_data["Filtered Voltage"].values # Our voltage values from our dataframe

    fig, ax = plt.subplots(figsize=(14,8)) # We set the dimentions for our figure and our axis
    ax.set_facecolor('black') # We put a black backround, typical of a clinical setting 
    ax.grid(True) # We show the plot's grid
    ax.set_xlim(min(times), max(times)) # We set the time limit  to the max, which is 10s
    ax.set_ylim(min(voltage), max(voltage) + 2) # We set the voltage from the min and the max, and add 2 for insurance

    line, = ax.plot([], [], '-', color = 'lime', lw = 2) # We set the line type, the color 'lime' which is typical for a clinical setting, and the line width to 2
    print(line)

    def animate(i): # We define the animation function 
        frequency = 400 # Sampling frequency
        start = max(0, i - frequency)
        x = times[:i] # The time of the index in question
        y = voltage[:i] # The voltage of the index in question
        line.set_data(x, y) # We set a line with respect to x and y
        ax.set_xlim(times[start], times[i]) # We set the x limit to the i'th index of time
        ax.set_title(f"Time: {times[i]:.2f} s") # We set the title and we make it change depending on the time. WE WILL TRY TO MAKE THE GRAPH SCROLL FASTER
        return line # We return the line

    anim = animation.FuncAnimation(fig, animate, frames=len(times), interval = 1) # Finally, we plot the animation
    plt.xlabel("Time (s)") # We label the x axis
    plt.ylabel("Voltage (mV)") # We label the y axis
    plt.show() # This will make the plot appear
    anim.save(r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\Final Submission files\ecg_scrolling.gif") # Replace the file path for your computer
    plt.close() # This will close the plot


def export_results(rr_df, summary_df):
# TODO ecg_summary.csv and rr_intervals.csv

    rr_df.to_csv(r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\Final Submission files\rr_intervals.csv") # We export the results of the rr intervals to a new csv file
    summary_df.to_csv(r"K:\Science and Music Double DCS Program\Science\Programming in Science\Project\Final Submission files\ecg_summary.csv") # Working on it, but essentially the same as the latter


if __name__ == "__main__":
    df = load_ecg(path) # We define the dataframe by calling the csv conversion function
    df = filter_signal(df) # We pply filtering to the dataframe
    df, all_peaks = detect_r_peaks(df) # We proceed by calling the function
    rr_df, summary_df, metrics = compute_metrics(all_peaks) # We proceed by calling the function
    #create_plots(df, rr_df) # We call the plot creating function
    #print(make_animation(df)) # We call the animation function
    export_results(rr_df, summary_df) # We call the export results function
