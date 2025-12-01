def detect_r_peaks(df, threshold_percentage=0.5):
    all_peaks = {}
    for patient, group in df.groupby('Patient ID'):
        group = group[group['Time'] > 0.2] # Excluding the first 0.2 seconds of data since it is not representative
        ecg = group['Voltage'].values
        time = group['Time'].values
        baseline = np.median(ecg) # Computing baseline for this patient
        peak_height = np.max(ecg) - baseline # Computing peak height for this patient
        threshold = threshold_percentage * peak_height # Computing threshold for this patient
        peak_times = [] # Initiating an empty array to store peak times
        for i in range(1, len(ecg) - 1):
            if (ecg[i] - baseline) > threshold and ecg[i] >= ecg[i-1] and ecg[i] > ecg[i+1]:
                peak_times.append(float(time[i]))
            all_peaks[patient] = peak_times  # Storing the peak times for this patient
        print(f"Patient {patient} R-Peaks times (in seconds): {peak_times}") 
    return all_peaks
