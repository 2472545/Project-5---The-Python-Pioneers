def compute_metrics(all_peaks):
    metrics = []
    for patient, peak_times in all_peaks.items():
        bpm = 60 * (len(peak_times) - 1) / (peak_times[-1] - peak_times[0]) # 60*(number of beats in 10 sec - 1)/(time of last peak - time of first peak in seconds)
        rr = (np.diff(peak_times)) # Computes RR-intervals as the time difference between each pair of R-peaks
        hrv = np.std(rr) # Standard deviation of RR-intervals
        patient_metrics = [patient, bpm, rr, hrv]
        metrics.append(patient_metrics)
        print(f"Patient {patient}: BPM = {bpm:.2f}, HRV = {hrv:.4f}")
    return metrics
    
