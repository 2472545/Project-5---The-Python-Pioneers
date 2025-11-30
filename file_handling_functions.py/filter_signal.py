WINDOW_SIZE = 15  
def moving_average(signal, window = WINDOW_SIZE):
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode = 'same')

def filter_signal(df, window_size=WINDOW_SIZE):
    df_filtered = df.copy()
    df_filtered['Filtered Voltage'] = (df_filtered.groupby('Patient ID')['Voltage'].transform(lambda x: moving_average(x, window=window_size)))
    return df_filtered
