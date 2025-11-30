DATAFILE = r"/Users/gabri/Downloads/ECG_DATA.csv" # put your own file pathname
FS = 400 # sampling frequency

def load_ecg(path):
    df = pd.read_csv(path)
    return df
path = DATAFILE
print(load_ecg(path)) # Temporarily print to confirm data loading success
