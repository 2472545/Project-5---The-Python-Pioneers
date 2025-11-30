plt.figure(figsize=(14,8))
    sns.lineplot(data=df_filtered, x='Time', y="Filtered Voltage", hue="Patient ID", palette ='cool') 
    plt.title("Filtered ECG Signals (Moving Average Smoothing)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.xlim(df_filtered['Time'].min(), df_filtered['Time'].min() + 10)  # safer limit
    plt.savefig(r"/Users/gabri/Downloads/ECG_DATA.png")
    plt.show()
    plt.close()
