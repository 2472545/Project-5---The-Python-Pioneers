plt.figure(figsize=(14,8))
    sns.lineplot(data=df,  x= 'Time', y="Voltage", hue="Patient ID", palette = 'rainbow')
    plt.title("Voltage vs. Time")
    plt.savefig(r"C:\Users\gabri\OneDrive\Desktop\raw_ecg.png") # Saving in png format within the file path
    plt.xlim(0, 10)
    plt.show()
    plt.close()
