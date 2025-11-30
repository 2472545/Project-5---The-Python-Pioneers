def make_animation(df_filtered):
    ecg_data = df_filtered.groupby("Time")["Voltage"].mean().reset_index()
    times = ecg_data["Time"].values
    voltage = ecg_data["Voltage"].values

    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_facecolor('black')
    ax.grid(True)
    ax.set_xlim(min(times), max(times))
    ax.set_ylim(min(voltage), max(voltage) + 2)

    line, = ax.plot([], [], '-', color = 'lime', lw = 2)
    print(line)

# NOT SURE IF THIS IS THE ANIMATION HE WANTS; WE CAN ASK HIM ON TUESDAY

    def animate(i):
        window = 400 # Sampling frequency
        start = max(0, i - window)
        x = times[:i]
        y = voltage[:i]
        line.set_data(x, y)
        ax.set_xlim(times[start], times[i])
        ax.set_title(f"Time: {times[i]:.2f} s")
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=len(times), interval = 1)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.show()
    anim.save(r"C:\Users\gabri\OneDrive\Desktop\ecg_scrolling.gif")
    plt.close()

print(make_animation(df_filtered))
