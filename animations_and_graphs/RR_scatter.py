plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x="pH", y="Turbidity_NTU", hue="SiteID")
    plt.title("pH vs Turbidity")
    plt.savefig(r"C:\Users\azharh\Desktop\ListofFinalProjects\WaterQuality_Project\scatter_pH_vs_turbidity.png") # PUT YOUR PATH
    plt.close()

# REPLACE WITH TITLE AND DATA ACCORDING TO OUR PROJECT
