import numpy as np
import matplotlib.pyplot as plt



def hist_plot_num_medication(adata_to_plot, title, has_missing):
    """Plot histogram of number of medications per visit."""
    num_medications = adata_to_plot.X[:, adata_to_plot.var_names == "num_medications"]
    mean_value = np.nanmean(num_medications)
    median_value = np.nanmedian(num_medications)
    std_value = np.nanstd(num_medications)

    plt.hist(num_medications, bins=10, alpha=0.5)

    missing_hint = " (nan ignored)" if has_missing else ""
    plt.axvline(
        mean_value,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean{missing_hint}: {mean_value:.2f}",
    )
    plt.axvline(
        median_value,
        color="g",
        linestyle="dashed",
        linewidth=1,
        label=f"Median{missing_hint}: {median_value:.2f}",
    )
    plt.axvline(
        mean_value + std_value,
        color="b",
        linestyle="dashed",
        linewidth=1,
        label=f"Stdev{missing_hint}: {std_value:.2f}",
    )
    plt.axvline(mean_value - std_value, color="b", linestyle="dashed", linewidth=1)
    plt.xlabel("Num. Medications")
    plt.ylabel("Num. Visits")
    plt.title(title)
    plt.legend()
    return mean_value, median_value, std_value