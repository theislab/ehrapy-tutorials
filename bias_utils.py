import numpy as np
import matplotlib.pyplot as plt



def plot_hist_num_medication(adata_to_plot, title, has_missing):
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



def plot_hist_normalization(adata, adata_scaled_together, adata_scaled_separate, batch_key="age_group", var_of_interest="num_medications"):
    """Plot histogram of original data, jointly normalized, and split normalization (by batch_key variable)."""

    fig, axs = plt.subplots(1, 3, figsize=(21, 5))

    # plot raw adata
    for group in adata.obs[batch_key].unique():
        adata_group = adata[adata.obs[batch_key] == group, var_of_interest]
        axs[0].hist(
            adata_group.X,
            bins=10,
            alpha=0.7,
            label=group,
            orientation="horizontal",
            weights = np.ones(len(adata_group)) / len(adata_group)
        )
        axs[0].set_ylabel(var_of_interest)
        axs[0].set_xlabel("% of visits per group")
        axs[0].legend(title=batch_key)
        axs[0].set_title(f"Original Distributions of {var_of_interest}")

    # plot together normalized data
    for group in adata_scaled_together.obs[batch_key].unique():
        adata_group = adata_scaled_together[adata_scaled_together.obs[batch_key] == group, var_of_interest]
        axs[1].hist(
            adata_group.X,
            bins=10,
            alpha=0.7,
            label=group,
            orientation="horizontal",
            weights = np.ones(len(adata_group)) / len(adata_group)
        )
        axs[1].set_ylabel(var_of_interest)
        axs[1].set_xlabel("% of visits per group")
        axs[1].legend(title=batch_key)
        axs[1].set_title(f"Overall normalization of {var_of_interest}")

    # plot separately normalized data
    for group in adata_scaled_separate.obs[batch_key].unique():
        adata_group = adata_scaled_separate[adata_scaled_separate.obs[batch_key] == group, var_of_interest]
        axs[2].hist(
            adata_group.X,
            bins=10,
            alpha=0.7,
            label=group,
            orientation="horizontal",
            weights = np.ones(len(adata_group)) / len(adata_group)
        )
        axs[2].set_ylabel(var_of_interest)
        axs[2].set_xlabel("% of visits per group")
        axs[2].legend(title=batch_key)
        axs[2].set_title(f"Separate normalization of {var_of_interest}")

    plt.tight_layout()
    plt.show()