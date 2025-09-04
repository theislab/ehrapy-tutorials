import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def distributions_plot(labels, adatas, var_of_interest):
    """Creates a raincloud-plot inspired distribution plot for the `var_of_interest` variable within given adatas."""
    colors = sns.color_palette("tab10")[: len(adatas)]

    height_ratios = [10] + [1] * len(adatas)

    # Create the subplots with specified height ratios
    fig, axes = plt.subplots(
        nrows=1 + len(adatas),
        figsize=(6, 1.5 * len(adatas)),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0)

    for i, (color, label, data) in enumerate(zip(colors, labels, adatas)):
        statistics = {
            "Mean": np.nanmean(data[:, data.var_names == var_of_interest].X.astype(np.float64)),
            "Stdev": np.nanstd(data[:, data.var_names == var_of_interest].X.astype(np.float64)),
        }
        stat_string = rf"${round(statistics['Mean'].item(), 2)} \pm {round(statistics['Stdev'].item(), 2)}$"

        sns.kdeplot(
            x=data[:, data.var_names == var_of_interest].X.flatten(),
            color=color,
            ax=axes[0],
            label=f"{label}: {stat_string}",
        )
        sns.boxplot(
            x=data[:, data.var_names == var_of_interest].X.flatten(),
            orient="h",
            color=color,
            ax=axes[i + 1],
            width=0.25,
            fliersize=0.3,
        )
        sns.rugplot(
            x=data[:, data.var_names == var_of_interest].X.flatten(),
            color=color,
            alpha=0.1,
            ax=axes[i + 1],
            height=0.2,
        )

        axes[i + 1].spines["right"].set_visible(False)
        axes[i + 1].spines["left"].set_visible(False)
        axes[i + 1].spines["bottom"].set_visible(False)
        axes[i + 1].set_yticks([])

        if i > 0:
            axes[i + 1].spines["top"].set_visible(False)

    fig.suptitle(f"Distribution of the {var_of_interest} variable")
    axes[0].legend(fontsize=9)


def grouped_barplot(df):
    """Creates grouped barplots for a Fairlearn MetricFrame.bygroups dataframe."""
    df_long = pd.melt(df.reset_index(), id_vars=["race"], var_name="metric", value_name="value")

    # cohorttracker color palette
    colors = sns.color_palette("colorblind", n_colors=14)

    # colors from cohorttracker
    palette = [colors[0], colors[2], colors[3], colors[5]]

    fig, ax = plt.subplots()
    sns.barplot(data=df_long, x="metric", y="value", hue="race", palette=palette)
    ax.legend(title="Race", bbox_to_anchor=(1.05, 1))

    ax.set_ylabel("Value")
    ax.set_xlabel("Metric")
    ax.set_xticklabels(labels=["Selection Rate", "False Negative Rate", "Balanced Accuracy"])


def plot_hist_normalization(
    adata,
    adata_scaled_together,
    adata_scaled_separate,
    group_key="age_group",
    var_of_interest="num_medications",
):
    """Plot histogram of original data, jointly normalized, and split normalization (by group_key variable)."""
    fig, axs = plt.subplots(1, 3, figsize=(21, 5))

    # plot raw adata
    for group in adata.obs[group_key].unique():
        adata_group = adata[adata.obs[group_key] == group, var_of_interest]
        axs[0].hist(
            adata_group.X,
            bins=10,
            alpha=0.7,
            label=group,
            orientation="horizontal",
            weights=np.ones(len(adata_group)) / len(adata_group),
        )
        axs[0].set_ylabel(var_of_interest)
        axs[0].set_xlabel("% of visits per group")
        axs[0].legend(title=group_key)
        axs[0].set_title(f"Original Distributions of {var_of_interest}")

    # plot together normalized data
    for group in adata_scaled_together.obs[group_key].unique():
        adata_group = adata_scaled_together[adata_scaled_together.obs[group_key] == group, var_of_interest]
        axs[1].hist(
            adata_group.X,
            bins=10,
            alpha=0.7,
            label=group,
            orientation="horizontal",
            weights=np.ones(len(adata_group)) / len(adata_group),
        )
        axs[1].set_ylabel(var_of_interest)
        axs[1].set_xlabel("% of visits per group")
        axs[1].legend(title=group_key)
        axs[1].set_title(f"Overall normalization of {var_of_interest}")

    # plot separately normalized data
    for group in adata_scaled_separate.obs[group_key].unique():
        adata_group = adata_scaled_separate[adata_scaled_separate.obs[group_key] == group, var_of_interest]
        axs[2].hist(
            adata_group.X,
            bins=10,
            alpha=0.7,
            label=group,
            orientation="horizontal",
            weights=np.ones(len(adata_group)) / len(adata_group),
        )
        axs[2].set_ylabel(var_of_interest)
        axs[2].set_xlabel("% of visits per group")
        axs[2].legend(title=group_key)
        axs[2].set_title(f"Separate normalization of {var_of_interest}")

    plt.tight_layout()
    plt.show()
