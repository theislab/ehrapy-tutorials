import matplotlib.pyplot as plt
import numpy as np
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

    for i, (color, label, data) in enumerate(zip(colors, labels, adatas, strict=False)):
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
