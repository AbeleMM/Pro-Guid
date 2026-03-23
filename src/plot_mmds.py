from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates


def main() -> None:
    path = Path(__file__).parent / "planar_mmd_128.csv"
    df = pd.read_csv(path)
    df["Method"] = df["Backbone"] + ' ' + df["Sampling"]
    df = df.drop(["Backbone", "Sampling"], axis=1)


    fig, ax = plt.subplots()  # add constrained_layout=True
    ax = parallel_coordinates(df, "Method", ax=ax, colormap="Set2")
    ax.invert_yaxis()
    ax.legend(loc="lower left", fontsize="small")
    ax.set(title="MMD Comparison Planar N=128", yscale="log", xlabel="Metric", ylabel="MMD Value (log-scale)")
    fig.savefig(f"{path.stem}_pc.pdf", pad_inches=0., bbox_inches="tight")


    ###


    import numpy as np
    from matplotlib.cm import get_cmap

    categories = [col for col in df.columns if col != "Method"]
    N = len(categories)
    # Calculate angles for the spokes (we divide the circle into N parts)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    # Close the loop for the plot by appending the start to the end
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # Set the start to 12 o'clock and direction clockwise (optional, for aesthetics)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories)
    # Using a colormap to differentiate methods
    cmap = get_cmap("Set1")

    for idx, row in df.iterrows():
        values = row[categories].tolist()
        values += values[:1] # Close the loop
        color = cmap(idx) # Cycle through colors
        ax.plot(angles, values, label=row["Method"], color=color)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize="small")
    ax.set(title="MMD Comparison Planar N=128")
    fig.savefig(f"{path.stem}_radar.pdf", pad_inches=0., bbox_inches="tight")


    ###


    import seaborn as sns

    df_melted = df.melt(
        id_vars="Method",
        var_name="Metric",
        value_name="MMD Value"
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()
    sns.barplot(
        data=df_melted,
        x="Metric",
        y="MMD Value",
        hue="Method",
        ax=ax,
        palette="Set2",
        edgecolor="black"
    )
    sns.move_legend(ax, "upper left", fontsize="small")
    ax.set(title="MMD Comparison Planar N=128", yscale="log", xlabel="Metric", ylabel="MMD Value (log-scale)")
    fig.savefig(f"{path.stem}_bars.pdf", pad_inches=0., bbox_inches="tight")


if __name__ == "__main__":
    main()
