# %%
from operator import sub
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PAHs = [
    "Nap",
    "Acy",
    "Ace",
    "Flu",
    "Phe",
    "Ant",
    "Fla",
    "Pyr",
    "Chr",
    "BaA",
    "BbF",
    "BkF",
    "BaP",
    "Ind",
    "DbA",
    "BPer",
]


RING_TO_PAHS = {
    "2-ring": ["Nap"],
    "3-ring": ["Acy", "Ace", "Flu", "Phe", "Ant"],
    "4-ring": ["Fla", "Pyr", "Chr", "BaA"],
    "5-ring": ["BbF", "BkF", "BaP", "DbA"],
    "6-ring": ["Ind", "BPer"],
}


def scatter_pah():
    pah_file = "data/ambient/PAH_dot1_v3.xlsx"
    sheets = ["HP", "QN", "HY"]
    dfs = []

    for s in sheets:
        df = pd.read_excel(pah_file, sheet_name=s)

        df.rename(columns={"PM": "SIZE"}, inplace=True)

        df["Ind/(Ind+BPer)"] = df["Ind"] / (df["Ind"] + df["BPer"])
        df["BaA/(BaA+Chr)"] = df["BaA"] / (df["BaA"] + df["Chr"])
        df["BaP/BPer"] = df["BaP"] / df["BPer"]
        df["BaA/Chr"] = df["BaA"] / df["Chr"]
        df["Fla/(Fla+Pyr)"] = df["Fla"] / (df["Fla"] + df["Pyr"])
        df["Fla/Pyr"] = df["Fla"] / df["Pyr"]
        df["Phe/(Phe+Ant)"] = df["Phe"] / (df["Phe"] + df["Ant"])

        df["Province"] = df["ID"].apply(lambda x: x[:2])
        dfs.append(df)
    dfs = pd.concat(dfs, ignore_index=True)

    # Set up the figure with better styling
    fig, axes = plt.subplots(5, 2, figsize=(6 * 2, 5 * 3), constrained_layout=True)

    # Define a more readable color palette
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E2E"]

    thresholds = {
        "Diesel Emission": [0.38, 0.64],
        "Fuel Combustion": 0.35,
        "Traffic Emission": 0.6,
        "Grass/Wood/Coal Combustion": 0.5,
        "Lubricant oils/fossil fuel combustion": 0.7,
    }
    thresholds_lables = list(thresholds.keys())

    for i, col in enumerate(
        [
            "BaA/(BaA+Chr)",
            "BaA/Chr",
            "BaP/BPer",
            "Fla/(Fla+Pyr)",
            "Phe/(Phe+Ant)",
        ]
    ):

        tkey = thresholds_lables[i]

        for j, size in enumerate(["PM2.5", "PM10"]):
            dfs_size = dfs[dfs["SIZE"] == size]

            # Create barplot with custom styling
            sns.barplot(
                dfs_size,
                x="Province",
                y=col,
                hue="Sitetype",
                ax=axes[i, j],
                legend=(i == 0 and j == 0),  # Only show legend on first subplot
                palette=colors,
                width=0.7,  # Adjust bar width
                edgecolor="white",
                linewidth=0.5,
            )

            # Style the plot
            axes[i, j].set_title(
                f"{col} ratio ({size})", fontsize=18, fontweight="bold", pad=10
            )
            if j == 1:
                axes[i, j].set_ylabel("")
            axes[i, j].set_xlabel("")

            # Increase tick label font sizes
            axes[i, j].tick_params(axis="both", which="major", labelsize=14)

            # Style the axes
            axes[i, j].grid(True, alpha=0.3, axis="y")
            axes[i, j].set_axisbelow(True)

            # Add threshold lines
            if isinstance(thresholds[tkey], list):
                axes[i, j].axhline(
                    thresholds[tkey][0],
                    color="darkred",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                )
                axes[i, j].axhline(
                    thresholds[tkey][1],
                    color="darkred",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                )
                yannot = thresholds[tkey][1]
            else:
                axes[i, j].axhline(
                    thresholds[tkey],
                    color="darkred",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                )
                yannot = thresholds[tkey]

            # Add annotation with better positioning and styling
            axes[i, j].annotate(
                tkey,
                xy=(0, yannot + 0.02),
                xycoords=(
                    "axes fraction",
                    "data",
                ),  # Changed to axes fraction for x-coordinate
                ha="left",  # Changed to left alignment
                va="bottom",
                color="darkred",
                fontsize=14,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="darkred",
                    alpha=0.8,
                ),
            )

    # Remove legend from first subplot and add a single legend for the entire figure
    if axes[0, 0].get_legend():
        axes[0, 0].get_legend().remove()

    # Get legend handles and labels from the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles and labels:  # Only add legend if there are items to display
        fig.legend(
            handles,
            labels,
            title="Site Type",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(labels),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=16,
            title_fontsize=18,
        )

    plt.tight_layout()
    plt.show()

    return dfs


def plot_pah_composition():
    # Load PAH data
    pah_file = "data/ambient/PAH_dot1_v3.xlsx"
    sheets = ["HP", "QN", "HY"]
    rings = list(RING_TO_PAHS.keys())
    ylabels = ["Concentration (ng/m³)", "Percent composition (%)"]

    # Define consistent color palettes
    ring_colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E2E"]
    pah_colors = plt.cm.Set3(np.linspace(0, 1, len(PAHs)))

    # melt the PAH columns into long form for composition
    for s in sheets:
        df = pd.read_excel(pah_file, sheet_name=s)

        for r in rings:
            pahs = RING_TO_PAHS[r]
            df[r] = df[pahs].sum(axis=1)

        df["Total"] = df[rings].sum(axis=1)
        for r in rings:
            df[r] = df[r] / df["Total"] * 100  # percent

        # Compute mean by ID × Sitetype
        for i, cols in enumerate([rings, PAHs]):
            df_group = df.groupby(["ID", "Sitetype"])[cols].mean().reset_index()

            # Unique IDs
            ids = df_group["ID"].unique()
            n_ids = len(ids)

            # Make subplots (one panel per ID)
            # --- make subplot grid: max 3 per row ---
            n_cols = 3
            n_rows = int(np.ceil(n_ids / n_cols))

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6 * n_cols, 5 * n_rows),
                sharey=True,
                constrained_layout=True,
            )

            # flatten axes for easy iteration
            axes = np.array(axes).reshape(-1)

            # hide empty axes (if IDs < grid size)
            for ax in axes[n_ids:]:
                ax.set_visible(False)

            # Choose colors based on current column set
            colors = ring_colors if i == 0 else pah_colors

            # --- plot each ID ---
            for ax, id_val in zip(axes, ids):

                sub = df_group[df_group["ID"] == id_val]

                # prepare x positions
                x = np.arange(len(sub["Sitetype"]))

                # stacking
                bottom = np.zeros(len(sub))

                for j, pah in enumerate(cols):
                    color = colors[j] if i == 0 else colors[j]
                    ax.bar(
                        x,
                        sub[pah],
                        bottom=bottom,
                        label=pah,
                        color=color,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    bottom += sub[pah].values

                # cosmetics - match scatter_pah styling
                ax.set_title(f"ID = {id_val}", fontsize=20, fontweight="bold", pad=10)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    sub["Sitetype"], rotation=45, ha="right", fontsize=14
                )
                ax.set_ylabel(ylabels[i], fontsize=16)
                ax.tick_params(axis="y", which="major", labelsize=14)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                ax.set_axisbelow(True)

            # place legend outside with consistent styling
            fig.legend(
                cols,
                title="PAH species",
                bbox_to_anchor=(1.02, 0.5),
                loc="center left",
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=14,
                title_fontsize=16,
            )

            plt.tight_layout()
            plt.show()


def plot_btex():

    # Load your BTEX file
    df = pd.read_csv("./data/ambient/BTEX.dot1.csv")

    # BTEX species columns
    btex_cols = ["Ben", "Tol", "Eth", "Xyl"]

    # Ensure proper categorical order for nice plotting
    time_order = ["morning", "afternoon", "evening"]
    sitetype_order = ["Background", "Residential", "Traffic"]

    df["Time"] = pd.Categorical(df["Time"], categories=time_order, ordered=True)
    df["Sitetype"] = pd.Categorical(
        df["Sitetype"], categories=sitetype_order, ordered=True
    )

    # ---- Group by ID × Sitetype × Time ----
    df_group = df.groupby(["ID", "Sitetype", "Time"])[btex_cols].mean().reset_index()

    # unique IDs
    ids = df_group["ID"].unique()
    n_ids = len(ids)

    # ---- Subplot layout: max 3 per row ----
    n_cols = 3
    n_rows = int(np.ceil(n_ids / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        sharey=False,
        constrained_layout=True,
    )

    axes = np.array(axes).reshape(-1)

    # Use consistent color palette matching scatter_pah
    colors = {"Ben": "#2E86AB", "Tol": "#A23B72", "Eth": "#F18F01", "Xyl": "#C73E1D"}

    # ---- Plot each ID as one subplot ----
    for ax, id_val in zip(axes, ids):

        sub = df_group[df_group["ID"] == id_val]

        # Build ordered group index
        combinations = []
        for sitetype in sitetype_order:
            for time in time_order:
                combinations.append((sitetype, time))

        x_positions = np.arange(len(combinations))

        # ---- Plot stacked BTEX ----
        for i, (sitetype, time) in enumerate(combinations):

            row = sub[(sub["Sitetype"] == sitetype) & (sub["Time"] == time)]
            if row.empty:
                continue

            bottom = 0
            for pol in btex_cols:
                val = row[pol].values[0]
                ax.bar(
                    i,
                    val,
                    bottom=bottom,
                    color=colors[pol],
                    edgecolor="white",
                    linewidth=0.5,
                    label=pol if (i == 0 and bottom == 0) else None,
                )
                bottom += val

        # ---- X tick labels (Time) ----
        time_labels = [t for (_, t) in combinations]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(time_labels, rotation=45, ha="right", fontsize=14)

        # ---- Add vertical separators between Sitetype groups ----
        ticks_per_group = len(time_order)
        for g in range(1, len(sitetype_order)):
            sep = g * ticks_per_group - 0.5
            ax.axvline(sep, color="darkred", linewidth=1, alpha=0.6)

        # ---- Add group labels ABOVE using secondary axis ----
        secax = ax.secondary_xaxis("top")
        group_centers = []
        for i, sitetype in enumerate(sitetype_order):
            start = i * ticks_per_group
            end = start + ticks_per_group - 1
            if end < len(x_positions):
                center = (start + end) / 2
                group_centers.append((center, sitetype))

        secax.set_xticks([c for c, _ in group_centers])
        secax.set_xticklabels(
            [lab for _, lab in group_centers], fontsize=14, fontweight="bold"
        )
        secax.tick_params(axis="x", pad=5)

        # ---- Cosmetics matching scatter_pah ----
        ax.set_title(f"ID = {id_val}", fontsize=20, fontweight="bold", pad=10)
        ax.set_ylabel("BTEX concentration (µg/m³)", fontsize=16)
        ax.tick_params(axis="y", which="major", labelsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    # Create legend with consistent styling
    legend_handles = [
        mpatches.Patch(color=colors["Ben"], label="Ben"),
        mpatches.Patch(color=colors["Tol"], label="Tol"),
        mpatches.Patch(color=colors["Eth"], label="Eth"),
        mpatches.Patch(color=colors["Xyl"], label="Xyl"),
    ]

    # Place legend in figure instead of using empty subplot
    fig.legend(
        handles=legend_handles,
        title="BTEX species",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_handles),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=16,
        title_fontsize=18,
    )

    # hide remaining empty axes (if any)
    for ax in axes[n_ids:]:
        ax.set_visible(False)


# %%
