# %%
from operator import sub
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from prep_cmb_result import *

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

PAH_SPECIES_NAMES = {
    "TMAC": "Total Mass",
    "FLUC": "Fluorene",
    "PHEC": "Phenanthrene",
    "ANTC": "Anthracene",
    "PYRC": "Pyrene",
    "CHRC": "Chrysene",
    "BAAC": "Benzo[a]anthracene",
    "BBFC": "Benzo[b]fluoranthene",
    "BKFC": "Benzo[k]fluoranthene",
    "BAPC": "Benzo[a]pyrene",
}

BTEX_SPECIES_NAMES = {
    "TMAC": "Total Mass",
    "TOLC": "Toluene",
    "BENC": "Benzene",
    "ETHC": "Ethylbenzene",
    "XYLC": "Xylene",
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
            sep = g * ticks_per_group - 0.2
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


def plot_cmb_source_contribution_details(group="pah"):
    """
    Plot CMB8.2 source contribution results as pie charts for each species at each station

    Args:
        df_sourcename: DataFrame with CMB results containing id, SPECIES, CALCULATED, MEASURED and source columns
    """

    areas_dict = {
        "QN": "Quang Ninh",
        "HP": "Hai Phong",
        "HY": "Hung Yen",
    }
    station_type = {
        "RS": "Residential",
        "TR": "Traffic",
        "BG": "Background",
    }
    source_types = {
        "BIOPAD": "Bioamass burning",
        "TRANSP": "Transportation - Petrol",
        "WASTE": "Waste burning",
        "NONRDD": "Non-road diesel engine",
        "BIOWOO": "Biomass Fuel - Wood",
    }
    df_sourcename = df_sourcename_pah
    if group == "btex":
        source_types = {
            "AUTOC": "Auto coating",
            "GAS92": "Gas stations (A92)",
            "GAS95": "Gas stations (A95)",
            "OPRINT": "Offset printing",
            "GPRINT": "Gravure printing",
        }
        df_sourcename = df_sourcename_btex
    # Remove NaN columns and get numeric source columns
    numeric_cols = df_sourcename.select_dtypes(include=[np.number]).columns.tolist()
    # Remove CALCULATED and MEASURED from source columns
    source_cols = [
        col
        for col in numeric_cols
        if col not in ["CALCULATED", "MEASURED"] and not df_sourcename[col].isna().all()
    ]

    # Calculate Unknown column
    df_plot = df_sourcename.copy()
    df_plot["Unknown"] = (df_plot["MEASURED"] - df_plot["CALCULATED"]) / df_plot[
        "MEASURED"
    ]
    # Only include Unknown if > 0
    df_plot["Unknown"] = df_plot["Unknown"].where(df_plot["Unknown"] > 0, 0)

    # Extract area from ID (first 2 characters)

    df_plot["Area"] = df_plot["id"].apply(lambda x: areas_dict.get(x[:2], "Unknown"))

    # Get species title (remove 'C' from end, use 'Total Mass' for TMAC)
    def get_species_title(group, species):
        if group == "btex":
            return BTEX_SPECIES_NAMES.get(species, species)
        return PAH_SPECIES_NAMES.get(species, species)

    # df_plot["Species_Title"] = df_plot["SPECIES"].apply(
    #     lambda x: get_species_title(group, x)
    # )
    # Get unique areas
    areas = df_plot["Area"].unique()

    # Set up colors - use consistent palette
    colors = [
        "#2E86AB",
        "#A23B72",
        "#F18F01",
        "#C73E1D",
        "#592E2E",
        "#8B4513",
        "#FF6347",
        "#4682B4",
        "#9ACD32",
        "#DDA0DD",
    ]

    for area in areas:
        area_data = df_plot[df_plot["Area"] == area]

        # Get unique stations and species in this area
        unique_stations = area_data["id"].unique()
        unique_species = area_data["SPECIES"].unique()

        # Set up subplot layout
        n_species = len(unique_species)
        n_stations = len(unique_stations)

        if group == "pah":
            # PAH: arrange species in 2 rows for each station
            species_per_row = int(np.ceil(n_species / 2))
            total_rows = n_stations * 2
            total_cols = species_per_row
        else:
            # BTEX: each station has 3 time periods (MO, EV, AF), all species in one row per time
            # Group stations by base name (remove last 2 characters for time)
            base_stations = list(set([station[:-2] for station in unique_stations]))
            n_base_stations = len(base_stations)
            total_rows = n_base_stations * 3  # 3 time periods per base station
            total_cols = n_species

        fig, axes = plt.subplots(
            total_rows,
            total_cols,
            figsize=(8 * total_cols, 7 * total_rows),
            constrained_layout=True,
        )

        # Adjust layout to make room for title and proper spacing
        plt.subplots_adjust(top=0.85, bottom=0.08, hspace=0.5, wspace=0.3)

        # Ensure axes is 2D
        if total_rows == 1 and total_cols == 1:
            axes = np.array([[axes]])
        elif total_rows == 1:
            axes = axes.reshape(1, -1)
        elif total_cols == 1:
            axes = axes.reshape(-1, 1)

        # Helper function to process individual subplots
        def process_subplot(station_id, species, subplot_row, subplot_col):
            # Skip if subplot position is outside grid
            if subplot_row >= total_rows or subplot_col >= total_cols:
                return

            ax = axes[subplot_row, subplot_col]

            # Get data for this station-species combination
            station_species_data = area_data[
                (area_data["id"] == station_id) & (area_data["SPECIES"] == species)
            ]

            if not station_species_data.empty:
                row_data = station_species_data.iloc[0]

                # Prepare data for pie chart
                pie_data = []
                pie_labels = []
                pie_colors = []

                # Add source contributions
                for i, col in enumerate(source_cols):
                    value = row_data[col]
                    if pd.notna(value) and value > 0:
                        pie_data.append(value)
                        pie_labels.append(col)
                        pie_colors.append(colors[i % len(colors)])

                # Add Unknown if > 0
                if row_data["Unknown"] > 0:
                    pie_data.append(row_data["Unknown"])
                    pie_labels.append("Unknown")
                    pie_colors.append("#808080")  # Gray for unknown

                # Create pie chart with larger text
                if pie_data:
                    wedges, texts, autotexts = ax.pie(
                        pie_data,
                        autopct="%1.1f%%",
                        colors=pie_colors,
                        startangle=90,
                        textprops={"fontsize": 30},
                    )

                    # Style the percentage text - much larger font
                    for autotext in autotexts:
                        autotext.set_color("white")
                        autotext.set_fontweight("bold")
                        autotext.set_fontsize(30)

                # Add species title above each pie chart
                species_title = get_species_title(group, species)
                ax.set_title(species_title, fontsize=34, pad=8)
            else:
                # Empty subplot if no data
                ax.set_visible(False)

        # Plot each station-species combination
        if group == "pah":
            for station_idx, station_id in enumerate(unique_stations):
                # Each station gets 2 rows (station_idx * 2 and station_idx * 2 + 1)
                station_row_start = station_idx * 2

                # Process PAH data
                for species_idx, species in enumerate(unique_species):
                    # Calculate which row and column this species should go in
                    species_row = (
                        species_idx // species_per_row
                    )  # 0 or 1 (first or second row)
                    species_col = species_idx % species_per_row  # column within the row

                    # Actual subplot position
                    subplot_row = station_row_start + species_row
                    subplot_col = species_col

                    process_subplot(station_id, species, subplot_row, subplot_col)

                # Add station label for PAH (spanning 2 rows)
                axes[station_row_start, 0].text(
                    -0.3,
                    1,
                    f"{station_type[station_id[2:]]}",
                    transform=axes[station_row_start, 0].transAxes,
                    rotation=0,
                    ha="center",
                    va="center",
                    fontsize=40,
                    fontweight="bold",
                )
        else:
            # BTEX: Group by base station and time period
            base_stations = sorted(
                list(set([station[:-2] for station in unique_stations]))
            )
            time_periods = ["MO", "AF", "EV"]

            for base_idx, base_station in enumerate(base_stations):
                # Add base station label (spanning 3 rows)
                base_row_start = base_idx * 3
                axes[base_row_start, 0].text(
                    -0.3,
                    1,
                    f"{station_type[base_station[2:5]]}",
                    transform=axes[base_row_start, 0].transAxes,
                    ha="center",
                    va="center",
                    fontsize=40,
                    fontweight="bold",
                )

                for time_idx, time_period in enumerate(time_periods):
                    station_id = base_station + time_period
                    if station_id in unique_stations:
                        subplot_row = base_idx * 3 + time_idx

                        # Add time period label for this row
                        time_labels = {
                            "MO": "Morning",
                            "AF": "Afternoon",
                            "EV": "Evening",
                        }
                        axes[subplot_row, 0].text(
                            -0.15,
                            0.5,
                            time_labels[time_period],
                            transform=axes[subplot_row, 0].transAxes,
                            rotation=0,
                            ha="right",
                            va="center",
                            fontsize=30,
                            # fontweight="bold",
                        )

                        # Process all species for this station-time combination
                        for species_idx, species in enumerate(unique_species):
                            subplot_col = species_idx
                            process_subplot(
                                station_id, species, subplot_row, subplot_col
                            )

        # Hide any unused subplots
        for row in range(total_rows):
            for col in range(total_cols):
                if row < total_rows and col < total_cols:
                    if axes[row, col].get_visible():
                        # Check if subplot has any content
                        if not axes[row, col].patches:  # No pie chart drawn
                            axes[row, col].set_visible(False)

        # Add overall title for the area
        fig.suptitle(
            f"CMB Source Contribution by Species - {area}",
            fontsize=40,
            fontweight="bold",
            y=1.02,
        )

        # Create comprehensive legend with larger text
        legend_elements = []
        for i, col in enumerate(source_cols):
            if not df_plot[col].isna().all():
                legend_elements.append(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=colors[i % len(colors)],
                        label=source_types.get(col, col),
                    )
                )

        # Add Unknown to legend if any data has Unknown > 0
        if (df_plot["Unknown"] > 0).any():
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor="#808080", label="Unknown")
            )

        fig.legend(
            handles=legend_elements,
            title="Source Types",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=min(len(legend_elements), 6),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=30,
            title_fontsize=30,
        )

        plt.tight_layout()
        plt.show()


def plot_cmb_tmac_pie_charts(sp="Delhi - India"):
    """Plot TMAC pie charts for each area with BG, RS, TR subplots"""

    areas_dict = {"QN": "Quang Ninh", "HP": "Hai Phong", "HY": "Hung Yen"}
    station_type = {"RS": "Residential", "TR": "Traffic", "BG": "Background"}
    source_types = {
        "BIOPAD": "Bioamass burning",
        "TRANSP": "Gasoline vehicles",
        "WASTE": "Waste burning",
        "NONRDD": "Non-road diesel engine",
        "BIOWOO": "Biomass Fuel - Wood",
    }
    df = df_sourcename_pah_ind
    if sp == "Seoul - Korea":
        source_types = {
            "CPP": "Coal Powerplant",
            "CCO": "Coal (Coke oven)",
            "CRD": "Coal (Residential)",
            "VGS": "Gasoline vehicles",
            "VDI": "Diesel vehicles",
            "CNG": "CNG vehicles",
            "BBN": "Biomass burning",
            "NONRDD": "Non-road diesel engine",
        }
        df = df_sourcename_pah_seoul

    # Filter TMAC data
    df_tmac = df[df["SPECIES"] == "TMAC"].copy()

    # Get source columns and calculate Unknown
    numeric_cols = df_tmac.select_dtypes(include=[np.number]).columns.tolist()
    source_cols = [
        col
        for col in numeric_cols
        if col not in ["CALCULATED", "MEASURED"] and not df_tmac[col].isna().all()
    ]

    df_tmac["Unknown"] = (df_tmac["MEASURED"] - df_tmac["CALCULATED"]) / df_tmac[
        "MEASURED"
    ]
    df_tmac["Unknown"] = df_tmac["Unknown"].where(df_tmac["Unknown"] > 0, 0)
    df_tmac["id"] = df_tmac["id"].apply(lambda x: x.split("_")[1])
    df_tmac["Area"] = df_tmac["id"].apply(lambda x: areas_dict.get(x[:2], "Unknown"))
    df_tmac["Station_Type"] = df_tmac["id"].apply(lambda x: x[2:4])

    # Create consistent color mapping for source types
    if sp == "Delhi - India":
        source_color_map = {
            "TRANSP": "#4682B4",  # Purple
            "WASTE": "#F18F01",  # Orange
            "NONRDD": "#C73E1D",  # Red
        }
    else:  # Seoul - Korea
        source_color_map = {
            "CRD": "#CFCF18",  # Orange
            "VGS": "#4682B4",  # Red
            "VDI": "#0A2187",  # Brown
            "NONRDD": "#C73E1D",  # Steel blue
        }

    for area in df_tmac["Area"].unique():
        area_data = df_tmac[df_tmac["Area"] == area]
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

        for idx, st_type in enumerate(["BG", "RS", "TR"]):
            ax = axes[idx]
            station_data = area_data[area_data["Station_Type"] == st_type]

            if not station_data.empty:
                row_data = station_data.iloc[0]
                pie_data, pie_colors = [], []

                for col in source_cols:
                    value = row_data[col]
                    if pd.notna(value) and value > 0:
                        pie_data.append(value)
                        pie_colors.append(source_color_map.get(col, "#808080"))

                if row_data["Unknown"] > 0:
                    pie_data.append(row_data["Unknown"])
                    pie_colors.append("#808080")

                if pie_data:
                    wedges, texts, autotexts = ax.pie(
                        pie_data,
                        autopct="%1.1f%%",
                        colors=pie_colors,
                        startangle=90,
                        textprops={"fontsize": 30},
                    )

                    for autotext in autotexts:
                        autotext.set_color("white")
                        autotext.set_fontweight("bold")
                        autotext.set_fontsize(30)

                ax.set_title(
                    station_type[st_type], fontsize=34, fontweight="bold", pad=10
                )
            else:
                ax.set_visible(False)

        fig.suptitle(
            f"Source Contribution - {area} (Source Profile: {sp})",
            fontsize=40,
            fontweight="bold",
            y=1.02,
        )

        # Create legend
        legend_elements = []
        for col in source_cols:
            legend_elements.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=source_color_map.get(col, "#808080"),
                    label=source_types.get(col, col),
                )
            )

        if (df_tmac["Unknown"] > 0).any():
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor="#808080", label="Unknown")
            )

        fig.legend(
            handles=legend_elements,
            title="Source Types",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=min(len(legend_elements) / 2, 6),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=30,
            title_fontsize=30,
        )

        plt.tight_layout()
        plt.show()


# %%
plot_cmb_tmac_pie_charts(sp="Delhi - India")
plot_cmb_tmac_pie_charts(sp="Seoul - Korea")
# %%
