# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec


from utils import *


def plot_air_quality_with_histograms(in_df):
    df = in_df.copy()
    # Ensure datetime index
    # if not isinstance(df.index, pd.DatetimeIndex):
    #     df[time_col] = pd.to_datetime(df[time_col])
    #     df = df.set_index(time_col)
    #     df = df.sort_index()

    variables = df.columns
    n = len(variables)

    # Set up figure with two columns: time series and histogram
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(10, 1.8 * n),
        gridspec_kw={"width_ratios": [4, 1]},
        constrained_layout=True,
    )

    if n == 1:  # Handle edge case of single row
        axes = np.array([axes])

    for i, var in enumerate(variables):
        ts_ax = axes[i, 0]
        hist_ax = axes[i, 1]
        series = df[var]

        valid = series.dropna()

        # --- Timeseries Panel ---
        ts_ax.plot(series.index, series.values, color="orange", linewidth=0.5)
        ts_ax.fill_between(
            series.index,
            0,
            1,
            where=series.isna(),
            transform=ts_ax.get_xaxis_transform(),
            color="red",
            alpha=0.2,
        )
        ts_ax.set_ylabel(var, fontsize=9)
        ts_ax.tick_params(labelsize=8)
        ts_ax.grid(True, linestyle=":", linewidth=0.5)

        # Compute stats
        coverage = 100 * valid.count() / series.shape[0]
        missing = series.isna().sum()
        mean = valid.mean()
        median = valid.median()
        p95 = valid.quantile(0.95)

        title = (
            f"Coverage = {coverage:.1f}% | Missing = {missing} | "
            f"Mean = {mean:.1f} | Median = {median:.1f} | "
            f"95th Percentile = {p95:.1f}"
        )

        ts_ax.set_title(title, fontsize=8, loc="left")

        # --- Histogram Panel ---
        counts, bins = np.histogram(valid.values, bins=30)
        percent = counts / counts.sum() * 100  # Convert to percent
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Plot
        hist_ax.bar(
            bin_centers,
            percent,
            width=(bins[1] - bins[0]),
            color="green",
            edgecolor="black",
        )

        # Labels
        hist_ax.set_xlabel("value", fontsize=7)
        hist_ax.set_ylabel("Percent of Total", fontsize=7)
        hist_ax.tick_params(labelsize=7)
        ts_ax.grid(True, linestyle=":", linewidth=0.5)

    plt.show()


def plot_air_quality_patterns(in_df):
    df = in_df.copy()
    # Ensure datetime index
    columns = df.columns
    # df = df.dropna(subset=[datetime_col])
    # if datetime_col in columns:
    #     df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    #     df = df.dropna(subset=[datetime_col])
    #     df = df.set_index(datetime_col)

    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek  # 0 = Monday
    df["month"] = df.index.month
    df["weekday_name"] = df.index.strftime("%A")
    df["month_name"] = df.index.strftime("%b")
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    for c in columns:
        for v in POLUTANTS:
            if v in c:
                var = c
                print(var)
                try:
                    # Use GridSpec for flexible layout
                    fig = plt.figure(figsize=(16, 10))
                    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])

                    # ==== 1. Top full-span hourly pattern across weekdays ====
                    ax_top = fig.add_subplot(gs[0, :])  # span all 3 columns

                    for i, day in enumerate(weekday_order):
                        group = df[df["weekday_name"] == day].groupby("hour")[var]
                        q25 = group.quantile(0.25)
                        q75 = group.quantile(0.75)
                        q5 = group.quantile(0.05)
                        q95 = group.quantile(0.95)
                        median = group.median()

                        offset = i * 24
                        x = median.index + offset
                        ax_top.plot(x, median, color="darkred")
                        ax_top.fill_between(x, q25, q75, color="red", alpha=0.3)
                        ax_top.fill_between(x, q5, q95, color="red", alpha=0.1)

                    ax_top.set_xticks(np.arange(0, 168, 24))
                    ax_top.set_xticklabels(weekday_order)
                    ax_top.set_xticks(np.arange(0, 168, 3), minor=True)
                    ax_top.tick_params(
                        axis="x", which="minor", labelsize=6, direction="in"
                    )
                    ax_top.set_title(f"Hourly {subscript(var)} pattern across weekdays")
                    ax_top.set_ylabel(var)
                    ax_top.set_xlabel("Hour → Weekday")
                    ax_top.grid(True, linestyle=":")

                    for tick in np.arange(0, 168, 3):
                        hour_label = f"{tick % 24}"  # hour within day

                        ax_top.text(
                            tick,
                            ax_top.get_ylim()[0]
                            + 0.03 * (ax_top.get_ylim()[1] - ax_top.get_ylim()[0]),
                            hour_label,
                            ha="center",
                            va="top",
                            fontsize=8,
                            color="gray",
                        )

                    # ==== 2. Hourly median + quantiles ====
                    ax_hour = fig.add_subplot(gs[1, 0])
                    hour_group = df.groupby("hour")[var]
                    ax_hour.plot(hour_group.median(), color="darkred")
                    ax_hour.fill_between(
                        hour_group.median().index,
                        hour_group.quantile(0.25),
                        hour_group.quantile(0.75),
                        color="red",
                        alpha=0.3,
                    )
                    ax_hour.fill_between(
                        hour_group.median().index,
                        hour_group.quantile(0.05),
                        hour_group.quantile(0.95),
                        color="red",
                        alpha=0.1,
                    )
                    ax_hour.set_title("Hourly distribution")
                    ax_hour.set_xlabel("Hour")
                    ax_hour.set_ylabel(var)
                    ax_hour.grid(True, linestyle=":")

                    # ==== 3. Monthly distribution ====
                    ax_month = fig.add_subplot(gs[1, 1])
                    sns.boxplot(
                        x="month_name",
                        y=var,
                        data=df,
                        order=[
                            "Jan",
                            "Feb",
                            "Mar",
                            "Apr",
                            "May",
                            "Jun",
                            "Jul",
                            "Aug",
                            "Sep",
                            "Oct",
                            "Nov",
                            "Dec",
                        ],
                        ax=ax_month,
                        color="lightcoral",
                    )
                    monthly_median = (
                        df.groupby("month")[var]
                        .median()
                        .reindex(range(1, 13))
                        .fillna(0)
                    )
                    ax_month.plot(
                        [
                            "Jan",
                            "Feb",
                            "Mar",
                            "Apr",
                            "May",
                            "Jun",
                            "Jul",
                            "Aug",
                            "Sep",
                            "Oct",
                            "Nov",
                            "Dec",
                        ],
                        monthly_median.values,
                        color="darkred",
                    )
                    ax_month.set_title("Monthly distribution")
                    ax_month.set_xlabel("Month")
                    ax_month.set_ylabel(var)
                    ax_month.grid(True, linestyle=":")

                    # ==== 4. Weekday distribution ====
                    ax_weekday = fig.add_subplot(gs[1, 2])
                    sns.boxplot(
                        x="weekday_name",
                        y=var,
                        data=df,
                        order=weekday_order,
                        ax=ax_weekday,
                        color="lightcoral",
                    )
                    weekday_median = df.groupby("weekday")[var].median()
                    ax_weekday.plot(
                        weekday_order, weekday_median.values, color="darkred"
                    )
                    ax_weekday.set_title("Weekday distribution")
                    ax_weekday.set_xticklabels(weekday_order, rotation=30)
                    ax_weekday.set_xlabel("Day")
                    ax_weekday.set_ylabel(var)
                    ax_weekday.grid(True, linestyle=":")

                    plt.tight_layout()
                    plt.show()
                except:
                    print("Error: ", var)


def plot_air_pollution(in_df, site_type):

    intervals = ["1h", "8h", "24h", "1y"]

    df = in_df.copy()
    columns = filter_pollutant_cols(df.columns)
    df = df[columns]

    df_resampleds = resample_with_threshold(df, site_type)

    for interval in intervals:
        pass
        if interval != "24h" and len(df.groupby(df.index.date)) == len(df):
            return
        else:
            df_resampled = df_resampleds[interval]
            # Plot
            fig, axes = plt.subplots(
                len(columns), 1, figsize=(8, 2 * len(columns)), sharex=True
            )
            if len(columns) < 2:
                axes = [axes]

            for i, col in enumerate(columns):
                axes[i].plot(
                    df_resampled.index,
                    df_resampled[col],
                    label=col,
                    color=plt.cm.tab10(i),
                )
                if site_type == "amb":
                    try:
                        std_vn = VN_AMB_STD[col2var(col)][interval]
                        std_who = WHO_AMB_STD[col2var(col)][interval]

                        if std_vn is not None:
                            axes[i].axhline(
                                y=std_vn,
                                color="red",
                                linestyle="--",
                                linewidth=1,
                                label="Vietnam Standard",
                            )
                            get_stats(df_resampled, col, std_vn, interval, "Vietnam")
                        if std_who is not None:
                            axes[i].axhline(
                                y=std_who,
                                color="red",
                                linestyle="-",
                                linewidth=1,
                                label="WHO Standard",
                            )
                            get_stats(df_resampled, col, std_who, interval, "WHO")

                    except:
                        pass
                else:
                    try:
                        std = NON_AMB_STD[site_type][col2var(col)][interval]
                        axes[i].axhline(
                            y=std,
                            color="red",
                            linestyle="--",
                            linewidth=1,
                            label="Vietnam Standard",
                        )
                        get_stats(df_resampled, col, std, interval, "Vietnam")
                    except:
                        pass

                axes[i].legend(loc="upper right")
                axes[i].grid(True)
                axes[i].set_title(f"{subscript(col)} - ({interval})")

            axes[-1].set_xlabel("Date")
            for label in axes[-1].get_xticklabels():
                label.set_rotation(30)
            plt.tight_layout()
            plt.show()


def plot_correlation_heatmap(df):
    # Compute correlation matrix

    cols = df.columns
    cols = {c: c.split("-")[0].strip() for c in cols}
    corr = df.rename(columns=cols)
    corr = corr.corr().round(2)
    corr = corr.dropna(axis=1, how="all")
    corr = corr.dropna(axis=0, how="all")

    # Plot the heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="Spectral",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar=True,
        annot_kws={"size": 8},
    )
    plt.title("Correlation Heatmap of Pollutants and Meteorological Variables")
    plt.tight_layout()
    plt.show()


# %%

for f in LIST_XLSX:
    if "amb" not in f:
        print(f)
        df = site_prep(f)
        plot_air_quality_with_histograms(df)
        plot_air_quality_patterns(df)

        # v1.HY/HD/HL: this is 24-hour data
        # average daily values for pollutants except CO, O3: 18/24 hours of data
        # max 8-hour CO, O3 --> daily data
        # average annually values for pollutants: 90%

        plot_air_pollution(df, get_station_type(f))
        plot_correlation_heatmap(df)
# %%
