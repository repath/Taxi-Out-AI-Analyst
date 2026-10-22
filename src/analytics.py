"""Shared analytics tables used by the dashboard and AI Analyst."""

import pandas as pd


MONTHLY_STAT_COLUMNS = [
    "YEAR",
    "MONTH_NUM",
    "MONTH",
    "Flights",
    "Mean_Taxi_Out",
    "Median_Taxi_Out",
    "P25_Taxi_Out",
    "P50_Taxi_Out",
    "P75_Taxi_Out",
    "P90_Taxi_Out",
    "P95_Taxi_Out",
    "Max_Taxi_Out",
    "Min_Taxi_Out",
]


def build_monthly_statistics(flight_data: pd.DataFrame, percentiles=None) -> pd.DataFrame:
    """Builds the canonical month-level taxi-out statistics table."""
    if flight_data is None or flight_data.empty:
        return pd.DataFrame(columns=MONTHLY_STAT_COLUMNS)

    required_columns = {"FL_DATE", "TAXI_OUT"}
    if not required_columns.issubset(flight_data.columns):
        return pd.DataFrame(columns=MONTHLY_STAT_COLUMNS)

    data = flight_data[["FL_DATE", "TAXI_OUT"]].copy()
    data["FL_DATE"] = pd.to_datetime(data["FL_DATE"], dayfirst=True, errors="coerce")
    data["TAXI_OUT"] = pd.to_numeric(data["TAXI_OUT"], errors="coerce")
    data = data.dropna(subset=["FL_DATE", "TAXI_OUT"])
    if data.empty:
        return pd.DataFrame(columns=MONTHLY_STAT_COLUMNS)

    data["YEAR"] = data["FL_DATE"].dt.year.astype(int)
    data["MONTH_NUM"] = data["FL_DATE"].dt.month.astype(int)
    data["MONTH"] = data["FL_DATE"].dt.strftime("%b")

    group_columns = ["YEAR", "MONTH_NUM", "MONTH"]
    monthly_stats = (
        data.groupby(group_columns, as_index=False)
        .agg(
            Flights=("TAXI_OUT", "count"),
            Mean_Taxi_Out=("TAXI_OUT", "mean"),
            Median_Taxi_Out=("TAXI_OUT", "median"),
            P25_Taxi_Out=("TAXI_OUT", lambda values: values.quantile(0.25)),
            P50_Taxi_Out=("TAXI_OUT", lambda values: values.quantile(0.50)),
            P75_Taxi_Out=("TAXI_OUT", lambda values: values.quantile(0.75)),
            P90_Taxi_Out=("TAXI_OUT", lambda values: values.quantile(0.90)),
            P95_Taxi_Out=("TAXI_OUT", lambda values: values.quantile(0.95)),
            Max_Taxi_Out=("TAXI_OUT", "max"),
            Min_Taxi_Out=("TAXI_OUT", "min"),
        )
    )

    requested_percentiles = sorted({int(value) for value in (percentiles or []) if 0 <= int(value) <= 100})
    for percentile in requested_percentiles:
        column_name = f"P{percentile}_Taxi_Out"
        if column_name in monthly_stats.columns:
            continue
        percentile_stats = (
            data.groupby(group_columns)["TAXI_OUT"]
            .quantile(percentile / 100.0)
            .reset_index(name=column_name)
        )
        monthly_stats = monthly_stats.merge(percentile_stats, on=group_columns, how="left")

    return monthly_stats.sort_values(group_columns).reset_index(drop=True)