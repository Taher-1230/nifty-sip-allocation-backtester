from __future__ import annotations

import base64
import io
from pathlib import Path

import pandas as pd
import matplotlib
from flask import Flask, jsonify, render_template_string, request

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import dynamic asset allocation module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from project2_asset_allocation import simulate_asset_allocation as project2_simulate
from project2_asset_allocation import calculate_summary as project2_summary
from project2_asset_allocation import load_price_data as project2_load_data
from project2_asset_allocation import load_price_data_from_dataframe as project2_load_data_from_dataframe
from project2_asset_allocation import mark_investment_days
from project2_asset_allocation import load_config as project2_load_config

# Import benchmark functions
from benchmark import (
    simulate_nifty_benchmark,
    calculate_comparison_columns,
    calculate_drawdown_benefit_summary,
)

def calculate_sip_xirr(dates: list, amounts: list, fallback: float = 0.0) -> float:
    """
    Calculates annualized XIRR for a series of dated cash flows.
    Returns the percentage XIRR (e.g. 15.5 for 15.5%).
    amounts: negative for investments, positive for the final portfolio value.
    """
    try:
        from pyxirr import xirr
        res = xirr(dates, amounts)
        if res is not None and pd.notna(res):
            return res * 100
    except Exception:
        pass

    cashflows = []
    for date_value, amount_value in zip(dates, amounts):
        try:
            date = pd.to_datetime(date_value)
            amount = float(amount_value)
        except (TypeError, ValueError):
            continue
        if pd.isna(date) or pd.isna(amount) or amount == 0:
            continue
        cashflows.append((date, amount))

    if not cashflows:
        return fallback

    cashflows.sort(key=lambda item: item[0])
    first_date = cashflows[0][0]
    if cashflows[-1][0] == first_date:
        return fallback

    amounts_only = [amount for _, amount in cashflows]
    if not any(amount < 0 for amount in amounts_only) or not any(amount > 0 for amount in amounts_only):
        return fallback

    year_fractions = [
        (date - first_date).days / 365.25
        for date, _ in cashflows
    ]

    def xnpv(rate: float) -> float | None:
        if rate <= -1:
            return None
        try:
            base = 1 + rate
            return sum(
                amount / (base ** years)
                for years, (_, amount) in zip(year_fractions, cashflows)
            )
        except (OverflowError, ZeroDivisionError, ValueError):
            return None

    bracket_rates = [
        -0.999999,
        -0.99,
        -0.95,
        -0.9,
        -0.75,
        -0.5,
        -0.25,
        -0.1,
        0.0,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        500.0,
        1000.0,
    ]

    previous_rate = None
    previous_value = None
    for rate in bracket_rates:
        value = xnpv(rate)
        if value is None or not pd.notna(value):
            continue
        if abs(value) < 1e-7:
            return rate * 100
        if previous_value is not None and previous_value * value < 0:
            low_rate, high_rate = previous_rate, rate
            low_value, high_value = previous_value, value
            for _ in range(200):
                mid_rate = (low_rate + high_rate) / 2
                mid_value = xnpv(mid_rate)
                if mid_value is None or not pd.notna(mid_value):
                    break
                if abs(mid_value) < 1e-7 or abs(high_rate - low_rate) < 1e-12:
                    return mid_rate * 100
                if low_value * mid_value <= 0:
                    high_rate = mid_rate
                    high_value = mid_value
                else:
                    low_rate = mid_rate
                    low_value = mid_value
            return ((low_rate + high_rate) / 2) * 100
        previous_rate = rate
        previous_value = value

    return fallback



# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Available market segments for SIP analysis
MARKET_SEGMENTS = {
    "NIFTY 50": "NIFTY 50_Historical_PR_01011990to11102024.csv",
    "NIFTY NEXT 50": "NIFTY NEXT 50_Data.csv",
    "NIFTY MIDCAP 100": "NIFTY MIDCAP 100_Data.csv",
    "NIFTY 100": "NIFTY 100_Data.csv",
  "NIFTY SMALLCAP 250": "NIFTY SMALLCAP 250_Historical_PR_01012000to31122023.csv",
}

SECTOR_SEGMENTS = {
    "NIFTY BANK": "NIFTY BANK_Data.csv",
    "NIFTY IT": "NIFTY IT_Data.csv",
    "NIFTY PHARMA": "NIFTY PHARMA_Data.csv",
    "NIFTY AUTO": "NIFTY AUTO_Data.csv",
    "NIFTY FMCG": "NIFTY FMCG_Data.csv",
    "NIFTY INFRASTRUCTURE": "NIFTY INFRASTRUCTURE_Data.csv",
}

ALL_SEGMENTS = {**MARKET_SEGMENTS, **SECTOR_SEGMENTS}

DEFAULT_SEGMENT = "NIFTY 50"
DATA_FILE = DATA_DIR / MARKET_SEGMENTS[DEFAULT_SEGMENT]

MONTHLY_INVESTMENT = 1000

DATE_COL = "Date"

# Use "Close" if you assume you invest at closing price.
# Use "Open" if you assume you invest at market opening price.
PRICE_COL = "Close"


def parse_date_values(values: pd.Series) -> pd.Series:
    text_values = values.astype(str).str.strip()
    iso_date_mask = text_values.str.match(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+.*)?$",
        na=False,
    )
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    if iso_date_mask.any():
        parsed.loc[iso_date_mask] = pd.to_datetime(
            text_values.loc[iso_date_mask],
            yearfirst=True,
            errors="coerce",
        )
    if (~iso_date_mask).any():
        parsed.loc[~iso_date_mask] = pd.to_datetime(
            text_values.loc[~iso_date_mask],
            dayfirst=True,
            errors="coerce",
        )

    return parsed


def get_dataframe_date_bounds(df: pd.DataFrame, date_col: str = DATE_COL) -> dict:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataset")

    dates = parse_date_values(df[date_col]).dropna()
    if dates.empty:
        raise ValueError("No valid dates found in dataset")

    return {
        "min_date": dates.min().strftime("%Y-%m-%d"),
        "max_date": dates.max().strftime("%Y-%m-%d"),
    }


def get_dataset_file_date_bounds(file_name: str, date_col: str = DATE_COL) -> dict | None:
    if not file_name or file_name not in set(ALL_SEGMENTS.values()):
        return None

    file_path = DATA_DIR / file_name
    if not file_path.exists():
        return None

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    return get_dataframe_date_bounds(df, date_col=date_col)


def build_dataset_date_ranges(date_col: str = DATE_COL) -> dict:
    ranges = {}
    for file_name in sorted(set(ALL_SEGMENTS.values())):
        try:
            bounds = get_dataset_file_date_bounds(file_name, date_col=date_col)
        except Exception:
            bounds = None
        if bounds:
            ranges[file_name] = bounds
    return ranges


def apply_auto_start_date_from_bounds(config: dict, bounds: dict | None) -> None:
    if not bounds:
        return

    config["start_date"] = bounds["min_date"]
    config["_start_date_source"] = "auto"


# =========================
# LOAD DATA
# =========================

def load_nifty_data(file_path: Path) -> pd.DataFrame:
    """
    Load NIFTY 50 historical data from CSV or Excel.
    Expected columns:
    Index Name, Date, Open, High, Low, Close
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    print("Columns found in data file:")
    print(df.columns.tolist())

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert date column
    df[DATE_COL] = parse_date_values(df[DATE_COL])

    # Remove rows where date or price is missing
    df = df.dropna(subset=[DATE_COL, PRICE_COL])

    # Sort from oldest to newest
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return df


def load_nifty_data_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an uploaded dataset into the format used by the SIP strategy.

    The dataset must contain Date and Close columns.
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    required_columns = [DATE_COL, PRICE_COL]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
        "Uploaded dataset is not matching the required format. "
        f"Missing columns: {', '.join(missing_columns)}. "
        "Please make sure the file contains at least Date and Close columns."
        )

    df[DATE_COL] = parse_date_values(df[DATE_COL])
    df = df.dropna(subset=[DATE_COL, PRICE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df.empty:
        raise ValueError(
        "Uploaded dataset is not matching the required format. "
        "No usable rows were found after checking the Date and Close columns."
        )

    return df


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """
    Read a user-attached CSV or Excel file into a DataFrame.
    """

    if uploaded_file is None or not getattr(uploaded_file, "filename", ""):
      return None

    filename = uploaded_file.filename.lower()
    file_bytes = io.BytesIO(uploaded_file.read())

    if filename.endswith(".csv"):
        df = pd.read_csv(file_bytes)
    elif filename.endswith(".xls") or filename.endswith(".xlsx"):
        df = pd.read_excel(file_bytes)
    else:
        raise ValueError(
        "Uploaded dataset is not matching the required format. "
        "Please attach a CSV or Excel file."
        )

    return df


# =========================
# MONTHLY DATA
# =========================

def get_monthly_first_trading_day_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the first available trading day of every month.

    Example:
    If 1st is holiday, this automatically selects 2nd or next available date.
    """

    df = df.copy()

    df["YearMonth"] = df[DATE_COL].dt.to_period("M")

    monthly_df = df.groupby("YearMonth", as_index=False).first()

    monthly_df = monthly_df.drop(columns=["YearMonth"])

    return monthly_df


# =========================
# SIP CALCULATION
# =========================

def calculate_sip(
    monthly_df: pd.DataFrame,
    latest_price: float,
    latest_date
) -> pd.DataFrame:
    """
    Calculate monthly SIP investment.

    Units bought each month = Monthly Investment / NIFTY price
    """

    sip_df = monthly_df.copy()

    sip_df["Investment"] = MONTHLY_INVESTMENT

    sip_df["Units_Bought"] = sip_df["Investment"] / sip_df[PRICE_COL]

    sip_df["Total_Units"] = sip_df["Units_Bought"].cumsum()

    # Portfolio value on each monthly investment date
    sip_df["Portfolio_Value"] = sip_df["Total_Units"] * sip_df[PRICE_COL]

    # Total money invested up to each month
    sip_df["Total_Invested"] = sip_df["Investment"].cumsum()

    sip_df["Profit"] = sip_df["Portfolio_Value"] - sip_df["Total_Invested"]

    sip_df["ROI_%"] = (sip_df["Profit"] / sip_df["Total_Invested"]) * 100

    # Drawdown
    sip_df["Peak_Value"] = sip_df["Portfolio_Value"].cummax()

    sip_df["Drawdown_%"] = (
        (sip_df["Portfolio_Value"] - sip_df["Peak_Value"])
        / sip_df["Peak_Value"]
    ) * 100

    # Store latest details
    final_units = sip_df["Total_Units"].iloc[-1]
    latest_portfolio_value = final_units * latest_price

    sip_df.attrs["latest_date"] = latest_date
    sip_df.attrs["latest_price"] = latest_price
    sip_df.attrs["latest_portfolio_value"] = latest_portfolio_value

    return sip_df


def calculate_daily_sip_portfolio(
    df: pd.DataFrame,
    sip_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Value the SIP portfolio on every trading day.

    Purchases still happen only on the monthly SIP dates, but drawdown should be
    measured on all available trading days because the portfolio value moves
    between purchase dates too.
    """

    daily_df = df[[DATE_COL, PRICE_COL]].copy()
    purchases = sip_df[[DATE_COL, "Investment", "Units_Bought"]].copy()

    investment_by_date = purchases.set_index(DATE_COL)["Investment"]
    units_by_date = purchases.set_index(DATE_COL)["Units_Bought"]

    daily_df["Investment"] = daily_df[DATE_COL].map(investment_by_date).fillna(0.0)
    daily_df["Units_Bought"] = daily_df[DATE_COL].map(units_by_date).fillna(0.0)
    daily_df["Total_Units"] = daily_df["Units_Bought"].cumsum()
    daily_df["Portfolio_Value"] = daily_df["Total_Units"] * daily_df[PRICE_COL]
    daily_df["Total_Invested"] = daily_df["Investment"].cumsum()
    daily_df["Profit"] = daily_df["Portfolio_Value"] - daily_df["Total_Invested"]
    daily_df["ROI_%"] = (
        (daily_df["Profit"] / daily_df["Total_Invested"]) * 100
    ).where(daily_df["Total_Invested"] > 0, 0.0)
    daily_df["Peak_Value"] = daily_df["Portfolio_Value"].cummax()
    daily_df["Drawdown_%"] = (
        ((daily_df["Portfolio_Value"] - daily_df["Peak_Value"]) / daily_df["Peak_Value"]) * 100
    ).where(daily_df["Peak_Value"] > 0, 0.0)

    return daily_df


# =========================
# SUMMARY
# =========================

def calculate_summary(
    sip_df: pd.DataFrame,
    daily_sip_df: pd.DataFrame | None = None,
) -> dict:
    """
    Calculate final summary.
    """

    total_invested = sip_df["Total_Invested"].iloc[-1]
    current_value = sip_df.attrs["latest_portfolio_value"]
    total_months = len(sip_df)

    profit = current_value - total_invested

    roi = (profit / total_invested) * 100
    
    # Accurate XIRR Calculation
    c_dates = sip_df["Date"].tolist() + [sip_df.attrs["latest_date"]]
    c_amounts = [-MONTHLY_INVESTMENT] * len(sip_df) + [current_value]
    average_yearly_return = calculate_sip_xirr(c_dates, c_amounts)
    average_monthly_return = ((1 + average_yearly_return / 100) ** (1 / 12) - 1) * 100 if average_yearly_return > -100 else 0

    drawdown_df = daily_sip_df if daily_sip_df is not None else sip_df
    max_drawdown = drawdown_df["Drawdown_%"].min()

    summary = {
        "Latest Date": sip_df.attrs["latest_date"],
        "Latest NIFTY Price": sip_df.attrs["latest_price"],
        "Monthly Investment": MONTHLY_INVESTMENT,
      "Total Months Invested": total_months,
        "Total Invested": total_invested,
        "Current Value": current_value,
        "Profit": profit,
        "ROI %": roi,
      "Average Monthly Return %": average_monthly_return,
      "Average Yearly Return %": average_yearly_return,
        "Max Drawdown %": max_drawdown,
    }

    return summary


# =========================
# SAVE OUTPUT
# =========================

def save_outputs(sip_df: pd.DataFrame, summary: dict):
    """
    Save summary and detailed data into Excel, with CSV fallback.
    """

    output_file = OUTPUT_DIR / "nifty_sip_results.xlsx"
    summary_csv = OUTPUT_DIR / "nifty_sip_results_summary.csv"
    monthly_csv = OUTPUT_DIR / "nifty_sip_results_monthly.csv"

    summary_df = pd.DataFrame([summary])

    try:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            sip_df.to_excel(writer, sheet_name="Monthly SIP Data", index=False)

        output_message = f"Output saved to: {output_file}"
    except ImportError:
        summary_df.to_csv(summary_csv, index=False)
        sip_df.to_csv(monthly_csv, index=False)

        output_message = (
            "openpyxl is not installed, so CSV outputs were saved instead:\n"
            f"Summary: {summary_csv}\n"
            f"Monthly data: {monthly_csv}"
        )

    return output_message


# =========================
# WEB REPORT
# =========================

app = Flask(__name__)


def format_currency(value: float) -> str:
    return f"Rs {value:,.2f}"


def ui_icon(name: str) -> str:
  icons = {
    "chart": '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M4 19.5h16" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/><path d="M6 16.5V12" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/><path d="M11 16.5V8.5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/><path d="M16 16.5V5.5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/></svg>',
    "bank": '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M3 10h18" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/><path d="M5 10v8h14v-8" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.8"/><path d="M4 18h16" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/><path d="M6 9h12L12 4 6 9Z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.8"/></svg>',
    "spark": '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M13 3l-7 9h5l-1 9 7-10h-5l1-8Z" fill="currentColor"/></svg>',
    "shield": '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 3 19 6v5c0 4.9-3 8.8-7 10-4-1.2-7-5.1-7-10V6l7-3Z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.8"/><path d="M9.2 12.1 11 13.9 15 9.9" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"/></svg>',
    "refresh": '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M20 6v5h-5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"/><path d="M20 11a8 8 0 1 0 2 5.4" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/></svg>',
    "calendar": '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M7 3v3M17 3v3M4 8h16" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/><rect x="4" y="5" width="16" height="15" rx="2.5" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>',
    "strategy": '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 3 4.5 8v8L12 21l7.5-5V8L12 3Z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.8"/><path d="M12 8v8" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/><path d="M8.5 11.5 12 8l3.5 3.5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"/></svg>',
  }
  return icons.get(name, icons["chart"])


def figure_to_base64(figure) -> str:
    buffer = io.BytesIO()
    figure.patch.set_facecolor('#ffffff')
    for ax in figure.get_axes():
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#5f6b76', labelsize=9)
        ax.xaxis.label.set_color('#5f6b76')
        ax.yaxis.label.set_color('#5f6b76')
        ax.title.set_color('#154360')
        for spine in ax.spines.values():
            spine.set_color('#d9ddd8')
        ax.grid(True, alpha=0.25, color='#d9ddd8')
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor('#ffffff')
            legend.get_frame().set_edgecolor('#d9ddd8')
            for text in legend.get_texts():
                text.set_color('#263747')
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=160, facecolor='#ffffff')
    plt.close(figure)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def build_charts(
    df: pd.DataFrame,
    sip_df: pd.DataFrame,
    daily_sip_df: pd.DataFrame | None = None,
) -> dict:
    charts = {}
    portfolio_df = daily_sip_df if daily_sip_df is not None else sip_df

    price_figure, price_axis = plt.subplots(figsize=(12, 4))
    price_axis.plot(df[DATE_COL], df[PRICE_COL], color="#154360", linewidth=1.2)
    price_axis.set_title("NIFTY 50 Price Over Time", fontsize=13, fontweight="bold")
    price_axis.set_xlabel("Date")
    price_axis.set_ylabel("Index Level")
    price_axis.grid(True, alpha=0.25)
    charts["price"] = figure_to_base64(price_figure)

    wealth_figure, wealth_axis = plt.subplots(figsize=(12, 4))
    wealth_axis.plot(portfolio_df[DATE_COL], portfolio_df["Total_Invested"], label="Total Money Put In", color="#7f8c8d", linewidth=2)
    wealth_axis.plot(portfolio_df[DATE_COL], portfolio_df["Portfolio_Value"], label="Portfolio Value", color="#0b5345", linewidth=2.4)
    wealth_axis.set_title("How the Money Grew Over Time", fontsize=13, fontweight="bold")
    wealth_axis.set_xlabel("Date")
    wealth_axis.set_ylabel("Value")
    wealth_axis.legend()
    wealth_axis.grid(True, alpha=0.25)
    charts["wealth"] = figure_to_base64(wealth_figure)

    drawdown_figure, drawdown_axis = plt.subplots(figsize=(12, 4))
    drawdown_axis.fill_between(portfolio_df[DATE_COL], portfolio_df["Drawdown_%"], 0, color="#c0392b", alpha=0.25)
    drawdown_axis.plot(portfolio_df[DATE_COL], portfolio_df["Drawdown_%"], color="#922b21", linewidth=1.5)
    drawdown_axis.set_title("Drawdown: How Far the Portfolio Fell From Its Peak", fontsize=13, fontweight="bold")
    drawdown_axis.set_xlabel("Date")
    drawdown_axis.set_ylabel("Drawdown %")
    drawdown_axis.grid(True, alpha=0.25)
    charts["drawdown"] = figure_to_base64(drawdown_figure)

    return charts


def build_charts_project2(daily_df: pd.DataFrame, config: dict) -> dict:
    """Generate premium charts for the dynamic asset allocation investor report."""
    charts = {}

    segment_file = config.get("excel_file_name", "NIFTY 50_Historical_PR_01011990to11102024.csv")
    segment_name = "NIFTY 50"
    for name, file in MARKET_SEGMENTS.items():
        if file == segment_file:
            segment_name = name
            break

    # Chart 1: Portfolio Growth
    fig1, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.fill_between(daily_df["Date"], daily_df["Total Invested"], alpha=0.15, color="#7f8c8d")
    ax1.plot(daily_df["Date"], daily_df["Total Invested"], label="Money You Put In", color="#7f8c8d", linewidth=2, linestyle="--")
    ax1.fill_between(daily_df["Date"], daily_df["Total Asset Value"], alpha=0.12, color="#0b5345")
    ax1.plot(daily_df["Date"], daily_df["Total Asset Value"], label="What It Became", color="#0b5345", linewidth=2.5)
    ax1.set_title("Your Money Growing Over Time", fontsize=14, fontweight="bold", color="#154360", pad=12)
    ax1.set_xlabel("Year", fontsize=11, color="#5f6b76")
    ax1.set_ylabel("Value (Rs)", fontsize=11, color="#5f6b76")
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    charts["growth"] = figure_to_base64(fig1)

    # Chart 2: Equity vs Bond
    fig2, ax2 = plt.subplots(figsize=(12, 4.5))
    ax2.fill_between(daily_df["Date"], daily_df["Equity Value"], alpha=0.3, color="#0b5345", label=f"Stocks ({segment_name})")
    ax2.fill_between(daily_df["Date"], daily_df["Bond Value"], alpha=0.3, color="#2e86c1", label="Safe Bonds")
    ax2.plot(daily_df["Date"], daily_df["Equity Value"], color="#0b5345", linewidth=1.8)
    ax2.plot(daily_df["Date"], daily_df["Bond Value"], color="#2e86c1", linewidth=1.8)
    ax2.set_title("Stocks vs Bonds \u2014 How Each Bucket Grew", fontsize=14, fontweight="bold", color="#154360", pad=12)
    ax2.set_xlabel("Year", fontsize=11, color="#5f6b76")
    ax2.set_ylabel("Value (Rs)", fontsize=11, color="#5f6b76")
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    charts["split"] = figure_to_base64(fig2)

    # Chart 3: Market Drawdown
    fig3, ax3 = plt.subplots(figsize=(12, 3.5))
    dd_values = daily_df["Drawdown %"]
    ax3.fill_between(daily_df["Date"], dd_values, 0, color="#c0392b", alpha=0.2)
    ax3.plot(daily_df["Date"], dd_values, color="#922b21", linewidth=1.2)
    ax3.set_title(f"Market Drawdown \u2014 How Far {segment_name} Fell From Its Peak", fontsize=14, fontweight="bold", color="#154360", pad=12)
    ax3.set_xlabel("Year", fontsize=11, color="#5f6b76")
    ax3.set_ylabel("Drawdown %", fontsize=11, color="#5f6b76")
    ax3.grid(True, alpha=0.2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()
    charts["drawdown"] = figure_to_base64(fig3)

    # Chart 4: Strategy vs Benchmark Portfolio Value (if benchmark exists)
    if "Benchmark NIFTY Value" in daily_df.columns:
        fig4, ax4 = plt.subplots(figsize=(12, 4.5))
        ax4.plot(daily_df["Date"], daily_df["Total Asset Value"], label="Our Strategy (Stocks + Bonds)", color="#0b5345", linewidth=2.5)
        ax4.plot(daily_df["Date"], daily_df["Benchmark NIFTY Value"], label="100% Stocks (Benchmark)", color="#c0392b", linewidth=2.5)
        ax4.plot(daily_df["Date"], daily_df["Total Invested"], label="Money You Put In", color="#7f8c8d", linewidth=2, linestyle="--")
        ax4.set_title("Strategy vs 100% Stock Benchmark — Which Wins?", fontsize=14, fontweight="bold", color="#154360", pad=12)
        ax4.set_xlabel("Year", fontsize=11, color="#5f6b76")
        ax4.set_ylabel("Portfolio Value (Rs)", fontsize=11, color="#5f6b76")
        ax4.legend(fontsize=10, framealpha=0.9)
        ax4.grid(True, alpha=0.2)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        fig4.tight_layout()
        charts["vs_benchmark"] = figure_to_base64(fig4)

        # Chart 5: Drawdown Protection (Strategy vs Benchmark)
        fig5, ax5 = plt.subplots(figsize=(12, 4.5))
        ax5.fill_between(daily_df["Date"], daily_df["Strategy Portfolio Drawdown %"], 0, color="#0b5345", alpha=0.2, label="Our Strategy Drawdown")
        ax5.plot(daily_df["Date"], daily_df["Strategy Portfolio Drawdown %"], color="#0b5345", linewidth=2)
        ax5.fill_between(daily_df["Date"], daily_df["Benchmark Portfolio Drawdown %"], 0, color="#c0392b", alpha=0.2, label="100% Stock Drawdown")
        ax5.plot(daily_df["Date"], daily_df["Benchmark Portfolio Drawdown %"], color="#c0392b", linewidth=2)
        ax5.set_title("Drawdown Comparison — How Much Protection Did Bonds Provide?", fontsize=14, fontweight="bold", color="#154360", pad=12)
        ax5.set_xlabel("Year", fontsize=11, color="#5f6b76")
        ax5.set_ylabel("Drawdown %", fontsize=11, color="#5f6b76")
        ax5.legend(fontsize=10, framealpha=0.9)
        ax5.grid(True, alpha=0.2)
        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)
        fig5.tight_layout()
        charts["dd_protection"] = figure_to_base64(fig5)

        # Chart 6: Drawdown Benefit
        fig6, ax6 = plt.subplots(figsize=(12, 3.5))
        ax6.fill_between(daily_df["Date"], daily_df["Drawdown Benefit %"], 0, 
                         where=(daily_df["Drawdown Benefit %"] >= 0), 
                         color="#0b5345", alpha=0.3, label="Strategy Better")
        ax6.fill_between(daily_df["Date"], daily_df["Drawdown Benefit %"], 0,
                         where=(daily_df["Drawdown Benefit %"] < 0),
                         color="#c0392b", alpha=0.3, label="Benchmark Better")
        ax6.plot(daily_df["Date"], daily_df["Drawdown Benefit %"], color="#154360", linewidth=1.5)
        ax6.axhline(y=0, color="#333333", linestyle="-", linewidth=0.8)
        ax6.set_title("Drawdown Benefit % - How Much Better Is Our Strategy?", fontsize=14, fontweight="bold", color="#154360", pad=12)
        ax6.set_xlabel("Year", fontsize=11, color="#5f6b76")
        ax6.set_ylabel("Drawdown Benefit %", fontsize=11, color="#5f6b76")
        ax6.legend(fontsize=10, framealpha=0.9)
        ax6.grid(True, alpha=0.2)
        ax6.spines["top"].set_visible(False)
        ax6.spines["right"].set_visible(False)
        fig6.tight_layout()
        charts["dd_benefit"] = figure_to_base64(fig6)

    return charts


def render_table_rows(sample_rows: list[dict]) -> str:
    rows_html = []
    for row in sample_rows:
        rows_html.append(
            "<tr>"
            f"<td>{row[DATE_COL]}</td>"
            f"<td>{row[PRICE_COL]:,.2f}</td>"
            f"<td>{row['Investment']:,.0f}</td>"
            f"<td>{row['Units_Bought']:.4f}</td>"
            f"<td>{row['Total_Units']:.4f}</td>"
            f"<td>{row['Portfolio_Value']:,.2f}</td>"
            "</tr>"
        )
    return "".join(rows_html)


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NIFTY Strategy Suite | Investor Reports</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0a0a0a;
      --panel: #111111;
      --panel-2: #1a1a1a;
      --text: #e8e8e8;
      --muted: #777777;
      --accent: #ffffff;
      --accent-2: #cccccc;
      --border: #2a2a2a;
      --soft: #161616;
      --danger: #ff4444;
      --green: #00d26a;
      --red: #ff3b3b;
      --shadow: 0 4px 30px rgba(0,0,0,0.5);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text);
      background: var(--bg);
      -webkit-font-smoothing: antialiased;
    }
    .container { max-width: 1280px; margin: 0 auto; padding: 24px 20px 60px; }
    .hero {
      background: linear-gradient(160deg, #111 0%, #0a0a0a 50%, #111 100%);
      color: white;
      border-radius: 2px;
      padding: 40px 36px;
      border: 1px solid var(--border);
      overflow: hidden;
      position: relative;
    }
    .hero::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    }
    .eyebrow { text-transform: uppercase; letter-spacing: 0.25em; font-size: 11px; color: var(--muted); margin-bottom: 12px; font-family: 'JetBrains Mono', monospace; }
    h1 { margin: 0 0 12px; font-size: clamp(28px, 3.5vw, 42px); font-weight: 900; letter-spacing: -0.02em; }
    .hero p { margin: 0; max-width: 900px; line-height: 1.7; font-size: 15px; color: #999; }
    .toolbar { margin-top: 24px; display: flex; flex-wrap: wrap; gap: 14px; align-items: end; }
    .field {
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      border-radius: 2px;
      padding: 14px 16px;
      min-width: 200px;
    }
    .field label { display: block; font-size: 10px; color: var(--muted); margin-bottom: 6px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'JetBrains Mono', monospace; }
    .field select {
      width: 100%;
      border: 0;
      outline: none;
      background: rgba(255,255,255,0.06);
      color: white;
      font-size: 13px;
      padding: 10px;
      border-radius: 2px;
      cursor: pointer;
      font-family: 'Inter', sans-serif;
    }
    .field select option { background: #111; color: white; }
    .field select optgroup { background: #111; color: #999; font-weight: 700; }
    .button {
      border: 1px solid #fff;
      background: #ffffff;
      color: #000000;
      border-radius: 2px;
      font-size: 14px;
      font-weight: 800;
      padding: 14px 28px;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      transition: all 0.2s ease;
      font-family: 'Inter', sans-serif;
    }
    .button:hover { background: #000; color: #fff; border-color: #fff; }
    .notice, .card, .section, .table-wrap {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 2px;
    }
    .notice { margin-top: 18px; padding: 20px 24px; color: #aaa; line-height: 1.7; font-size: 14px; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px; margin-top: 18px; background: var(--border); border: 1px solid var(--border); border-radius: 2px; overflow: hidden; }
    .card { padding: 20px; border-top: none; border: none; border-radius: 0; background: var(--panel); }
    .card .label { color: var(--muted); font-size: 10px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'JetBrains Mono', monospace; }
    .card .value { font-size: 24px; font-weight: 800; line-height: 1.1; color: #fff; font-family: 'JetBrains Mono', monospace; }
    .section { margin-top: 18px; padding: 28px; }
    .section h2 { margin: 0 0 14px; font-size: 18px; font-weight: 800; color: #fff; text-transform: uppercase; letter-spacing: 0.05em; }
    .section p, .section li { line-height: 1.75; color: #aaa; font-size: 14px; }
    .section ul { margin: 8px 0 0 18px; padding: 0; }
    .section li::marker { color: var(--muted); }
    .split { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; align-items: start; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    .chip { background: transparent; border: 1px solid var(--border); padding: 8px 14px; border-radius: 2px; color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; font-family: 'JetBrains Mono', monospace; }
    .ui-icon { display: inline-flex; width: 16px; height: 16px; margin-right: 8px; vertical-align: -2px; color: var(--muted); }
    .ui-icon svg { width: 100%; height: 100%; display: block; }
    .chart { margin-top: 16px; overflow: hidden; border: 1px solid var(--border); border-radius: 2px; background: var(--panel); }
    .chart img { display: block; width: 100%; height: auto; }
    .table-wrap { margin-top: 16px; overflow: hidden; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }
    th { background: var(--soft); font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-family: 'JetBrains Mono', monospace; }
    td { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #ccc; }
    tbody tr:nth-child(even) { background: rgba(255,255,255,0.02); }
    tbody tr:hover { background: rgba(255,255,255,0.04); }
    .footer-note { color: var(--muted); font-size: 12px; margin-top: 8px; font-family: 'JetBrains Mono', monospace; }
    .error { margin-top: 18px; padding: 14px 18px; background: rgba(255,60,60,0.08); color: var(--danger); border-radius: 2px; border: 1px solid rgba(255,60,60,0.2); }
    .cfg-panel { background:var(--panel); border:1px solid var(--border); border-radius:2px; margin-top:18px; padding:28px; }
    .cfg-panel h3 { margin:0 0 14px; font-size:13px; color:#fff; border-bottom:1px solid var(--border); padding-bottom:10px; text-transform:uppercase; letter-spacing:0.08em; font-weight:800; }
    .cfg-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px; margin-bottom:18px; }
    .cfg-f label { display:block; font-size:10px; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; font-family:'JetBrains Mono',monospace; }
    .cfg-f input,.cfg-f select { width:100%; padding:9px 10px; border:1px solid var(--border); border-radius:2px; font-size:13px; background:var(--soft); color:var(--text); outline:none; }
    .cfg-f input:focus,.cfg-f select:focus { border-color:#555; box-shadow:0 0 0 2px rgba(255,255,255,.05); }
    .rt { width:100%; border-collapse:collapse; margin-bottom:8px; }
    .rt th { background:var(--soft); font-size:10px; padding:8px; text-align:left; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; font-family:'JetBrains Mono',monospace; }
    .rt td { padding:5px 6px; border-bottom:1px solid var(--border); }
    .rt input { width:100%; padding:7px; border:1px solid var(--border); border-radius:2px; font-size:12px; background:var(--soft); color:var(--text); }
    .rt input[type=checkbox] { width:auto; accent-color:#fff; }
    .ba,.br { border:none; padding:6px 14px; border-radius:2px; font-size:11px; font-weight:700; cursor:pointer; text-transform:uppercase; letter-spacing:0.05em; }
    .ba { background:#fff; color:#000; }
    .ba:hover { background:#ccc; }
    .br { background:#333; color:#ff4444; }
    .br:hover { background:#444; }
    .cfg-chk { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
    .cfg-chk input[type=checkbox] { width:16px; height:16px; accent-color:#fff; }
    .cfg-chk label { font-size:13px; margin:0; color:#aaa; }
    .cfg-sec { margin-bottom:18px; }
    strong { color: #fff; }
    @media (max-width: 900px) {
      .grid, .split { grid-template-columns: 1fr; }
      .hero { padding: 24px; }
      .button { width: 100%; }
      .cfg-grid { grid-template-columns: 1fr; }
    }
    /* Restored light report theme */
    :root {
      --bg: #f5f3ef;
      --panel: #ffffff;
      --text: #13202b;
      --muted: #5f6b76;
      --accent: #0b5345;
      --accent-2: #154360;
      --border: #d9ddd8;
      --soft: #eef2ef;
      --danger: #a93226;
      --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
    }
    body {
      font-family: Arial, Helvetica, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(21, 67, 96, 0.12), transparent 32%),
        radial-gradient(circle at top right, rgba(11, 83, 69, 0.12), transparent 28%),
        linear-gradient(180deg, #faf8f5 0%, #f1f5f0 100%);
    }
    .container { max-width: 1220px; padding: 28px 18px 56px; }
    .hero {
      background: linear-gradient(135deg, #102a43 0%, #0b5345 56%, #154360 100%);
      color: white;
      border: 0;
      border-radius: 28px;
      padding: 34px;
      box-shadow: var(--shadow);
    }
    .hero::before { display: none; }
    .eyebrow {
      color: rgba(255,255,255,0.72);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0.18em;
    }
    h1 {
      color: white;
      letter-spacing: 0;
      text-shadow: none;
    }
    .hero p { color: rgba(255,255,255,0.88); }
    .toolbar { align-items: end; }
    .field {
      background: rgba(255,255,255,0.10);
      border: 1px solid rgba(255,255,255,0.22);
      border-radius: 16px;
    }
    .field label {
      color: rgba(255,255,255,0.72);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0.12em;
    }
    .field select {
      background: rgba(255,255,255,0.92);
      color: #13202b;
      border-radius: 10px;
      font-family: Arial, Helvetica, sans-serif;
    }
    .field select option,
    .field select optgroup {
      background: #ffffff;
      color: #13202b;
    }
    .button {
      background: #ffffff;
      border: 0;
      color: #0b5345;
      border-radius: 999px;
      box-shadow: 0 12px 28px rgba(0,0,0,0.14);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0.03em;
    }
    .button:hover {
      background: #eef2ef;
      color: #154360;
      border: 0;
    }
    .notice, .card, .section, .table-wrap, .cfg-panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }
    .notice { color: var(--text); }
    .grid {
      gap: 16px;
      background: transparent;
      border: 0;
      border-radius: 0;
    }
    .card {
      border: 1px solid var(--border);
      border-radius: 18px;
      background: #ffffff;
      box-shadow: var(--shadow);
    }
    .card .label {
      color: var(--muted);
      font-family: Arial, Helvetica, sans-serif;
      letter-spacing: 0.08em;
    }
    .card .value {
      color: var(--accent-2);
      font-family: Arial, Helvetica, sans-serif;
    }
    .section h2, .cfg-panel h3 {
      color: var(--accent-2);
      letter-spacing: 0;
    }
    .section p, .section li { color: #263747; }
    .chip {
      background: var(--soft);
      border-color: #d9ddd8;
      color: var(--accent-2);
      border-radius: 999px;
      font-family: Arial, Helvetica, sans-serif;
    }
    .ui-icon { color: var(--accent); }
    .chart { background: #ffffff; border-color: var(--border); border-radius: 18px; }
    th { background: var(--soft); color: var(--accent-2); font-family: Arial, Helvetica, sans-serif; }
    td { color: #263747; font-family: Arial, Helvetica, sans-serif; }
    tbody tr:nth-child(even) { background: #f8faf8; }
    tbody tr:hover { background: #eef2ef; }
    .footer-note { color: var(--muted); font-family: Arial, Helvetica, sans-serif; }
    .cfg-f label {
      color: var(--muted);
      font-family: Arial, Helvetica, sans-serif;
    }
    .cfg-f input,.cfg-f select,.rt input {
      background: #ffffff;
      color: var(--text);
      border-color: var(--border);
      border-radius: 10px;
    }
    .rt th {
      background: var(--soft);
      color: var(--accent-2);
      font-family: Arial, Helvetica, sans-serif;
    }
    .ba { background:#0b5345; color:#fff; }
    .ba:hover { background:#154360; }
    .br { background:#fff2f0; color:#a93226; }
    .br:hover { background:#fde2dc; }
    .cfg-chk label { color: var(--text); }
    strong { color: var(--accent-2); }
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <div class="eyebrow">
        {% if selected_project == 'project1' %}
          SIP Wealth Strategy
        {% else %}
          Dynamic Asset Allocation
        {% endif %}
      </div>
      <h1>
        {% if selected_project == 'project1' %}
          NIFTY SIP Investor Report
        {% else %}
          NIFTY Asset Allocation Report
        {% endif %}
      </h1>
      <p>
        {% if selected_project == 'project1' %}
          This page explains, in very simple language, what happens when you invest the same amount every month in NIFTY 50.
          It shows where the money goes, how units are bought, how compounding builds value, and what the final outcome looks like.
        {% else %}
          This page demonstrates a quant-based strategy that dynamically allocates your monthly investment between the selected market segment
          and a bond portfolio (6.5% annual return) based on the market's drawdown from its all-time high. The strategy automatically buys
          more equity when the market falls and resets when it recovers to new highs.
        {% endif %}
      </p>
      <form method="post" id="main-form" enctype="multipart/form-data">
        <div class="toolbar">
          <div class="field">
            <label for="project">Strategy</label>
            <select id="project" name="project" onchange="updateProjectUI(this.value)">
              <option value="project1" {% if selected_project == 'project1' %}selected{% endif %}>SIP Wealth Strategy</option>
              <option value="project2" {% if selected_project == 'project2' %}selected{% endif %}>Dynamic Asset Allocation</option>
            </select>
          </div>
          <div class="field" id="segment-field">
            <label for="segment">Market Segment</label>
            <select id="segment" name="segment">
              <optgroup label="Market-Cap Segments">
                {% for seg_name in market_segments.keys() %}
                  <option value="{{ seg_name }}" {% if seg_name == selected_segment %}selected{% endif %}>{{ seg_name }}</option>
                {% endfor %}
              </optgroup>
              <optgroup label="Sector Segments">
                {% for seg_name in sector_segments.keys() %}
                  <option value="{{ seg_name }}" {% if seg_name == selected_segment %}selected{% endif %}>{{ seg_name }}</option>
                {% endfor %}
              </optgroup>
            </select>
          </div>
          <div class="field" id="dataset-field">
            <label for="dataset_file">Attach Dataset</label>
            <input id="dataset_file" name="dataset_file" type="file" accept=".csv,.xls,.xlsx" style="width:100%;color:white;background:rgba(255,255,255,0.08);padding:10px;border-radius:8px;border:0;" />
          </div>
          <button class="button" type="submit">Run Detailed Report</button>
        </div>
    </div>

    <!-- Dynamic allocation config panel -->
    <div id="p2-config" class="cfg-panel" {% if selected_project != 'project2' %}style="display:none"{% endif %}>
          <h3><span class="ui-icon">{{ ui_icon('chart') | safe }}</span> Data &amp; Investment Settings</h3>
      <div class="cfg-grid">
        <div class="cfg-f"><label>Market Segment</label>
          <select name="p2_excel_file" id="p2_excel_file">
            {% for name, file in market_segments.items() %}
              <option value="{{ file }}" {% if file == p2.get('excel_file_name','') %}selected{% endif %}>{{ name }}</option>
            {% endfor %}
          </select>
          <input type="hidden" name="p2_previous_excel_file" value="{{ p2.get('excel_file_name','') }}">
          <input type="hidden" name="p2_start_date_source" id="p2_start_date_source" value="{{ p2.get('_start_date_source','auto') }}">
        </div>
        <div class="cfg-f"><label>Start Date (YYYY-MM-DD)</label><input type="date" id="p2_start_date" name="p2_start_date" value="{{ p2.get('start_date','') or '' }}" placeholder="e.g. 1999-01-01"></div>
        <div class="cfg-f"><label>End Date (YYYY-MM-DD)</label><input type="date" id="p2_end_date" name="p2_end_date" value="{{ p2.get('end_date','') or '' }}" placeholder="Leave empty for latest"></div>
        <div class="cfg-f"><label>Price Column</label><input type="text" name="p2_price_col" value="{{ p2.get('price_column','Close') }}"></div>
        <div class="cfg-f"><label>Investment Amount (₹)</label><input type="number" name="p2_invest_amt" value="{{ p2.get('monthly_investment',1000) }}" step="100"></div>
        <div class="cfg-f"><label>Frequency</label>
          <select name="p2_freq">
            <option value="monthly" {{ 'selected' if p2.get('investment_frequency')=='monthly' }}>Monthly</option>
            <option value="weekly" {{ 'selected' if p2.get('investment_frequency')=='weekly' }}>Weekly</option>
            <option value="daily" {{ 'selected' if p2.get('investment_frequency')=='daily' }}>Daily</option>
          </select></div>
        <div class="cfg-f"><label>Day Rule</label>
          <select name="p2_day_rule">
            <option value="first_trading_day" {{ 'selected' if p2.get('investment_day_rule')=='first_trading_day' }}>First Trading Day</option>
            <option value="last_trading_day" {{ 'selected' if p2.get('investment_day_rule')=='last_trading_day' }}>Last Trading Day</option>
          </select></div>
      </div>
      <h3><span class="ui-icon">{{ ui_icon('bank') | safe }}</span> Bond Settings</h3>
      <div class="cfg-grid">
        <div class="cfg-f"><label>Bond Annual Return (%)</label><input type="number" name="p2_bond_ret" value="{{ (p2.get('bond_annual_return',0.065)*100)|round(2) }}" step="0.1"></div>
        <div class="cfg-f"><label>Bond Compounding</label>
          <select name="p2_bond_comp">
            <option value="continuous" {{ 'selected' if p2.get('bond_compounding')=='continuous' }}>Continuous</option>
            <option value="annual" {{ 'selected' if p2.get('bond_compounding')=='annual' }}>Annual</option>
            <option value="daily" {{ 'selected' if p2.get('bond_compounding')=='daily' }}>Daily</option>
          </select></div>
        <div class="cfg-f"><label>Trigger Mode</label>
          <select name="p2_trigger_mode">
            <option value="highest_only" {{ 'selected' if p2.get('trigger_mode')=='highest_only' }}>Highest Only</option>
            <option value="all_crossed" {{ 'selected' if p2.get('trigger_mode')=='all_crossed' }}>All Crossed</option>
          </select></div>
        <div class="cfg-f"><label>Output File Name</label><input type="text" name="p2_output" value="{{ p2.get('output_file_name','dynamic_asset_allocation_results.xlsx') }}"></div>
      </div>
      <h3><span class="ui-icon">{{ ui_icon('strategy') | safe }}</span> Allocation Rules <button type="button" class="ba" id="btn-add-alloc">+ Add</button></h3>
      <input type="hidden" id="alloc_count" name="alloc_count" value="{{ p2.get('allocation_rules',[])|length }}">
      <table class="rt"><thead><tr><th>Min DD %</th><th>Max DD %</th><th>Equity %</th><th>Bond %</th><th></th></tr></thead>
      <tbody id="alloc-body">
        {% for r in p2.get('allocation_rules',[]) %}
        <tr>
          <td><input type="number" name="alloc_min_{{ loop.index0 }}" value="{{ (r.min_drawdown*100)|round(1) }}" step="1"></td>
          <td><input type="number" name="alloc_max_{{ loop.index0 }}" value="{{ (r.max_drawdown*100)|round(1) }}" step="1"></td>
          <td><input type="number" name="alloc_eq_{{ loop.index0 }}" value="{{ (r.equity_allocation*100)|round(1) }}" step="1"></td>
          <td><input type="number" name="alloc_bd_{{ loop.index0 }}" value="{{ (r.bond_allocation*100)|round(1) }}" step="1"></td>
          <td><button type="button" class="br btn-rm" data-counter="alloc_count">✕</button></td>
        </tr>
        {% endfor %}
      </tbody></table>
      <h3><span class="ui-icon">{{ ui_icon('refresh') | safe }}</span> Bond Shift Rules <button type="button" class="ba" id="btn-add-bshift">+ Add</button></h3>
      <input type="hidden" id="bshift_count" name="bshift_count" value="{{ p2.get('bond_shift_rules',[])|length }}">
      <table class="rt"><thead><tr><th>Trigger DD %</th><th>Shift %</th><th>Once/ATH?</th><th></th></tr></thead>
      <tbody id="bshift-body">
        {% for r in p2.get('bond_shift_rules',[]) %}
        <tr>
          <td><input type="number" name="bshift_tr_{{ loop.index0 }}" value="{{ (r.trigger_drawdown*100)|round(1) }}" step="1"></td>
          <td><input type="number" name="bshift_pct_{{ loop.index0 }}" value="{{ (r.bond_shift_percentage*100)|round(1) }}" step="1"></td>
          <td><input type="checkbox" name="bshift_once_{{ loop.index0 }}" {{ 'checked' if r.trigger_once_per_ath_cycle }}></td>
          <td><button type="button" class="br btn-rm" data-counter="bshift_count">✕</button></td>
        </tr>
        {% endfor %}
      </tbody></table>
      <h3><span class="ui-icon">{{ ui_icon('chart') | safe }}</span> Metrics</h3>
      <div style="display:flex;flex-wrap:wrap;gap:16px">
        <div class="cfg-chk"><input type="checkbox" name="p2_show_roi" id="sr" {{ 'checked' if p2.get('metrics',{}).get('show_roi',true) }}><label for="sr">Show ROI</label></div>
        <div class="cfg-chk"><input type="checkbox" name="p2_show_dd" id="sd" {{ 'checked' if p2.get('metrics',{}).get('show_max_drawdown',true) }}><label for="sd">Show Max DD</label></div>
        <div class="cfg-chk"><input type="checkbox" name="p2_show_cagr" id="sc" {{ 'checked' if p2.get('metrics',{}).get('show_cagr',true) }}><label for="sc">Show CAGR</label></div>
        <div class="cfg-chk"><input type="checkbox" name="p2_show_split" id="ss" {{ 'checked' if p2.get('metrics',{}).get('show_equity_bond_split',true) }}><label for="ss">Show Equity/Bond Split</label></div>
      </div>
      <div style="margin-top:22px;text-align:center">
        <button class="button" type="submit" style="font-size:18px;padding:18px 40px"><span class="ui-icon">{{ ui_icon('refresh') | safe }}</span> Update &amp; Run Report</button>
      </div>
    </div>
    </form>
    <script>
      const p2DatasetDateRanges = {{ dataset_date_ranges | tojson }};

      function setP2DateFields(minDate, maxDate) {
        const startDateField = document.getElementById('p2_start_date');
        const endDateField = document.getElementById('p2_end_date');
        const sourceField = document.getElementById('p2_start_date_source');

        if (!startDateField || !minDate) return;

        startDateField.value = minDate;
        startDateField.min = minDate;
        if (maxDate) startDateField.max = maxDate;

        if (endDateField) {
          endDateField.min = minDate;
          if (maxDate) endDateField.max = maxDate;
          if (endDateField.value && (endDateField.value < minDate || (maxDate && endDateField.value > maxDate))) {
            endDateField.value = '';
          }
        }

        if (sourceField) sourceField.value = 'auto';
      }

      function setP2DateFieldsForFile(fileName) {
        const bounds = p2DatasetDateRanges[fileName];
        if (!bounds) return;
        setP2DateFields(bounds.min_date, bounds.max_date);
      }

      function updateProjectUI(v){
        document.getElementById('segment-field').style.display=v==='project1'?'block':'none';
        document.getElementById('p2-config').style.display=v==='project2'?'block':'none';
      }
      window.addEventListener('DOMContentLoaded',function(){
        updateProjectUI('{{ selected_project }}');

        const p2FileSelect = document.getElementById('p2_excel_file');
        const p2StartDate = document.getElementById('p2_start_date');
        const p2StartDateSource = document.getElementById('p2_start_date_source');

        if (p2FileSelect) {
          p2FileSelect.addEventListener('change', function() {
            setP2DateFieldsForFile(this.value);
          });

          if (!p2StartDateSource || p2StartDateSource.value !== 'user') {
            setP2DateFieldsForFile(p2FileSelect.value);
          }
        }

        if (p2StartDate && p2StartDateSource) {
          p2StartDate.addEventListener('input', function() {
            p2StartDateSource.value = 'user';
          });
        }

        /* --- Add Allocation Row --- */
        document.getElementById('btn-add-alloc').addEventListener('click',function(){
          var tb=document.getElementById('alloc-body');
          var i=tb.rows.length;
          var tr=document.createElement('tr');
          tr.innerHTML='<td><input type="number" name="alloc_min_'+i+'" value="0" step="1"></td>'
            +'<td><input type="number" name="alloc_max_'+i+'" value="100" step="1"></td>'
            +'<td><input type="number" name="alloc_eq_'+i+'" value="80" step="1"></td>'
            +'<td><input type="number" name="alloc_bd_'+i+'" value="20" step="1"></td>'
            +'<td><button type="button" class="br btn-rm" data-counter="alloc_count">\u2715</button></td>';
          tb.appendChild(tr);
          document.getElementById('alloc_count').value=i+1;
        });

        /* --- Add Bond Shift Row --- */
        document.getElementById('btn-add-bshift').addEventListener('click',function(){
          var tb=document.getElementById('bshift-body');
          var i=tb.rows.length;
          var tr=document.createElement('tr');
          tr.innerHTML='<td><input type="number" name="bshift_tr_'+i+'" value="10" step="1"></td>'
            +'<td><input type="number" name="bshift_pct_'+i+'" value="20" step="1"></td>'
            +'<td><input type="checkbox" name="bshift_once_'+i+'" checked></td>'
            +'<td><button type="button" class="br btn-rm" data-counter="bshift_count">\u2715</button></td>';
          tb.appendChild(tr);
          document.getElementById('bshift_count').value=i+1;
        });

        /* --- Remove Row (event delegation) --- */
        document.addEventListener('click',function(e){
          var btn=e.target;
          if(!btn.classList.contains('btn-rm')) return;
          var tr=btn.closest('tr');
          var tbody=tr.parentNode;
          var countId=btn.getAttribute('data-counter');
          tbody.removeChild(tr);
          /* reindex all rows */
          var rows=tbody.querySelectorAll('tr');
          for(var idx=0;idx<rows.length;idx++){
            var inputs=rows[idx].querySelectorAll('input');
            for(var j=0;j<inputs.length;j++){
              var nm=inputs[j].name;
              if(nm) inputs[j].name=nm.replace(/_[0-9]+$/,'_'+idx);
            }
          }
          document.getElementById(countId).value=rows.length;
        });
      });
    </script>

    {% if error %}
      <script>
        alert({{ error | tojson }});
      </script>
      <div class="error">{{ error }}</div>
    {% endif %}

    {% if report %}
      {% if selected_project == 'project2' %}
      <!-- ========== INVESTOR PITCH REPORT ========== -->
      <div class="notice">
        {{ report.story | safe }}
          <p style="margin-top: 12px; font-weight: 700; color: #154360;">
          <span class="ui-icon">{{ ui_icon('calendar') | safe }}</span> Investment Period: <strong>{{ report.start_month_year }}</strong> to <strong>{{ report.end_month_year }}</strong>
        </p>
      </div>

      <div class="grid">
        {% for card in cards %}
          <div class="card">
            <div class="label">{{ card.label }}</div>
            <div class="value">{{ card.value }}</div>
          </div>
        {% endfor %}
      </div>

      <div class="section split">
        <div>
          <h2>How this strategy works</h2>
          <p>
            {{ report.frequency_intro }}, a fixed amount is invested automatically. A smart system decides how much goes into
            stocks ({{ report.segment_name }}) and how much into safe bonds - based on how the market is behaving.
          </p>
          <ul>
            {% for item in report.how_it_works %}
              <li>{{ item | safe }}</li>
            {% endfor %}
          </ul>
          <div class="chips">
            <div class="chip">Smart allocation</div>
            <div class="chip">Buy-the-dip automation</div>
            <div class="chip">Bond safety cushion</div>
            <div class="chip">ATH reset logic</div>
          </div>
        </div>
        <div>
          <h2>Plain-English result</h2>
          <ul>
            {% for item in report.plain_english %}
              <li>{{ item }}</li>
            {% endfor %}
          </ul>
          <p class="footer-note">Latest market date used: {{ report.latest_date.strftime('%d-%b-%Y') }}</p>
          <p class="footer-note">Data points loaded: {{ report.rows }} trading days; {{ report.investment_count_label }}: {{ report.months }}.</p>
        </div>
      </div>

      <div class="section">
        <h2>What this means for an investor</h2>
        <p>{{ report.investor_view }}</p>
        <ul>
          {% for item in report.potential %}
            <li>{{ item | safe }}</li>
          {% endfor %}
        </ul>
      </div>

      <div class="section">
        <h2>The kitchen-table explanation</h2>
        <p style="font-style: italic; line-height: 1.8; color: #154360;">
          "{{ report.kitchen_table }}"
        </p>
      </div>

      {% if report.alloc_narrative %}
      <div class="section split">
        <div>
          <h2>How the system splits your money</h2>
          <p>
            The allocation between stocks and bonds changes automatically based on how far the market has fallen from its all-time high.
          </p>
          <ul>
            {% for rule in report.alloc_narrative %}
              <li>{{ rule }}</li>
            {% endfor %}
          </ul>
        </div>
        <div>
          <h2>Buy-the-dip triggers</h2>
          <p>
            When markets fall hard, the system moves bond savings into stocks to buy at lower prices.
            These triggers reset when the market reaches a new all-time high.
          </p>
          <ul>
            {% for shift in report.shift_narrative %}
              <li>{{ shift }}</li>
            {% endfor %}
          </ul>
        </div>
      </div>
      {% endif %}

      {% if report.benchmark_stats %}
      <div class="section" style="margin-top: 32px; background: linear-gradient(135deg, #fdfefe 0%, #f4f6f7 100%); padding: 32px; border-radius: 12px; border: 1px solid #d5dbdb;">
        <h2 style="margin-top:0; color: #154360;">🛡️ Benchmark Comparison (The Protection Benefit)</h2>
        <p style="margin-bottom: 24px;">
          How did our strategy (Stocks + Bonds) perform during crashes compared to investing 100% only in {{ report.segment_name }}?
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px;">
          <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #c0392b; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 13px; font-weight: 600; color: #7f8c8d; text-transform: uppercase; margin-bottom: 8px;">Severe Crash Scenario (â‰¥ 30% drop)</div>
            <p style="margin: 0; font-size: 15px; line-height: 1.5;">When the market ({{ report.segment_name }}) fell by 30% or more from its peak, our strategy's worst fall was only <strong>{{ "%.2f"|format(report.benchmark_stats.get('Worst Strategy Drawdown When Benchmark DD >= 30%', 0)) }}%</strong>.</p>
          </div>
          <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #f39c12; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 13px; font-weight: 600; color: #7f8c8d; text-transform: uppercase; margin-bottom: 8px;">Max Drawdown Comparison</div>
            <p style="margin: 0; font-size: 15px; line-height: 1.5;">The worst the market EVER fell was <strong>{{ "%.2f"|format(report.benchmark_stats.get('Benchmark Max Drawdown %', 0)) }}%</strong>.<br>The worst our strategy fell was <strong>{{ "%.2f"|format(report.summary.get('Max Drawdown %', 0)) }}%</strong>.<br><em>That's a protection benefit of <strong>{{ "%.2f"|format(report.benchmark_stats.get('Max Drawdown Benefit %', 0)) }}%</strong>.</em></p>
          </div>
          <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #27ae60; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 13px; font-weight: 600; color: #7f8c8d; text-transform: uppercase; margin-bottom: 8px;">Returns & ROI</div>
            <p style="margin: 0; font-size: 15px; line-height: 1.5;">Our strategy's final value was <strong>Rs {{ "{:,.2f}".format(report.summary.get('Current Value', 0)) }}</strong> vs the market's <strong>Rs {{ "{:,.2f}".format(report.benchmark_stats.get('Benchmark Final Value', 0)) }}</strong>.<br>We achieved this with significantly lower risk.</p>
          </div>
        </div>
      </div>
      {% endif %}

      <div class="section">
        <h2>Charts</h2>
        <p>These graphs show how your portfolio grew, how stocks and bonds performed separately, and how deep the market fell during crashes.</p>
        {% if report.charts.get('growth') %}
        <div class="chart"><img src="data:image/png;base64,{{ report.charts.growth }}" alt="Portfolio growth chart"></div>
        {% endif %}
        {% if report.charts.get('split') %}
        <div class="chart"><img src="data:image/png;base64,{{ report.charts.split }}" alt="Stocks vs Bonds chart"></div>
        {% endif %}
        {% if report.charts.get('drawdown') %}
        <div class="chart"><img src="data:image/png;base64,{{ report.charts.drawdown }}" alt="Market drawdown chart"></div>
        {% endif %}
      </div>

      <div class="section">
        <h2>Important things to know</h2>
        <ul>
          {% for note in report.risk_notes %}
            <li>{{ note }}</li>
          {% endfor %}
        </ul>
      </div>

      <div class="section">
        <h2>Data used and logic</h2>
        <p>These are the exact ingredients used to create the report.</p>
        <ul>
          {% for item in report.data_used %}
            <li>{{ item | safe }}</li>
          {% endfor %}
        </ul>
      </div>

      {% else %}
      <!-- SIP wealth strategy report -->
      <div class="notice">
        {{ report.story | safe }}
        <p style="margin-top: 12px; font-weight: 700; color: #154360;">
          <span class="ui-icon">{{ ui_icon('calendar') | safe }}</span> Investment Period: <strong>{{ report.start_month_year }}</strong> to <strong>{{ report.end_month_year }}</strong>
        </p>
      </div>
      <div class="grid">
        {% for card in cards %}
          <div class="card">
            <div class="label">{{ card.label }}</div>
            <div class="value">{{ card.value }}</div>
          </div>
        {% endfor %}
      </div>
      <div class="section split">
        <div>
          <h2>How the money moves</h2>
          <p>
            The report uses the first trading day of every month. That is the day the model pretends you make your SIP purchase.
            The same rupee amount goes in each month, but the number of units you receive changes with price.
          </p>
          <ul>
            {% for item in report.how_it_works %}
              <li>{{ item }}</li>
            {% endfor %}
          </ul>
          <div class="chips">
            <div class="chip">Monthly discipline</div>
            <div class="chip">Automatic unit accumulation</div>
            <div class="chip">Compounding over time</div>
            <div class="chip">Long-term growth focus</div>
          </div>
        </div>
        <div>
          <h2>Plain-English result</h2>
          <ul>
            {% for item in report.plain_english %}
              <li>{{ item }}</li>
            {% endfor %}
          </ul>
          <p class="footer-note">Latest market date used: {{ report.latest_date.strftime('%d-%b-%Y') }}</p>
          <p class="footer-note">Data points loaded: {{ report.rows }} trading days; monthly purchase points: {{ report.months }}.</p>
        </div>
      </div>
      <div class="section">
        <h2>What this means for an investor</h2>
        <p>{{ report.investor_view }}</p>
        <ul>
          {% for item in report.potential %}
            <li>{{ item }}</li>
          {% endfor %}
        </ul>
      </div>
      <div class="section">
        <h2>Data used and logic</h2>
        <p>These are the exact ingredients used to create the report.</p>
        <ul>
          {% for item in report.data_used %}
            <li>{{ item | safe }}</li>
          {% endfor %}
        </ul>
      </div>
      <div class="section">
        <h2>Charts</h2>
        <p>These graphs show the market level, the money you put in, the value of the growing portfolio, and the maximum fall from the peak.</p>
        <div class="chart"><img src="data:image/png;base64,{{ report.charts.price }}" alt="NIFTY 50 price chart"></div>
        <div class="chart"><img src="data:image/png;base64,{{ report.charts.wealth }}" alt="Portfolio growth chart"></div>
        <div class="chart"><img src="data:image/png;base64,{{ report.charts.drawdown }}" alt="Drawdown chart"></div>
      </div>
      <div class="section">
        <h2>Sample monthly purchases</h2>
        <p>This table shows how the fixed monthly amount turns into units and portfolio value in the early months of the plan.</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Price</th>
                <th>Investment</th>
                <th>Units Bought</th>
                <th>Total Units</th>
                <th>Portfolio Value</th>
              </tr>
            </thead>
            <tbody>
              {{ table_rows | safe }}
            </tbody>
          </table>
        </div>
      </div>
      {% endif %}
    {% else %}
      <div class="notice">
        {% if selected_project == 'project1' %}
          Select a market segment and click <strong>Run Detailed Report</strong> to generate the full investor-friendly explanation, charts, and results.
        {% else %}
          Click <strong>Run Detailed Report</strong> to analyze the dynamic asset allocation strategy for NIFTY 50 with bonds.
        {% endif %}
      </div>
    {% endif %}
  </div>

  <script>
    // Auto-detect dataset start date when file is selected (using event delegation)
    document.addEventListener("change", async function(e) {
      // Only respond to changes on the dataset_file input
      if (e.target.id !== "dataset_file") return;
      
      const file = e.target.files[0];
      if (!file) return;
      
      console.log("File selected:", file.name);
      
      const formData = new FormData();
      formData.append("file", file);
      
      try {
        const response = await fetch("/parse-dataset-dates", {
          method: "POST",
          body: formData
        });
        
        const data = await response.json();
        console.log("Parse response:", data);
        
        if (data.error) {
          console.warn("Date detection error:", data.error);
        } else if (data.min_date) {
          console.log("Updating start date to:", data.min_date);
          setP2DateFields(data.min_date, data.max_date);
        }
      } catch (err) {
        console.error("Failed to parse dataset dates:", err);
      }
    });
  </script>
</body>
</html>
"""



def parse_p2_form(form, defaults, dataset_date_ranges=None):
    """Build a config dict from form values, falling back to defaults."""
    c = dict(defaults)
    c["excel_file_name"] = form.get("p2_excel_file") or c.get("excel_file_name", "")
    previous_excel_file = form.get("p2_previous_excel_file")
    start_date_source = form.get("p2_start_date_source") or "auto"
    c["start_date"] = form.get("p2_start_date") or None
    c["_start_date_source"] = start_date_source
    c["end_date"] = form.get("p2_end_date") or None
    c["price_column"] = form.get("p2_price_col") or c.get("price_column", "Close")
    c["date_column"] = c.get("date_column", "Date")

    selected_bounds = None
    if dataset_date_ranges:
        selected_bounds = dataset_date_ranges.get(c["excel_file_name"])

    if start_date_source != "user" or (
        previous_excel_file and previous_excel_file != c["excel_file_name"] and start_date_source != "user"
    ):
        apply_auto_start_date_from_bounds(c, selected_bounds)

    try:
        c["monthly_investment"] = float(form.get("p2_invest_amt", c.get("monthly_investment", 1000)))
    except ValueError:
        pass
    c["investment_frequency"] = form.get("p2_freq") or c.get("investment_frequency", "monthly")
    c["investment_day_rule"] = form.get("p2_day_rule") or c.get("investment_day_rule", "first_trading_day")
    try:
        c["bond_annual_return"] = float(form.get("p2_bond_ret", 6.5)) / 100
    except ValueError:
        pass
    c["bond_compounding"] = form.get("p2_bond_comp") or c.get("bond_compounding", "continuous")
    c["trigger_mode"] = form.get("p2_trigger_mode") or c.get("trigger_mode", "highest_only")
    c["output_file_name"] = form.get("p2_output") or c.get("output_file_name", "dynamic_asset_allocation_results.xlsx")
    # Allocation rules
    alloc_count = int(form.get("alloc_count", 0))
    if alloc_count > 0:
        rules = []
        for i in range(alloc_count):
            try:
                rules.append({
                    "min_drawdown": float(form.get(f"alloc_min_{i}", 0)) / 100,
                    "max_drawdown": float(form.get(f"alloc_max_{i}", 100)) / 100,
                    "equity_allocation": float(form.get(f"alloc_eq_{i}", 80)) / 100,
                    "bond_allocation": float(form.get(f"alloc_bd_{i}", 20)) / 100,
                })
            except (ValueError, TypeError):
                pass
        if rules:
            c["allocation_rules"] = sorted(rules, key=lambda r: r["min_drawdown"])
    # Bond shift rules
    bshift_count = int(form.get("bshift_count", 0))
    if bshift_count > 0:
        rules = []
        for i in range(bshift_count):
            try:
                rules.append({
                    "trigger_drawdown": float(form.get(f"bshift_tr_{i}", 10)) / 100,
                    "bond_shift_percentage": float(form.get(f"bshift_pct_{i}", 20)) / 100,
                    "trigger_once_per_ath_cycle": form.get(f"bshift_once_{i}") == "on",
                })
            except (ValueError, TypeError):
                pass
        if rules:
            c["bond_shift_rules"] = sorted(rules, key=lambda r: r["trigger_drawdown"])
    # Metrics
    c["metrics"] = {
        "show_roi": form.get("p2_show_roi") == "on",
        "show_max_drawdown": form.get("p2_show_dd") == "on",
        "show_cagr": form.get("p2_show_cagr") == "on",
        "show_equity_bond_split": form.get("p2_show_split") == "on",
    }
    return c


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    report = None
    cards = []
    table_rows = ""
    selected_segment = DEFAULT_SEGMENT
    selected_project = "project1"
    dataset_date_ranges = build_dataset_date_ranges()

    # Load default P2 config from JSON
    try:
        p2_config = project2_load_config()
    except Exception:
        p2_config = {
            "excel_file_name": "NIFTY 50_Historical_PR_01011990to11102024.csv",
            "start_date": "1999-01-01", "end_date": None,
            "monthly_investment": 1000, "investment_frequency": "monthly",
            "investment_day_rule": "first_trading_day", "price_column": "Close",
            "date_column": "Date", "bond_annual_return": 0.065,
            "bond_compounding": "continuous", "trigger_mode": "highest_only",
            "allocation_rules": [], "bond_shift_rules": [],
            "metrics": {"show_roi": True, "show_max_drawdown": True, "show_cagr": True, "show_equity_bond_split": True},
            "output_file_name": "dynamic_asset_allocation_results.xlsx",
        }
    p2_config.setdefault("_start_date_source", "auto")
    if request.method == "GET":
        apply_auto_start_date_from_bounds(
            p2_config,
            dataset_date_ranges.get(p2_config.get("excel_file_name")),
        )

    if request.method == "POST":
        try:
            selected_project = request.form.get("project", "project1")
            uploaded_file = request.files.get("dataset_file")
            uploaded_df = None

            if uploaded_file is not None and getattr(uploaded_file, "filename", ""):
                uploaded_df = load_uploaded_dataset(uploaded_file)

            if selected_project == "project2":
                p2_config = parse_p2_form(
                    request.form,
                    p2_config,
                    dataset_date_ranges=dataset_date_ranges,
                )
                if uploaded_df is not None and p2_config.get("_start_date_source") != "user":
                    apply_auto_start_date_from_bounds(
                        p2_config,
                        get_dataframe_date_bounds(
                            uploaded_df,
                            date_col=p2_config.get("date_column", DATE_COL),
                        ),
                    )
                report = build_report_project2(p2_config, uploaded_df=uploaded_df)
                cards = [
                    {"label": "Money invested", "value": format_currency(report["summary"]["Total Invested"])},
                    {"label": "Current value", "value": format_currency(report["summary"]["Current Value"])},
                    {"label": "Equity value", "value": format_currency(report["summary"]["Final Equity Value"])},
                    {"label": "Bond value", "value": format_currency(report["summary"]["Final Bond Value"])},
                    {"label": "Profit", "value": format_currency(report["summary"]["Profit"])},
                    {"label": "ROI", "value": f"{report['summary']['ROI %']:,.2f}%"},
                    {"label": "Avg monthly return", "value": f"{report['summary']['Average Monthly Return %']:,.2f}%"},
                    {"label": "Avg yearly return", "value": f"{report['summary']['Average Yearly Return %']:,.2f}%"},
                    {"label": "Max drawdown", "value": f"{report['summary']['Max Drawdown %']:,.2f}%"},
                ]
            else:
                selected_segment = request.form.get("segment", DEFAULT_SEGMENT)
                if selected_segment not in ALL_SEGMENTS:
                    selected_segment = DEFAULT_SEGMENT

                report = build_report(selected_segment, uploaded_df=uploaded_df)
                cards = [
                    {"label": "Money invested", "value": format_currency(report["summary"]["Total Invested"])},
                    {"label": "Current value", "value": format_currency(report["summary"]["Current Value"])},
                    {"label": "Profit", "value": format_currency(report["summary"]["Profit"])},
                    {"label": "ROI", "value": f"{report['summary']['ROI %']:,.2f}%"},
                    {"label": "Avg monthly return", "value": f"{report['summary']['Average Monthly Return %']:,.2f}%"},
                    {"label": "Avg yearly return", "value": f"{report['summary']['Average Yearly Return %']:,.2f}%"},
                    {"label": "Months", "value": f"{int(report['summary']['Total Months Invested'])}"},
                    {"label": "Max drawdown", "value": f"{report['summary']['Max Drawdown %']:,.2f}%"},
                ]
                table_rows = render_table_rows(report["sample_rows"])
        except Exception as exc:
            import traceback
            traceback.print_exc()
            error = f"Could not generate the report: {exc}"

    # Ensure report contains keys the template expects (safe defaults)
    if report:
        report.setdefault("latest_date", report.get("end_date"))
        report.setdefault("rows", report.get("rows", (report.get("daily_df").shape[0] if report.get("daily_df") is not None else 0)))
        report.setdefault("months", report.get("months", report.get("summary", {}).get("Total Months Invested", 0)))
        report.setdefault("investor_view", report.get("investor_view", ""))
        report.setdefault("charts", report.get("charts", {"price": "", "wealth": "", "drawdown": ""}))
        report.setdefault("sample_rows", report.get("sample_rows", []))

    return render_template_string(
        PAGE_TEMPLATE,
        report=report,
        cards=cards,
        table_rows=table_rows,
        selected_segment=selected_segment,
        selected_project=selected_project,
        market_segments=MARKET_SEGMENTS,
        sector_segments=SECTOR_SEGMENTS,
        error=error,
        p2=p2_config,
        dataset_date_ranges=dataset_date_ranges,
        ui_icon=ui_icon,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/parse-dataset-dates", methods=["POST"])
def parse_dataset_dates():
    """
    Parse uploaded dataset file and return the min/max dates.
    Used to auto-populate the start date field.
    """
    try:
        uploaded_file = request.files.get("file")
        if not uploaded_file or not getattr(uploaded_file, "filename", ""):
            return jsonify({"error": "No file provided"}), 400
        
        # Read the file into a DataFrame
        df = load_uploaded_dataset(uploaded_file)
        return jsonify(get_dataframe_date_bounds(df))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def build_report(segment: str = DEFAULT_SEGMENT, uploaded_df: pd.DataFrame | None = None) -> dict:
    segment_file = ALL_SEGMENTS.get(segment, MARKET_SEGMENTS[DEFAULT_SEGMENT])

    if uploaded_df is not None:
        df = load_nifty_data_from_dataframe(uploaded_df)
        segment_file = "Uploaded dataset"
        segment = "Uploaded dataset"
    else:
        data_file = DATA_DIR / segment_file
        df = load_nifty_data(data_file)

    first_row = df.iloc[0]
    latest_row = df.iloc[-1]
    start_date = first_row[DATE_COL]
    latest_date = latest_row[DATE_COL]
    latest_price = latest_row[PRICE_COL]

    start_month_year = start_date.strftime("%B %Y")
    end_month_year = latest_date.strftime("%B %Y")

    monthly_df = get_monthly_first_trading_day_data(df)
    sip_df = calculate_sip(monthly_df, latest_price, latest_date)
    daily_sip_df = calculate_daily_sip_portfolio(df, sip_df)
    summary = calculate_summary(sip_df, daily_sip_df=daily_sip_df)
    output_message = save_outputs(sip_df, summary)
    charts = build_charts(df, sip_df, daily_sip_df=daily_sip_df)

    sample_rows = sip_df[[DATE_COL, PRICE_COL, "Investment", "Units_Bought", "Total_Units", "Portfolio_Value"]].head(8).copy()
    sample_rows[DATE_COL] = sample_rows[DATE_COL].dt.strftime("%d-%b-%Y")

    return {
        "summary": summary,
        "segment": segment,
        "story": (
            f"This report shows what would have happened if you invested {format_currency(MONTHLY_INVESTMENT)} every month on the first trading day in <strong>{segment}</strong>, "
            f"from {start_month_year} to {end_month_year}. "
            f"Over the full {summary['Total Months Invested']:.0f}-month period, you put in {format_currency(summary['Total Invested'])} and the investment grew to {format_currency(summary['Current Value'])}."
        ),
        "how_it_works": [
            "Each month, the same amount of money is added.",
            "When NIFTY is expensive, that money buys fewer units.",
            "When NIFTY is cheaper, the same money buys more units.",
            "That steady buying is what people call SIP: Systematic Investment Plan.",
        ],
        "data_used": [
            f"Market Segment: <strong>{segment}</strong>",
            f"Data coverage: {start_month_year} to {end_month_year} ({summary['Total Months Invested']:.0f} months).",
            f"Source file: {segment_file}",
            "Columns in the file: Index Name, Date, Open, High, Low, Close.",
            "This report uses the Date and Close columns to measure the market and the monthly purchase price.",
            "Only the first trading day of every month is used so each month counts once.",
        ],
        "plain_english": [
            f"Investment period: {start_month_year} to {end_month_year} ({summary['Total Months Invested']:.0f} months).",
            f"Money invested slowly over time: {format_currency(summary['Total Invested'])}.",
            f"Current value of the same money today: {format_currency(summary['Current Value'])}.",
            f"Extra wealth created by staying invested: {format_currency(summary['Profit'])}.",
            f"Every {format_currency(1)} became about {(summary['Current Value'] / summary['Total Invested']):.2f} times larger.",
            f"Average monthly return: {summary['Average Monthly Return %']:.2f}%.",
            f"Average yearly return: {summary['Average Yearly Return %']:.2f}%.",
        ],
        "potential": [
            "Long-term investing works because the stock market has historically rewarded patience, discipline, and time.",
            "A fixed monthly plan removes emotion: you do not need to guess the perfect day to invest.",
            "The upside is compounding, which means the money you already invested can also start earning returns.",
            "The risk is that markets can fall hard for long periods, so past performance is not a promise of future results.",
        ],
        "investor_view": (
            "If you are explaining this to a non-technical investor: think of it like planting a small tree every month. "
            "Some months the seed is cheap, some months it is expensive, but over time the garden grows. "
            "The important part is consistency, not timing perfection."
        ),
        "sample_rows": sample_rows.to_dict(orient="records"),
        "charts": charts,
        "output_message": output_message,
        "start_date": start_date,
        "latest_date": latest_date,
        "start_month_year": start_month_year,
        "end_month_year": end_month_year,
        "rows": len(df),
        "months": len(monthly_df),
    }


def build_report_project2(config: dict, uploaded_df: pd.DataFrame | None = None) -> dict:
    """
    Build the dynamic asset allocation strategy report using user-provided config.
    Generates investor-friendly narrative that a layman can understand.
    Includes benchmark comparison showing drawdown protection benefits.
    """
    if uploaded_df is not None:
      df = project2_load_data_from_dataframe(uploaded_df, config)
    else:
      df = project2_load_data(config)
    df = mark_investment_days(df, config)

    daily_df, actions_df = project2_simulate(df, config)
    
    # Add benchmark portfolio
    daily_df = simulate_nifty_benchmark(daily_df, config)
    # Add comparison columns
    daily_df = calculate_comparison_columns(daily_df)
    
    summary = project2_summary(daily_df, config)
    p2_charts = build_charts_project2(daily_df, config)

    start_date = daily_df.iloc[0]["Date"]
    end_date = daily_df.iloc[-1]["Date"]
    start_month_year = start_date.strftime("%B %Y")
    end_month_year = end_date.strftime("%B %Y")

    # Determine segment name from file or uploaded dataset
    segment_file = config.get("excel_file_name", "NIFTY 50_Historical_PR_01011990to11102024.csv")
    segment_name = "NIFTY 50"
    for name, file in MARKET_SEGMENTS.items():
      if file == segment_file:
        segment_name = name
        break
    if uploaded_df is not None:
      segment_file = "Uploaded dataset"
      segment_name = "Uploaded dataset"

    # Key metrics
    final_equity = summary["Final Equity Value"]
    final_bond = summary["Final Bond Value"]
    final_total = summary["Final Total Asset Value"]
    total_invested = summary["Total Invested"]
    roi = summary["ROI %"]
    max_dd = summary["Max Portfolio Drawdown %"]
    invest_amt = config.get("monthly_investment", 1000)
    bond_ret = config.get("bond_annual_return", 0.065) * 100

    # Accurate XIRR Calculation using actual investment flows
    investments = daily_df["Total Invested"].diff().fillna(daily_df["Total Invested"].iloc[0])
    investments = investments[investments > 0]
    investment_periods = max(1, len(investments))
    n_months = investment_periods
    years_elapsed = (end_date - start_date).days / 365.25
    n_years = round(years_elapsed, 1) if years_elapsed > 0 else 0
    frequency = config.get("investment_frequency", "monthly")
    period_singular = {
        "monthly": "month",
        "weekly": "week",
        "daily": "trading day",
    }.get(frequency, "period")
    period_plural = {
        "monthly": "months",
        "weekly": "weeks",
        "daily": "trading days",
    }.get(frequency, "periods")
    frequency_intro = {
        "monthly": "Every month",
        "weekly": "Every week",
        "daily": "Every trading day",
    }.get(frequency, "Every period")
    recurring_label = {
        "monthly": "monthly",
        "weekly": "weekly",
        "daily": "daily",
    }.get(frequency, "scheduled")
    
    dates = daily_df.loc[investments.index, 'Date'].tolist()
    amounts = [-amt for amt in investments.tolist()]
    
    dates.append(daily_df['Date'].iloc[-1])
    amounts.append(final_total)
    
    avg_yearly_return = calculate_sip_xirr(dates, amounts)
    avg_monthly_return = ((1 + avg_yearly_return / 100) ** (1 / 12) - 1) * 100 if avg_yearly_return > -100 else 0

    multiplier = round(final_total / total_invested, 2)
    profit = final_total - total_invested
    equity_pct = round(final_equity / final_total * 100, 1) if final_total else 0
    bond_pct = round(final_bond / final_total * 100, 1) if final_total else 0

    # Build allocation rules narrative
    alloc_rules = config.get("allocation_rules", [])
    alloc_narrative = []
    for r in alloc_rules:
        mn = round(r["min_drawdown"] * 100)
        mx = round(r["max_drawdown"] * 100)
        eq = round(r["equity_allocation"] * 100)
        bd = round(r["bond_allocation"] * 100)
        if mn == 0:
            alloc_narrative.append(f"When market is near its peak (0-{mx}% fall): {eq}% goes to stocks, {bd}% to bonds")
        elif mx >= 100:
            alloc_narrative.append(f"When market crashes {mn}%+ from peak: {eq}% goes to stocks, {bd}% to bonds")
        else:
            alloc_narrative.append(f"When market falls {mn}-{mx}% from peak: {eq}% goes to stocks, {bd}% to bonds")

    # Bond shift narrative
    bshift_rules = config.get("bond_shift_rules", [])
    shift_narrative = []
    for r in bshift_rules:
        tr = round(r["trigger_drawdown"] * 100)
        pct = round(r["bond_shift_percentage"] * 100)
        shift_narrative.append(f"If market drops {tr}% from peak → move {pct}% of bond savings into stocks (buy the dip)")

    report_dict = {
        "summary": {
            "Total Invested": total_invested,
            "Current Value": final_total,
            "Final Equity Value": final_equity,
            "Final Bond Value": final_bond,
            "Profit": profit,
            "ROI %": roi,
            "Average Monthly Return %": avg_monthly_return,
            "Average Yearly Return %": avg_yearly_return,
            "Max Drawdown %": max_dd,
            "Total Months Invested": n_months,
        },
        "daily_df": daily_df,
        "actions_df": actions_df,
        "charts": p2_charts,
        # ---- INVESTOR-PITCH NARRATIVE ----
        "story": (
          f"<strong>Imagine you saved just {format_currency(invest_amt)} {frequency_intro.lower()}</strong> - "
          f"like a small recurring deposit - but instead of letting it sit idle, "
          f"a smart system split it between Indian stocks ({segment_name}) and safe bonds. "
          f"Over <strong>{n_years} years</strong> ({start_month_year} to {end_month_year}), "
          f"your total savings of <strong>{format_currency(total_invested)}</strong> "
          f"would have grown to <strong>{format_currency(final_total)}</strong>. "
          f"That's <strong>{format_currency(profit)}</strong> in pure profit - "
          f"your money multiplied <strong>{multiplier}x</strong>."
        ),
        "headline_profit": format_currency(profit),
        "headline_invested": format_currency(total_invested),
        "headline_value": format_currency(final_total),
        "headline_multiplier": f"{multiplier}x",
        "headline_years": f"{n_years}",
        "headline_monthly": format_currency(invest_amt),
        "equity_pct": equity_pct,
        "bond_pct": bond_pct,
        "how_it_works": [
          f"<strong>Step 1 - You save a fixed amount:</strong> {frequency_intro}, {format_currency(invest_amt)} leaves your bank account automatically. Think of it as paying yourself first, like an EMI - except this EMI builds YOUR wealth.",
          f"<strong>Step 2 - The system splits your money smartly:</strong> When the stock market is doing well (near its peak), most of your money goes into stocks for growth, and a small part goes into safe bonds (earning {bond_ret:.1f}% per year) as a safety net.",
          f"<strong>Step 3 - When markets fall, the system gets greedy:</strong> If the market drops 10%, 20%, or 30% from its peak, the system automatically puts MORE money into stocks (because stocks are 'on sale'). It even moves money FROM bonds INTO stocks - buying low is the secret to wealth building.",
          f"<strong>Step 4 - When market recovers to a new high:</strong> Everything resets. The system goes back to normal mode and waits for the next opportunity. This cycle of 'buy more when cheap, hold steady when expensive' has repeated many times over {n_years} years.",
        ],
        "kitchen_table": (
            f"Think of your money like a garden with two types of plants. The stock market plants ({segment_name}) "
            f"can grow very tall - but sometimes storms knock them down. The bond plants are shorter but "
            f"almost never break - they grow slowly and steadily at {bond_ret:.1f}% every year. "
            f"Our system is like a smart gardener: when a storm hits (market falls), the gardener "
            f"takes seeds from the safe plants and plants them where the tall plants fell - because "
            f"that's where the biggest growth happens when the sun comes back. "
            f"Over {n_years} years, this smart gardening turned {format_currency(total_invested)} "
            f"into {format_currency(final_total)}."
        ),
        "alloc_narrative": alloc_narrative,
        "shift_narrative": shift_narrative,
        "data_used": [
            f"Strategy: Smart Asset Allocation (Quant-Based)",
            f"Data coverage: {start_month_year} to {end_month_year} ({investment_periods} {period_plural} / {n_years} years)",
            f"Market data: {segment_name} historical prices ({segment_file})",
            f"Bond return: {bond_ret:.1f}% annual ({config.get('bond_compounding', 'continuous')} compounding)",
            f"Monthly investment: {format_currency(invest_amt)}",
            f"Rebalancing: Automatic, based on how far market has fallen from its all-time high",
        ],
        "plain_english": [
            f"You put in {format_currency(invest_amt)} {frequency_intro.lower()} for {n_years} years.",
            f"Total money out of your pocket: {format_currency(total_invested)}.",
            f"What it became: {format_currency(final_total)} - that's {multiplier}x your money.",
            f"Of this, {format_currency(final_equity)} ({equity_pct}%) is in stocks and {format_currency(final_bond)} ({bond_pct}%) is in bonds.",
            f"Pure profit (money you DIDN'T put in): {format_currency(profit)}.",
            f"Average return: {avg_yearly_return:.1f}% per year ({avg_monthly_return:.2f}% per month).",
        ],
        "potential": [
            f"<strong>Wealth multiplication:</strong> Every {format_currency(1)} you invested became {format_currency(multiplier)}. A {format_currency(10000)}/month plan would have created {format_currency(profit * 10)} in profit.",
            f"<strong>Beats inflation:</strong> At {avg_yearly_return:.1f}% average yearly returns, this strategy significantly outperformed bank FDs (6-7%) and inflation (5-6%).",
            f"<strong>Downside protection:</strong> The worst your portfolio ever fell was {abs(max_dd):.1f}% - and it recovered. Bonds acted as a safety cushion during crashes.",
            f"<strong>No expertise needed:</strong> The system runs on simple rules - you don't need to watch the market, read charts, or make decisions. Just invest on schedule and the algorithm does the rest.",
            f"<strong>Discipline over timing:</strong> You don't need to guess 'is it the right time to invest?' - the system handles timing automatically by buying more when prices are low.",
        ],
        "risk_notes": [
            "Past performance does not guarantee future results. Markets can behave differently.",
            "This is a backtest (simulation using old data), not a live trading result.",
            f"The worst temporary loss was {abs(max_dd):.1f}%. You need the patience to stay invested during drops.",
            "Returns shown are before taxes and transaction costs.",
        ],
        "start_date": start_date,
        "end_date": end_date,
        "latest_date": end_date,
        "rows": len(daily_df),
        "months": n_months,
        "investor_view": (
            f"If someone gave you a magic box that turned {format_currency(total_invested)} into "
            f"{format_currency(final_total)} over {n_years} years — would you use it? "
            f"This strategy IS that box. It uses a simple, rule-based approach: save {recurring_label}, "
            f"split between stocks and bonds, and automatically buy more stocks when they're cheap. "
            f"No guesswork, no trading, no stress."
        ),
        "start_month_year": start_month_year,
        "end_month_year": end_month_year,
        "segment_name": segment_name,
        "frequency_intro": frequency_intro,
        "investment_count_label": f"{period_plural} invested",
    }
    
    # Add benchmark comparison metrics if available
    if "Benchmark NIFTY Value" in daily_df.columns:
        benchmark_final_value = daily_df.iloc[-1]["Benchmark NIFTY Value"]
        benchmark_max_dd = daily_df.iloc[-1].get("Benchmark Max Drawdown %", daily_df["Benchmark Drawdown %"].min())
        strategy_max_dd_abs = abs(max_dd)
        benchmark_max_dd_abs = abs(benchmark_final_value) if benchmark_final_value < 0 else abs(daily_df["Benchmark Drawdown %"].min())
        
        # Get benchmark final ROI
        benchmark_total_invested = daily_df.iloc[-1]["Benchmark Total Invested"]
        benchmark_roi = ((benchmark_final_value - benchmark_total_invested) / benchmark_total_invested * 100) if benchmark_total_invested > 0 else 0
        
        # Calculate max DD correctly for benchmark
        benchmark_max_dd = daily_df["Benchmark Drawdown %"].min() if "Benchmark Drawdown %" in daily_df.columns else 0
        
        # Drawdown benefit
        dd_benefit = abs(benchmark_max_dd) - abs(max_dd)
        
        # Value difference
        value_diff = final_total - benchmark_final_value
        value_diff_pct = (value_diff / benchmark_final_value * 100) if benchmark_final_value > 0 else 0
        
        # Days with protection
        protected_days = (daily_df["Strategy Portfolio Drawdown %"] > daily_df["Benchmark Portfolio Drawdown %"]).sum()
        pct_protected = (protected_days / len(daily_df) * 100) if len(daily_df) > 0 else 0
        
        bench_summary = calculate_drawdown_benefit_summary(daily_df)
        
        # Add to return dict
        ret = {
            "summary": {
                "Total Invested": total_invested,
                "Current Value": final_total,
                "Final Equity Value": final_equity,
                "Final Bond Value": final_bond,
                "Profit": profit,
                "ROI %": roi,
                "Average Monthly Return %": avg_monthly_return,
                "Average Yearly Return %": avg_yearly_return,
                "Max Drawdown %": max_dd,
                "Total Months Invested": n_months,
                "Benchmark Final Value": benchmark_final_value,
                "Benchmark ROI %": benchmark_roi,
                "Benchmark Max Drawdown %": benchmark_max_dd,
                "Drawdown Protection %": dd_benefit,
                "Value Advantage vs Benchmark": value_diff,
                "Days with Better Drawdown Protection": f"{protected_days} ({pct_protected:.1f}%)",
            },
            "daily_df": daily_df,
            "actions_df": actions_df,
            "charts": p2_charts,
            # ---- INVESTOR-PITCH NARRATIVE ----
            "story": (
              f"<strong>Imagine you saved just {format_currency(invest_amt)} {frequency_intro.lower()}</strong> - "
              f"like a small recurring deposit - but instead of letting it sit idle, "
              f"a smart system split it between Indian stocks ({segment_name}) and safe bonds. "
              f"Over <strong>{n_years} years</strong> ({start_month_year} to {end_month_year}), "
              f"your total savings of <strong>{format_currency(total_invested)}</strong> "
              f"would have grown to <strong>{format_currency(final_total)}</strong>. "
              f"That's <strong>{format_currency(profit)}</strong> in pure profit - "
              f"your money multiplied <strong>{multiplier}x</strong>. "
              f"<br/><br/><strong>But here's the kicker:</strong> If you had put ALL {format_currency(invest_amt)} into stocks {frequency_intro.lower()} (no bonds), "
              f"you would have had {format_currency(benchmark_final_value)} - but you would've suffered a {abs(benchmark_max_dd):.1f}% drawdown at the worst. "
              f"Our strategy? Only {abs(max_dd):.1f}% drawdown. That's {dd_benefit:.1f} percentage points of protection!"
            ),
            "headline_profit": format_currency(profit),
            "headline_invested": format_currency(total_invested),
            "headline_value": format_currency(final_total),
            "headline_multiplier": f"{multiplier}x",
            "headline_years": f"{n_years}",
            "headline_monthly": format_currency(invest_amt),
            "equity_pct": equity_pct,
            "bond_pct": bond_pct,
            "how_it_works": [
              f"<strong>Step 1 - You save a fixed amount:</strong> {frequency_intro}, {format_currency(invest_amt)} leaves your bank account automatically. Think of it as paying yourself first, like an EMI - except this EMI builds YOUR wealth.",
              f"<strong>Step 2 - The system splits your money smartly:</strong> When the stock market is doing well (near its peak), most of your money goes into stocks for growth, and a small part goes into safe bonds (earning {bond_ret:.1f}% per year) as a safety net.",
              f"<strong>Step 3 - When markets fall, the system gets greedy:</strong> If the market drops 10%, 20%, or 30% from its peak, the system automatically puts MORE money into stocks (because stocks are 'on sale'). It even moves money FROM bonds INTO stocks - buying low is the secret to wealth building.",
              f"<strong>Step 4 - When market recovers to a new high:</strong> Everything resets. The system goes back to normal mode and waits for the next opportunity. This cycle of 'buy more when cheap, hold steady when expensive' has repeated many times over {n_years} years.",
            ],
            "kitchen_table": (
              f"Think of your money like a garden with two types of plants. The stock market plants ({segment_name}) "
              f"can grow very tall - but sometimes storms knock them down. The bond plants are shorter but "
              f"almost never break - they grow slowly and steadily at {bond_ret:.1f}% every year. "
              f"Our system is like a smart gardener: when a storm hits (market falls), the gardener "
              f"takes seeds from the safe plants and plants them where the tall plants fell - because "
              f"that's where the biggest growth happens when the sun comes back. "
              f"Over {n_years} years, this smart gardening turned {format_currency(total_invested)} "
              f"into {format_currency(final_total)}. A purely stock-focused gardener would have had {format_currency(benchmark_final_value)}, "
              f"but suffered twice as much during storms."
            ),
            "alloc_narrative": alloc_narrative,
            "shift_narrative": shift_narrative,
            "data_used": [
                f"Strategy: Smart Asset Allocation (Quant-Based)",
                f"Data coverage: {start_month_year} to {end_month_year} ({investment_periods} {period_plural} / {n_years} years)",
                f"Market data: {segment_name} historical prices ({segment_file})",
                f"Bond return: {bond_ret:.1f}% annual ({config.get('bond_compounding', 'continuous')} compounding)",
                f"Monthly investment: {format_currency(invest_amt)}",
                f"Rebalancing: Automatic, based on how far market has fallen from its all-time high",
            ],
            "plain_english": [
                f"You put in {format_currency(invest_amt)} {frequency_intro.lower()} for {n_years} years.",
                f"Total money out of your pocket: {format_currency(total_invested)}.",
                f"What it became: {format_currency(final_total)} - that's {multiplier}x your money.",
                f"Of this, {format_currency(final_equity)} ({equity_pct}%) is in stocks and {format_currency(final_bond)} ({bond_pct}%) is in bonds.",
                f"Pure profit (money you DIDN'T put in): {format_currency(profit)}.",
                f"Average return: {avg_yearly_return:.1f}% per year ({avg_monthly_return:.2f}% per month).",
                f"Vs a 100% stock benchmark, our strategy had lower drawdown on {protected_days} out of {len(daily_df)} days ({pct_protected:.1f}%). Max protection: {dd_benefit:.1f}%.",
            ],
            "benchmark_stats": bench_summary,
            "potential": [
              f"<strong>Wealth multiplication:</strong> Every {format_currency(1)} you invested became {format_currency(multiplier)}. A {format_currency(10000)}/month plan would have created {format_currency(profit * 10)} in profit.",
              f"<strong>Beats inflation:</strong> At {avg_yearly_return:.1f}% average yearly returns, this strategy significantly outperformed bank FDs (6-7%) and inflation (5-6%).",
              f"<strong>Downside protection:</strong> The worst your portfolio ever fell was {abs(max_dd):.1f}% - but if you invested in 100% stocks, you would've fallen {abs(benchmark_max_dd):.1f}%. That's {dd_benefit:.1f} percentage points of protection from bonds.",
              f"<strong>Smarter than stocks alone:</strong> Over {n_years} years, stocks alone would give you {format_currency(benchmark_final_value)} with {abs(benchmark_max_dd):.1f}% max pain. Our strategy gives you {format_currency(final_total)} with only {abs(max_dd):.1f}% max pain - better returns AND better sleep.",
              f"<strong>No expertise needed:</strong> The system runs on simple rules - you don't need to watch the market, read charts, or make decisions. Just invest on schedule and the algorithm does the rest.",
              f"<strong>Discipline over timing:</strong> You don't need to guess 'is it the right time to invest?' - the system handles timing automatically by buying more when prices are low.",
            ],
            "risk_notes": [
                "Past performance does not guarantee future results. Markets can behave differently.",
                "This is a backtest (simulation using old data), not a live trading result.",
                f"The worst temporary loss was {abs(max_dd):.1f}%. You need the patience to stay invested during drops.",
                "Returns shown are before taxes and transaction costs.",
                f"Benchmark comparison (100% stocks) is for illustration only. Different investments, different results.",
            ],
            "start_date": start_date,
            "end_date": end_date,
            "latest_date": end_date,
            "rows": len(daily_df),
            "months": n_months,
            "investor_view": (
                f"If someone gave you a magic box that turned {format_currency(total_invested)} into "
                f"{format_currency(final_total)} over {n_years} years - would you use it? "
                f"This strategy IS that box. It uses a simple, rule-based approach: save {recurring_label}, "
                f"split between stocks and bonds, and automatically buy more stocks when they're cheap. "
                f"No guesswork, no trading, no stress. And the best part? During market crashes, bonds protected you. "
                f"While pure stock investors panicked at {abs(benchmark_max_dd):.1f}% losses, you only experienced {abs(max_dd):.1f}%.",
            ),
            "start_month_year": start_month_year,
            "end_month_year": end_month_year,
            "segment_name": segment_name,
            "frequency_intro": frequency_intro,
            "investment_count_label": f"{period_plural} invested",
        }
        return ret
    
    # Return without benchmark (fallback)
    return {
        "summary": {
            "Total Invested": total_invested,
            "Current Value": final_total,
            "Final Equity Value": final_equity,
            "Final Bond Value": final_bond,
            "Profit": profit,
            "ROI %": roi,
            "Average Monthly Return %": avg_monthly_return,
            "Average Yearly Return %": avg_yearly_return,
            "Max Drawdown %": max_dd,
            "Total Months Invested": n_months,
        },
        "daily_df": daily_df,
        "actions_df": actions_df,
        "charts": p2_charts,
        "story": (
            f"<strong>Imagine you saved just {format_currency(invest_amt)} {frequency_intro.lower()}</strong> — "
            f"like a small recurring deposit — but instead of letting it sit idle, "
            f"a smart system split it between Indian stocks ({segment_name}) and safe bonds. "
            f"Over <strong>{n_years} years</strong> ({start_month_year} to {end_month_year}), "
            f"your total savings of <strong>{format_currency(total_invested)}</strong> "
            f"would have grown to <strong>{format_currency(final_total)}</strong>. "
            f"That's <strong>{format_currency(profit)}</strong> in pure profit — "
            f"your money multiplied <strong>{multiplier}x</strong>."
        ),
        "headline_profit": format_currency(profit),
        "headline_invested": format_currency(total_invested),
        "headline_value": format_currency(final_total),
        "headline_multiplier": f"{multiplier}x",
        "headline_years": f"{n_years}",
        "headline_monthly": format_currency(invest_amt),
        "equity_pct": equity_pct,
        "bond_pct": bond_pct,
        "how_it_works": [
            f"<strong>Step 1 — You save a fixed amount:</strong> {frequency_intro}, {format_currency(invest_amt)} leaves your bank account automatically. Think of it as paying yourself first, like an EMI — except this EMI builds YOUR wealth.",
            f"<strong>Step 2 — The system splits your money smartly:</strong> When the stock market is doing well (near its peak), most of your money goes into stocks for growth, and a small part goes into safe bonds (earning {bond_ret:.1f}% per year) as a safety net.",
            f"<strong>Step 3 — When markets fall, the system gets greedy:</strong> If the market drops 10%, 20%, or 30% from its peak, the system automatically puts MORE money into stocks (because stocks are 'on sale'). It even moves money FROM bonds INTO stocks — buying low is the secret to wealth building.",
            f"<strong>Step 4 — When market recovers to a new high:</strong> Everything resets. The system goes back to normal mode and waits for the next opportunity. This cycle of 'buy more when cheap, hold steady when expensive' has repeated many times over {n_years} years.",
        ],
        "kitchen_table": (
            f"Think of your money like a garden with two types of plants. The stock market plants ({segment_name}) "
            f"can grow very tall — but sometimes storms knock them down. The bond plants are shorter but "
            f"almost never break — they grow slowly and steadily at {bond_ret:.1f}% every year. "
            f"Our system is like a smart gardener: when a storm hits (market falls), the gardener "
            f"takes seeds from the safe plants and plants them where the tall plants fell — because "
            f"that's where the biggest growth happens when the sun comes back. "
            f"Over {n_years} years, this smart gardening turned {format_currency(total_invested)} "
            f"into {format_currency(final_total)}."
        ),
        "alloc_narrative": alloc_narrative,
        "shift_narrative": shift_narrative,
        "data_used": [
            f"Strategy: Smart Asset Allocation (Quant-Based)",
            f"Data coverage: {start_month_year} to {end_month_year} ({investment_periods} {period_plural} / {n_years} years)",
            f"Market data: {segment_name} historical prices ({segment_file})",
            f"Bond return: {bond_ret:.1f}% annual ({config.get('bond_compounding', 'continuous')} compounding)",
            f"Monthly investment: {format_currency(invest_amt)}",
            f"Rebalancing: Automatic, based on how far market has fallen from its all-time high",
        ],
        "plain_english": [
            f"You put in {format_currency(invest_amt)} {frequency_intro.lower()} for {n_years} years.",
            f"Total money out of your pocket: {format_currency(total_invested)}.",
            f"What it became: {format_currency(final_total)} — that's {multiplier}x your money.",
            f"Of this, {format_currency(final_equity)} ({equity_pct}%) is in stocks and {format_currency(final_bond)} ({bond_pct}%) is in bonds.",
            f"Pure profit (money you DIDN'T put in): {format_currency(profit)}.",
            f"Average return: {avg_yearly_return:.1f}% per year ({avg_monthly_return:.2f}% per month).",
        ],
        "potential": [
            f"<strong>Wealth multiplication:</strong> Every {format_currency(1)} you invested became {format_currency(multiplier)}. A {format_currency(10000)}/month plan would have created {format_currency(profit * 10)} in profit.",
            f"<strong>Beats inflation:</strong> At {avg_yearly_return:.1f}% average yearly returns, this strategy significantly outperformed bank FDs (6-7%) and inflation (5-6%).",
            f"<strong>Downside protection:</strong> The worst your portfolio ever fell was {abs(max_dd):.1f}% — and it recovered. Bonds acted as a safety cushion during crashes.",
            f"<strong>No expertise needed:</strong> The system runs on simple rules — you don't need to watch the market, read charts, or make decisions. Just invest on schedule and the algorithm does the rest.",
            f"<strong>Discipline over timing:</strong> You don't need to guess 'is it the right time to invest?' — the system handles timing automatically by buying more when prices are low.",
        ],
        "risk_notes": [
            "Past performance does not guarantee future results. Markets can behave differently.",
            "This is a backtest (simulation using old data), not a live trading result.",
            f"The worst temporary loss was {abs(max_dd):.1f}%. You need the patience to stay invested during drops.",
            "Returns shown are before taxes and transaction costs.",
        ],
        "start_date": start_date,
        "end_date": end_date,
        "latest_date": end_date,
        "rows": len(daily_df),
        "months": n_months,
        "investor_view": (
            f"If someone gave you a magic box that turned {format_currency(total_invested)} into "
            f"{format_currency(final_total)} over {n_years} years — would you use it? "
            f"This strategy IS that box. It uses a simple, rule-based approach: save {recurring_label}, "
            f"split between stocks and bonds, and automatically buy more stocks when they're cheap. "
            f"No guesswork, no trading, no stress."
        ),
        "start_month_year": start_month_year,
        "end_month_year": end_month_year,
        "segment_name": segment_name,
        "frequency_intro": frequency_intro,
        "investment_count_label": f"{period_plural} invested",
    }


# =========================
# MAIN FUNCTION
# =========================

def main():
    print("Loading NIFTY data...")

    df = load_nifty_data(DATA_FILE)

    print("\nData loaded successfully.")
    print(f"Start Date: {df[DATE_COL].min().date()}")
    print(f"End Date: {df[DATE_COL].max().date()}")
    print(f"Total rows: {len(df)}")

    latest_row = df.iloc[-1]
    latest_date = latest_row[DATE_COL]
    latest_price = latest_row[PRICE_COL]

    print("\nCreating monthly first trading day data...")

    monthly_df = get_monthly_first_trading_day_data(df)

    print(f"Total months found: {len(monthly_df)}")

    print("\nCalculating SIP...")

    sip_df = calculate_sip(monthly_df, latest_price, latest_date)
    daily_sip_df = calculate_daily_sip_portfolio(df, sip_df)

    summary = calculate_summary(sip_df, daily_sip_df=daily_sip_df)

    print("\n========== FINAL SUMMARY ==========")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value}")

    save_outputs(sip_df, summary)


if __name__ == "__main__":
    print("NIFTY Strategy Suite web report is starting...")
    print("Open http://127.0.0.1:5000 in your browser and click Run Detailed Report.")
    app.run(host="127.0.0.1", port=5000, debug=False)
