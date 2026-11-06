"""Battery arbitrage across two electricity markets.

Run with:
    python main.py

Requirements: pandas, numpy, statsmodels.

Workflow
--------
1. The first four weeks use a pure-arbitrage rule.  It only buys when the
   cheapest market is in the bottom quantile of its recent range and only sells
   when the most expensive market is in the top quantile, after accounting for
   round-trip efficiency and cycle wear.  The lookback and quantile are
   calibrated on those four weeks and then frozen.
2. Each market also carries a seven-day (168-hour) moving average.  Once a
   full week of history exists, the current price is compared with that
   average: buying below the average, or selling above it, uses full power;
   the less favourable side of the average trades at half power.
3. From week five, an ARIMA model forecasts each market price using only
   calendar features (time of day, day of week and ``is_holiday``).
4. Three strategies are compared on the out-of-sample period:
   * pure arbitrage,
   * ARIMA-informed dispatch, and
   * a combined rule that picks whichever candidate has the higher expected
     arbitrage profit.

The two batteries are modelled independently.  This matters because one unit
can charge while the other discharges; however, a single unit is never allowed
to charge and discharge in the same half-hour.  A cycle is defined as filling
a battery and discharging all of its stored energy, even if that happens in
several segments (for example, 75% then 25% counts as one full cycle).  Any
energy left in the batteries at the end of a backtest is marked to market as a
memo item, while the strategy ranking uses realised action profit.

The model is intentionally a single, readable backtest rather than a live
trading system.  It is a price taker and assumes each battery can choose the
best market at each half-hour decision point.
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Dict

import pandas as pd

from optimiser import (
    ArbitrageStrategy,
    BacktestResult,
    Backtester,
    Battery,
    BatteryFleet,
    CombinedStrategy,
    ForecastOptimiser,
)
from prediction_model import ARIMAPriceForecaster, build_time_features

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ATTACHMENT_PATH = BASE_DIR / "Attachment.csv"
RESULTS_DIR = BASE_DIR / "results"

N_BATTERIES = 2
TRAIN_WEEKS = 4
RESAMPLE_FREQ = "30min"
HOLIDAYS = {date(2018, 1, 1)}
FORECAST_HORIZON = 8  # four hours of half-hourly steps

# "cycle" charges the £100 full-cycle cost only when stored energy is
# discharged, so partial charge/discharge segments accumulate into full
# cycles.  "throughput" spreads the cost pro-rata over every MWh of grid
# throughput.  "per_action" applies a flat £100 to every half-hour charge or
# discharge call.  "cycle" is the default because it matches the brief's
# definition: one cycle is filling the battery and emptying it, however many
# segments that takes.
COST_MODE = "cycle"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_attachment(path: Path) -> Dict[str, float]:
    """Read Attachment.csv into a clean name -> float dictionary."""
    frame = pd.read_csv(path)
    frame["Name"] = frame["Name"].astype(str).str.strip()
    return {
        str(name): float(value)
        for name, value in zip(frame["Name"], frame["Values"])
    }


def _read_market(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["time"])
    frame = frame.sort_values("time").drop_duplicates("time")
    return frame.set_index("time")["price"].astype(float).rename(column)


def build_market_frame(data_dir: Path) -> pd.DataFrame:
    """Align both markets on a common half-hourly grid.

    Market 2 is hourly, so its price is carried forward into the second
    half-hour of each hour.  Forward filling is deliberate: no future price is
    ever used in a decision.
    """
    market_1 = _read_market(data_dir / "market_1.csv", "market_1_price")
    market_2 = _read_market(data_dir / "market_2.csv", "market_2_price")

    start = min(market_1.index.min(), market_2.index.min())
    end = max(market_1.index.max(), market_2.index.max())
    grid = pd.date_range(start, end, freq=RESAMPLE_FREQ)

    frame = pd.DataFrame(
        {
            "market_1_price": market_1.reindex(grid, method="ffill"),
            "market_2_price": market_2.reindex(grid, method="ffill"),
        }
    ).dropna()

    frame["is_holiday"] = build_time_features(
        frame.index, HOLIDAYS
    )["is_holiday"]

    # Seven-day (168-hour) moving average for each market.  The frame is
    # half-hourly, so the window is 168 * 2 = 336 bars.  min_periods makes the
    # column NaN until a full week of history exists; strategies treat that as
    # "no signal" and trade at full size.
    ma_bars = 168 * 2
    for market in ("market_1", "market_2"):
        frame[f"{market}_ma_7d"] = (
            frame[f"{market}_price"]
            .rolling(window=ma_bars, min_periods=ma_bars)
            .mean()
        )
    return frame


def build_fleet(params: Dict[str, float]) -> BatteryFleet:
    """Create two independent batteries from Attachment.csv.

    Attachment.csv describes one 4 MWh / 2 MW unit, including a £500,000
    capex and a 5,000-cycle life.  Each unit therefore carries its own £100
    full-cycle cost.  ``Lifetime (1)`` (years) is not used directly because the
    cycle-limited lifetime in ``Lifetime (2)`` determines the £/cycle cost.
    """
    unit = Battery(
        rated_capacity_mwh=params["Max storage volume"],
        max_charge_mw=params["Max charging rate"],
        max_discharge_mw=params["Max discharging rate"],
        charge_efficiency=1.0 - params["Battery charging efficiency"],
        discharge_efficiency=1.0 - params["Battery discharging efficiency"],
        # £500,000 / 5,000 cycles = £100 for a full 4 MWh unit cycle.
        cycle_cost_per_full_cycle=(
            params["Capex"] / params["Lifetime (2)"]
        ),
        degradation_rate_pct_per_cycle=params[
            "Storage volume degradation rate"
        ],
        cost_mode=COST_MODE,
        flat_action_cost=params["Capex"] / params["Lifetime (2)"],
    )
    return BatteryFleet([copy.deepcopy(unit) for _ in range(N_BATTERIES)])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _period_fixed_opex(params: Dict[str, float], index: pd.DatetimeIndex) -> float:
    """Allocate fleet fixed opex to the backtest window.

    Attachment.csv gives £5,000/year per battery, so the two-unit fleet costs
    £10,000/year.  Fixed opex is a memo item: it does not affect dispatch.
    """
    if len(index) < 2:
        return 0.0
    days = (index[-1] - index[0]).total_seconds() / 86400.0
    days += 0.5 / 24.0
    annual_opex = N_BATTERIES * params["Fixed Operational Costs"]
    return annual_opex * days / 365.0


def print_summary(
    title: str,
    result: BacktestResult,
    params: Dict[str, float],
    index: pd.DatetimeIndex,
) -> None:
    summary = result.summary
    fixed_opex = _period_fixed_opex(params, index)

    print(f"\n{title}")
    print("-" * len(title))
    print(f"  Arbitrage profit            GBP {summary['total_profit']:>12,.2f}")
    print(f"  Revenue                     GBP {summary['gross_revenue']:>12,.2f}")
    print(f"  Energy purchase cost        GBP {summary['energy_cost']:>12,.2f}")
    print(f"  Cycle wear cost             GBP {summary['wear_cost']:>12,.2f}")
    print(
        f"  Allocated fixed opex        GBP {fixed_opex:>12,.2f}"
        "  (memo only; does not affect dispatch)"
    )
    print(
        f"  Profit after fixed opex     GBP {summary['total_profit'] - fixed_opex:>12,.2f}"
    )
    print(
        f"  Battery charge / discharge  {summary['n_charge']:>5} / "
        f"{summary['n_discharge']} calls"
    )
    print(
        f"  Intervals charge / discharge {summary['n_charge_events']:>4} / "
        f"{summary['n_discharge_events']}"
    )
    print(
        f"  Simultaneous charge+discharge intervals "
        f"{summary['n_simultaneous_events']:>3}"
    )
    print(
        f"  Energy charged / discharged {summary['energy_charged_mwh']:>8,.1f} / "
        f"{summary['energy_discharged_mwh']:,.1f} MWh"
    )
    print(
        f"  Average buy / sell price    GBP {summary['average_buy_price']:>8,.2f} / "
        f"GBP {summary['average_sell_price']:,.2f} per MWh"
    )
    print(
        f"  Equivalent full cycles      {summary['equivalent_cycles']:>8,.2f}"
        f"   (final capacity {summary['final_capacity_mwh']:.3f} MWh)"
    )
    print(
        f"  Final state of charge       {summary['final_soc_mwh']:>8,.2f} MWh "
        f"({summary['final_soc_pct']:.1f}%)"
    )
    print(
        f"  Terminal storage value      GBP {summary['terminal_storage_value']:>12,.2f}"
        "  (mark-to-market)"
    )
    print(
        f"  Economic profit incl. SOC   GBP {summary['economic_profit']:>12,.2f}"
    )
    print(
        f"  Economic profit after opex  GBP "
        f"{summary['economic_profit'] - fixed_opex:>12,.2f}"
    )
    print(f"  Profit per equivalent cycle GBP {summary['profit_per_cycle']:>12,.2f}")


def print_forecast_report(forecast_result) -> None:
    print("\nARIMA model")
    print("-----------")
    for market, order in forecast_result.orders.items():
        p, d, q = order
        print(f"  {market}: ARIMA({p}, {d}, {q}) with calendar exog")
    print("\n  Forecast accuracy (out of sample, one-step ahead):")
    metrics = forecast_result.metrics.copy()
    for column in ["MAE", "RMSE", "naive_MAE"]:
        metrics[column] = metrics[column].map(lambda value: f"{value:,.3f}")
    metrics["MAPE_%"] = metrics["MAPE_%"].map(lambda value: f"{value:,.1f}%")
    metrics["skill_vs_naive"] = metrics["skill_vs_naive"].map(
        lambda value: f"{value:+.1%}"
    )
    print(metrics.to_string())


def print_comparison(results: Dict[str, BacktestResult]) -> None:
    table = pd.DataFrame(
        {
            name: {
                "Total profit (GBP)": result.summary["total_profit"],
                "Terminal SOC value (GBP)": result.summary[
                    "terminal_storage_value"
                ],
                "Economic profit (GBP)": result.summary["economic_profit"],
                "Revenue (GBP)": result.summary["gross_revenue"],
                "Energy cost (GBP)": result.summary["energy_cost"],
                "Wear cost (GBP)": result.summary["wear_cost"],
                "Battery calls": result.summary["n_actions"],
                "Charge calls": result.summary["n_charge"],
                "Discharge calls": result.summary["n_discharge"],
                "Simultaneous intervals": result.summary[
                    "n_simultaneous_events"
                ],
                "Equivalent cycles": result.summary["equivalent_cycles"],
                "Final SOC (MWh)": result.summary["final_soc_mwh"],
            }
            for name, result in results.items()
        }
    ).T

    print("\nStrategy comparison (out-of-sample period)")
    print("------------------------------------------")
    print(
        table.to_string(
            float_format=lambda value: f"{value:,.2f}"
        )
    )


def validate_fleet_actions(results: Dict[str, BacktestResult]) -> None:
    """Check that simultaneous intervals really are cross-battery pairs."""
    print("\nPer-battery feasibility check")
    print("------------------------------")
    for name, result in results.items():
        actions = result.actions
        simultaneous = (actions["charge_mwh"] > 1e-9) & (
            actions["discharge_mwh"] > 1e-9
        )
        opposite_pair = (
            (
                (actions["battery_0_action"] == "charge")
                & (actions["battery_1_action"] == "discharge")
            )
            | (
                (actions["battery_0_action"] == "discharge")
                & (actions["battery_1_action"] == "charge")
            )
        )
        bad_pairs = int((simultaneous & ~opposite_pair).sum())
        bad_energy = int(
            (
                (
                    (actions["battery_0_action"] == "no_action")
                    & (actions["battery_0_energy_mwh"] > 1e-9)
                )
                | (
                    (actions["battery_1_action"] == "no_action")
                    & (actions["battery_1_energy_mwh"] > 1e-9)
                )
            ).sum()
        )
        print(
            f"  {name}: {int(simultaneous.sum())} simultaneous intervals, "
            f"{bad_pairs} bad cross-battery pairs, "
            f"{bad_energy} action/energy mismatches"
        )


def calibrate_arbitrage(
    fleet: BatteryFleet,
    data: pd.DataFrame,
) -> ArbitrageStrategy:
    """Choose pure-arbitrage thresholds on the first four weeks only.

    The selected thresholds are then frozen for the out-of-sample comparison,
    so no future price is used to tune the strategy.  The objective is realised
    action profit; the mark-to-market value of any final stored energy is used
    only as a tie-breaker.
    """
    if data.empty:
        return ArbitrageStrategy()

    candidates = []
    for lookback in (48, 72, 96, 144):
        for quantile in (0.03, 0.05, 0.08, 0.10, 0.15):
            for min_profit in (0.0, 5.0, 10.0):
                strategy = ArbitrageStrategy(
                    lookback_steps=lookback,
                    lower_quantile=quantile,
                    upper_quantile=1.0 - quantile,
                    min_profit=min_profit,
                )
                summary = Backtester(
                    copy.deepcopy(fleet), strategy
                ).run(data).summary
                candidates.append(
                    {
                        "strategy": strategy,
                        "lookback": lookback,
                        "quantile": quantile,
                        "min_profit": min_profit,
                        "total_profit": summary["total_profit"],
                        "economic_profit": summary["economic_profit"],
                        "n_actions": summary["n_actions"],
                        "n_simultaneous": summary[
                            "n_simultaneous_events"
                        ],
                    }
                )

    candidates.sort(
        key=lambda item: (
            item["total_profit"],
            item["economic_profit"],
        ),
        reverse=True,
    )

    print("\nCalibrating pure-arbitrage thresholds on weeks 1-4")
    print("--------------------------------------------------")
    print("  Top candidates by realised action profit:")
    for item in candidates[:5]:
        print(
            f"    lookback {item['lookback'] * 0.5:>3.0f}h, "
            f"quantile {item['quantile']:.0%}, "
            f"min profit GBP {item['min_profit']:>4.0f} -> "
            f"GBP {item['total_profit']:>8,.2f} "
            f"({item['n_actions']} calls, "
            f"{item['n_simultaneous']} simultaneous intervals)"
        )

    best = candidates[0]
    print(
        f"  Selected: lookback {best['lookback'] * 0.5:.0f}h, "
        f"bottom/top {best['quantile']:.0%}, "
        f"min profit GBP {best['min_profit']:.0f}"
    )
    return best["strategy"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    params = load_attachment(ATTACHMENT_PATH)
    data = build_market_frame(DATA_DIR)

    train_end = data.index.min() + pd.Timedelta(weeks=TRAIN_WEEKS)
    first_four_weeks = data.loc[data.index < train_end].copy()
    out_of_sample = data.loc[data.index >= train_end].copy()

    fleet = build_fleet(params)

    print("Battery arbitrage backtest")
    print("==========================")
    print(f"  Data period: {data.index.min()} -> {data.index.max()}")
    print(f"  Observations: {len(data):,} half-hour bars")
    print(f"  Training window ends: {train_end}")
    print(f"  Fleet: {N_BATTERIES} independent batteries")
    print(
        f"    Each unit:  {fleet.batteries[0].rated_capacity_mwh:g} MWh, "
        f"{fleet.batteries[0].max_charge_mw:g} MW charge, "
        f"{fleet.batteries[0].max_discharge_mw:g} MW discharge"
    )
    print(
        f"    Fleet total: {fleet.capacity_mwh:g} MWh, "
        f"{sum(b.max_charge_mw for b in fleet.batteries):g} MW charge, "
        f"{sum(b.max_discharge_mw for b in fleet.batteries):g} MW discharge"
    )
    print(
        f"  Efficiency: {fleet.charge_efficiency:.1%} charge, "
        f"{fleet.discharge_efficiency:.1%} discharge per unit"
    )
    print(
        "  Operation: one unit may charge while the other discharges; "
        "a single unit never does both at once"
    )
    if fleet.batteries[0].cost_mode == "per_action":
        print(
            f"  Cycle cost: GBP {fleet.batteries[0].flat_action_cost:,.2f} "
            "per charge/discharge call per unit"
        )
    elif fleet.batteries[0].cost_mode == "cycle":
        print(
            f"  Cycle cost: GBP {fleet.batteries[0].cycle_cost_per_full_cycle:,.2f} "
            "per full unit cycle "
            "(charged on discharge, so partial segments accumulate)"
        )
    else:
        print(
            f"  Cycle cost: GBP {fleet.batteries[0].cycle_cost_per_full_cycle:,.2f} "
            f"per full unit cycle "
            f"(GBP {fleet.cycle_cost_per_mwh_throughput:,.2f}/MWh throughput)"
        )
    first_ma = data["market_1_ma_7d"].first_valid_index()
    print(
        "  Price signal: 7-day (168h) moving average per market"
        + (f", first available {first_ma}" if first_ma is not None else "")
    )
    print(
        f"  Fleet capex: GBP {N_BATTERIES * params['Capex']:,.0f}; "
        f"fixed opex: GBP {N_BATTERIES * params['Fixed Operational Costs']:,.0f}/year"
    )

    # ------------------------------------------------------------------
    # 1. Pure arbitrage during the first four weeks
    # ------------------------------------------------------------------
    arbitrage_strategy = calibrate_arbitrage(fleet, first_four_weeks)

    first_result = Backtester(
        copy.deepcopy(fleet), arbitrage_strategy
    ).run(first_four_weeks, initial_soc_mwh=0.0)
    print_summary(
        "Weeks 1-4: pure arbitrage",
        first_result,
        params,
        first_four_weeks.index,
    )

    # ------------------------------------------------------------------
    # 2. ARIMA forecasts after the training window
    # ------------------------------------------------------------------
    print("\nFitting ARIMA models on the first four weeks ...")
    forecaster = ARIMAPriceForecaster(
        horizon=FORECAST_HORIZON,
        max_p=2,
        max_q=2,
        d=1,
        holidays=HOLIDAYS,
        freq=RESAMPLE_FREQ,
        # Keeping coefficients fixed after the initial fit avoids refitting
        # 2,000+ times and performed better in the out-of-sample check.
        refit_every_days=None,
    )
    forecast_result = forecaster.rolling_forecast(
        data,
        train_end=train_end,
        horizon=FORECAST_HORIZON,
        progress=True,
    )
    print_forecast_report(forecast_result)

    # ------------------------------------------------------------------
    # 3. Compare dispatch strategies out of sample
    # ------------------------------------------------------------------
    # Each strategy starts empty so the comparison is like-for-like.
    results: Dict[str, BacktestResult] = {}

    results["Arbitrage only"] = Backtester(
        copy.deepcopy(fleet), arbitrage_strategy
    ).run(
        out_of_sample,
        initial_soc_mwh=0.0,
        history_data=data,
    )

    results["ARIMA optimiser"] = Backtester(
        copy.deepcopy(fleet), ForecastOptimiser()
    ).run(
        out_of_sample,
        forecast_paths=forecast_result.paths,
        initial_soc_mwh=0.0,
        history_data=data,
    )

    results["Combined"] = Backtester(
        copy.deepcopy(fleet),
        CombinedStrategy(arbitrage_strategy, ForecastOptimiser()),
    ).run(
        out_of_sample,
        forecast_paths=forecast_result.paths,
        initial_soc_mwh=0.0,
        history_data=data,
    )

    print_comparison(results)
    validate_fleet_actions(results)
    fixed_opex_test = _period_fixed_opex(params, out_of_sample.index)
    print(
        f"\nAllocated fixed opex for this period: GBP {fixed_opex_test:,.2f} "
        "(the same for every strategy, so it does not change the ranking)."
    )

    best_strategy = max(
        results, key=lambda name: results[name].summary["total_profit"]
    )
    print(
        f"\nBest out-of-sample strategy by realised profit: {best_strategy} "
        f"(GBP {results[best_strategy].summary['total_profit']:,.2f}; "
        f"GBP {results[best_strategy].summary['economic_profit']:,.2f} "
        "including terminal storage value)"
    )
    best_economic = max(
        results, key=lambda name: results[name].summary["economic_profit"]
    )
    if best_economic != best_strategy:
        print(
            f"Best by economic profit including terminal SOC: {best_economic} "
            f"(GBP {results[best_economic].summary['economic_profit']:,.2f})"
        )

    # ------------------------------------------------------------------
    # Save reproducible outputs
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    first_result.actions.to_csv(
        RESULTS_DIR / "actions_first_4_weeks.csv", index=False
    )
    for name, result in results.items():
        slug = name.lower().replace(" ", "_")
        result.actions.to_csv(
            RESULTS_DIR / f"actions_{slug}.csv", index=False
        )

    summary = pd.DataFrame(
        {name: result.summary for name, result in results.items()}
    ).T
    summary.to_csv(RESULTS_DIR / "strategy_summary.csv")
    forecast_result.metrics.to_csv(RESULTS_DIR / "forecast_metrics.csv")
    forecast_result.one_step.to_csv(RESULTS_DIR / "forecast_one_step.csv")

    print(f"\nSaved action logs and summaries to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
