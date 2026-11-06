"""Data, backtest and presentation services for the local dashboard.

The dashboard deliberately reuses the project modules (``main.py``,
``optimiser.py`` and ``prediction_model.py``) rather than duplicating the
model.  Everything here is read-only: a request simulates the strategy from
the start of the sample up to the selected half-hour and then asks the same
strategy what it would do next.
"""

from __future__ import annotations

import copy
import math
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from main import (
    DATA_DIR,
    FORECAST_HORIZON,
    HOLIDAYS,
    RESAMPLE_FREQ,
    TRAIN_WEEKS,
    build_fleet,
    build_market_frame,
    calibrate_arbitrage,
    load_attachment,
    ATTACHMENT_PATH,
)
from optimiser import (
    EPS,
    MARKET_1,
    MARKET_2,
    NO_ACTION,
    ArbitrageStrategy,
    BatteryFleet,
    CombinedStrategy,
    Dispatch,
    ForecastOptimiser,
)
from prediction_model import ARIMAPriceForecaster

CACHE_DIR = Path(__file__).resolve().parent / "cache"
FORECAST_CACHE_PATH = CACHE_DIR / "forecasts.pkl"

_lock = threading.RLock()
_data: Optional[pd.DataFrame] = None
_params: Optional[Dict[str, float]] = None
_base_fleet: Optional[BatteryFleet] = None
_arbitrage_strategy: Optional[ArbitrageStrategy] = None
_combined_strategy: Optional[CombinedStrategy] = None
_forecasts: Optional[Dict[pd.Timestamp, np.ndarray]] = None
_forecast_meta: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Cached inputs
# ---------------------------------------------------------------------------
def get_data() -> pd.DataFrame:
    """Aligned half-hourly prices, holiday flag and 7-day averages."""
    global _data
    with _lock:
        if _data is None:
            _data = build_market_frame(DATA_DIR)
        return _data


def get_params() -> Dict[str, float]:
    global _params
    with _lock:
        if _params is None:
            _params = load_attachment(ATTACHMENT_PATH)
        return _params


def get_train_end() -> pd.Timestamp:
    data = get_data()
    return data.index.min() + pd.Timedelta(weeks=TRAIN_WEEKS)


def get_base_fleet() -> BatteryFleet:
    global _base_fleet
    with _lock:
        if _base_fleet is None:
            _base_fleet = build_fleet(get_params())
        return _base_fleet


def get_arbitrage_strategy() -> ArbitrageStrategy:
    global _arbitrage_strategy
    with _lock:
        if _arbitrage_strategy is None:
            data = get_data()
            train_end = get_train_end()
            first_four_weeks = data.loc[data.index < train_end]
            _arbitrage_strategy = calibrate_arbitrage(
                get_base_fleet(), first_four_weeks
            )
        return _arbitrage_strategy


def get_combined_strategy() -> CombinedStrategy:
    global _combined_strategy
    with _lock:
        if _combined_strategy is None:
            _combined_strategy = CombinedStrategy(
                get_arbitrage_strategy(), ForecastOptimiser()
            )
        return _combined_strategy


# ---------------------------------------------------------------------------
# Forecast cache
# ---------------------------------------------------------------------------
def _save_forecast_cache(result) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "paths": result.paths,
        "orders": result.orders,
        "metrics": result.metrics,
        "horizon": result.horizon,
    }
    with FORECAST_CACHE_PATH.open("wb") as handle:
        pickle.dump(payload, handle)


def build_and_cache_forecasts():
    """Run the rolling ARIMA forecast and persist it for fast page loads."""
    data = get_data()
    forecaster = ARIMAPriceForecaster(
        horizon=FORECAST_HORIZON,
        max_p=2,
        max_q=2,
        d=1,
        holidays=HOLIDAYS,
        freq=RESAMPLE_FREQ,
        refit_every_days=None,
    )
    result = forecaster.rolling_forecast(
        data,
        train_end=get_train_end(),
        horizon=FORECAST_HORIZON,
        progress=True,
    )
    _save_forecast_cache(result)

    global _forecasts, _forecast_meta
    _forecasts = result.paths
    _forecast_meta = {
        "orders": result.orders,
        "metrics": result.metrics,
        "horizon": result.horizon,
    }
    return result


def get_forecasts() -> Dict[pd.Timestamp, np.ndarray]:
    """Load the cached forecast paths, computing them once if necessary."""
    global _forecasts, _forecast_meta
    with _lock:
        if _forecasts is not None:
            return _forecasts

        if FORECAST_CACHE_PATH.exists():
            with FORECAST_CACHE_PATH.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict) and "paths" in payload:
                _forecasts = payload["paths"]
                _forecast_meta = {
                    "orders": payload.get("orders"),
                    "metrics": payload.get("metrics"),
                    "horizon": payload.get("horizon"),
                }
            else:  # tolerate an older cache containing only paths
                _forecasts = payload
                _forecast_meta = {}
            return _forecasts

        result = build_and_cache_forecasts()
        return result.paths


def get_forecast_meta() -> Dict[str, object]:
    get_forecasts()
    return _forecast_meta


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------
def resolve_time(
    data: pd.DataFrame, raw_value: Optional[str]
) -> Tuple[pd.Timestamp, Optional[str]]:
    """Clamp a user-supplied datetime to an available half-hour bar."""
    latest = data.index.max()
    if not raw_value:
        return latest, None

    timestamp = pd.to_datetime(raw_value, errors="coerce")
    if pd.isna(timestamp):
        return latest, "That datetime could not be read; showing the latest bar."
    if getattr(timestamp, "tzinfo", None) is not None:
        try:
            timestamp = timestamp.tz_localize(None)
        except TypeError:
            timestamp = timestamp.tz_convert(None)

    if timestamp < data.index.min():
        return (
            data.index.min(),
            "The requested time is before the sample; showing the first bar.",
        )
    if timestamp > data.index.max():
        return (
            data.index.max(),
            "The requested time is after the sample; showing the latest bar.",
        )

    position = int(data.index.searchsorted(timestamp, side="right") - 1)
    position = max(0, min(position, len(data.index) - 1))
    selected = data.index[position]
    if selected != timestamp:
        return (
            selected,
            f"Showing the latest half-hour bar at or before "
            f"{timestamp:%d %b %Y %H:%M}.",
        )
    return selected, None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    selected_time: pd.Timestamp
    action_records: List[Dict[str, object]]
    realised_profit: float
    revenue: float
    energy_cost: float
    wear_cost: float
    terminal_value: float
    economic_profit: float
    fleet: BatteryFleet
    battery_rows: List[Dict[str, object]]
    prices: Dict[str, float]
    averages: Dict[str, Optional[float]]
    suggested: Dispatch
    suggested_items: List[Dict[str, object]]
    forecast: Optional[pd.DataFrame]
    forecast_rows: List[Dict[str, object]]
    forecast_available: bool


def _dt_hours(data: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    position = data.index.get_loc(timestamp)
    if isinstance(position, slice):
        position = position.start
    if position + 1 < len(data.index):
        return (
            data.index[position + 1] - timestamp
        ).total_seconds() / 3600.0
    return 0.5


def _market_label(market: Optional[str]) -> str:
    if not market:
        return "—"
    return market.replace("_", " ").title()


def _signed_money(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}£{abs(value):,.2f}"


def _action_record(
    timestamp: pd.Timestamp,
    dispatch: Dispatch,
    execution,
) -> Dict[str, object]:
    actual = {action.battery_index: action for action in execution.actual_actions}

    def battery_field(index: int, field_name: str) -> object:
        action = actual.get(index)
        if action is None:
            return "—" if field_name == "action" else None
        if field_name == "action":
            return action.action.title()
        if field_name == "energy":
            return action.energy_mwh
        if field_name == "market":
            return _market_label(action.market)
        if field_name == "price":
            return action.price
        return None

    if execution.charge_mwh > EPS and execution.discharge_mwh > EPS:
        label = "Charge + discharge"
    elif execution.charge_mwh > EPS:
        label = "Charge"
    elif execution.discharge_mwh > EPS:
        label = "Discharge"
    else:
        label = "No action"

    return {
        "time": timestamp,
        "time_label": timestamp.strftime("%d %b %Y %H:%M"),
        "label": label,
        "tone": label.lower().replace(" + ", "-").replace(" ", "-"),
        "battery_0_action": battery_field(0, "action"),
        "battery_0_energy": battery_field(0, "energy"),
        "battery_0_market": battery_field(0, "market"),
        "battery_0_price": battery_field(0, "price"),
        "battery_1_action": battery_field(1, "action"),
        "battery_1_energy": battery_field(1, "energy"),
        "battery_1_market": battery_field(1, "market"),
        "battery_1_price": battery_field(1, "price"),
        "charge_mwh": execution.charge_mwh,
        "discharge_mwh": execution.discharge_mwh,
        "cash_flow": execution.cash_flow,
        "soc_mwh": execution.soc_mwh,
        "cycles": execution.equivalent_cycles,
        "reason": dispatch.reason,
    }


def build_snapshot(
    data: pd.DataFrame,
    selected: pd.Timestamp,
    strategy: CombinedStrategy,
    forecasts: Dict[pd.Timestamp, np.ndarray],
) -> Snapshot:
    """Simulate every bar strictly before ``selected`` and suggest the next action."""
    fleet = copy.deepcopy(get_base_fleet())
    fleet.reset(0.0)

    records: List[Dict[str, object]] = []
    realised_profit = 0.0
    revenue = 0.0
    energy_cost = 0.0
    wear_cost = 0.0

    history_before = data.loc[data.index < selected]
    for timestamp, row in history_before.iterrows():
        prices = {
            MARKET_1: float(row["market_1_price"]),
            MARKET_2: float(row["market_2_price"]),
        }
        history = data.loc[:timestamp]
        dispatch = strategy.decide(
            timestamp=timestamp,
            prices=prices,
            history=history,
            fleet=fleet,
            dt_hours=_dt_hours(data, timestamp),
            forecast_path=forecasts.get(timestamp),
        )
        execution = fleet.execute(dispatch, _dt_hours(data, timestamp))

        realised_profit += execution.cash_flow
        revenue += execution.revenue
        energy_cost += execution.energy_cost
        wear_cost += execution.wear_cost

        if execution.charge_mwh + execution.discharge_mwh > EPS:
            records.append(_action_record(timestamp, dispatch, execution))

    selected_row = data.loc[selected]
    prices = {
        MARKET_1: float(selected_row["market_1_price"]),
        MARKET_2: float(selected_row["market_2_price"]),
    }
    averages = {
        MARKET_1: _optional_float(selected_row.get("market_1_ma_7d")),
        MARKET_2: _optional_float(selected_row.get("market_2_ma_7d")),
    }

    # Mark the current inventory to market, deducting cycle wear that has been
    # deferred until the stored energy is discharged.
    terminal_sell_price = max(prices.values())
    terminal_value = 0.0
    for battery in fleet.batteries:
        grid_energy = battery.soc_mwh * battery.discharge_efficiency
        if grid_energy <= EPS:
            continue
        wear = battery.wear_cost_for_discharge(grid_energy)
        terminal_value += grid_energy * terminal_sell_price - wear

    # Suggested action for the selected bar, using the state before acting.
    suggested = strategy.decide(
        timestamp=selected,
        prices=prices,
        history=data.loc[:selected],
        fleet=fleet,
        dt_hours=_dt_hours(data, selected),
        forecast_path=forecasts.get(selected),
    )
    suggested_items = []
    for action in suggested.actions:
        if action.action == NO_ACTION or action.energy_mwh <= EPS:
            continue
        suggested_items.append(
            {
                "battery": f"Battery {action.battery_index + 1}",
                "action": action.action.title(),
                "energy": action.energy_mwh,
                "market": _market_label(action.market),
                "price": action.price,
            }
        )

    forecast = _forecast_frame(selected, forecasts.get(selected))
    forecast_rows = _forecast_rows(forecast)

    last_actions: Dict[int, Dict[str, object]] = {}
    for record in records:
        for index in (0, 1):
            if record[f"battery_{index}_action"] != "—":
                last_actions[index] = record

    battery_rows = []
    for index, battery in enumerate(fleet.batteries):
        last = last_actions.get(index)
        battery_rows.append(
            {
                "index": index,
                "name": f"Battery {index + 1}",
                "soc_mwh": battery.soc_mwh,
                "capacity_mwh": battery.capacity_mwh,
                "soc_pct": (
                    100.0 * battery.soc_mwh / battery.capacity_mwh
                    if battery.capacity_mwh > EPS
                    else 0.0
                ),
                "cycles": battery.equivalent_cycles,
                "stored_cost": battery.stored_cost,
                "average_cost": battery.average_stored_cost_per_mwh,
                "dischargeable_mwh": (
                    battery.soc_mwh * battery.discharge_efficiency
                ),
                "room_mwh": max(0.0, battery.capacity_mwh - battery.soc_mwh),
                "last_action": last,
            }
        )

    return Snapshot(
        selected_time=selected,
        action_records=records,
        realised_profit=realised_profit,
        revenue=revenue,
        energy_cost=energy_cost,
        wear_cost=wear_cost,
        terminal_value=terminal_value,
        economic_profit=realised_profit + terminal_value,
        fleet=fleet,
        battery_rows=battery_rows,
        prices=prices,
        averages=averages,
        suggested=suggested,
        suggested_items=suggested_items,
        forecast=forecast,
        forecast_rows=forecast_rows,
        forecast_available=forecast is not None,
    )


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0.0:
        return None
    return number


def _forecast_frame(
    selected: pd.Timestamp, path: Optional[np.ndarray]
) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    array = np.asarray(path, dtype=float)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).any():
        return None
    index = pd.date_range(selected, periods=len(array), freq=RESAMPLE_FREQ)
    return pd.DataFrame(
        array, index=index, columns=["market_1_price", "market_2_price"]
    )


def _forecast_rows(
    forecast: Optional[pd.DataFrame],
) -> List[Dict[str, object]]:
    if forecast is None:
        return []
    rows = []
    for timestamp, row in forecast.iterrows():
        rows.append(
            {
                "time": timestamp,
                "time_label": timestamp.strftime("%d %b %H:%M"),
                "market_1": float(row["market_1_price"]),
                "market_2": float(row["market_2_price"]),
            }
        )
    return rows


def recent_price_rows(
    data: pd.DataFrame, selected: pd.Timestamp, limit: int = 12
) -> List[Dict[str, object]]:
    """The most recent half-hourly prices and averages, newest first."""
    recent = data.loc[:selected].tail(limit)
    rows: List[Dict[str, object]] = []
    for timestamp, row in recent.iloc[::-1].iterrows():
        rows.append(
            {
                "time_label": timestamp.strftime("%d %b %H:%M"),
                "market_1": float(row["market_1_price"]),
                "market_2": float(row["market_2_price"]),
                "market_1_ma": _optional_float(row.get("market_1_ma_7d")),
                "market_2_ma": _optional_float(row.get("market_2_ma_7d")),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def _downsample(series: pd.Series, max_points: int = 900) -> pd.Series:
    series = series.dropna()
    if len(series) <= max_points:
        return series
    step = int(math.ceil(len(series) / max_points))
    sampled = series.iloc[::step]
    if len(sampled) and sampled.index[-1] != series.index[-1]:
        sampled = pd.concat([sampled, series.iloc[[-1]]])
    return sampled


def build_chart_svg(
    data: pd.DataFrame,
    selected: pd.Timestamp,
    forecast: Optional[pd.DataFrame],
) -> str:
    """Return a self-contained SVG chart (no external JavaScript required)."""
    history = data.loc[:selected]
    if history.empty:
        return ""

    width, height = 1120, 390
    left, top, right, bottom = 66, 18, 18, 44
    plot_width = width - left - right
    plot_height = height - top - bottom

    end_time = (
        forecast.index[-1]
        if forecast is not None and len(forecast)
        else selected
    )
    start_time = history.index[0]
    span = max((end_time - start_time).total_seconds(), 1.0)

    def x_position(timestamp: pd.Timestamp) -> float:
        return left + (
            (timestamp - start_time).total_seconds() / span
        ) * plot_width

    values: List[float] = []
    for column in (
        "market_1_price",
        "market_2_price",
        "market_1_ma_7d",
        "market_2_ma_7d",
    ):
        if column in history.columns:
            values.extend(
                float(value)
                for value in history[column].dropna().to_numpy()
            )
    if forecast is not None:
        values.extend(
            float(value)
            for value in forecast.to_numpy().ravel()
            if np.isfinite(value)
        )
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return ""

    y_min, y_max = min(values), max(values)
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0
    padding = 0.06 * (y_max - y_min)
    y_min -= padding
    y_max += padding
    # Wholesale prices cannot be negative; keep the axis at or above zero.
    y_min = max(0.0, y_min)

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts: List[str] = [
        f'<svg class="price-chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Market prices, 7-day averages and forecast" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    if forecast is not None and len(forecast):
        forecast_x0 = x_position(selected)
        forecast_x1 = x_position(end_time)
        parts.append(
            f'<rect x="{forecast_x0:.2f}" y="{top}" '
            f'width="{max(forecast_x1 - forecast_x0, 0.0):.2f}" '
            f'height="{plot_height}" fill="#eef2ff" opacity="0.75"/>'
        )

    for step in range(5):
        value = y_min + (y_max - y_min) * step / 4.0
        y_pos = y_position(value)
        parts.append(
            f'<line x1="{left}" y1="{y_pos:.2f}" '
            f'x2="{left + plot_width}" y2="{y_pos:.2f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y_pos + 4:.2f}" '
            f'text-anchor="end" class="chart-axis">£{value:,.0f}</text>'
        )

    for step in range(6):
        fraction = step / 5.0
        timestamp = start_time + pd.Timedelta(seconds=span * fraction)
        x_pos = left + fraction * plot_width
        parts.append(
            f'<line x1="{x_pos:.2f}" y1="{top}" '
            f'x2="{x_pos:.2f}" y2="{top + plot_height}" '
            'stroke="#eef2f7" stroke-width="1"/>'
        )
        label = (
            timestamp.strftime("%d %b")
            if span > 3.0 * 86400.0
            else timestamp.strftime("%H:%M")
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="{top + plot_height + 24}" '
            f'text-anchor="middle" class="chart-axis">{label}</text>'
        )

    selected_x = x_position(selected)
    parts.append(
        f'<line x1="{selected_x:.2f}" y1="{top}" '
        f'x2="{selected_x:.2f}" y2="{top + plot_height}" '
        'stroke="#0f172a" stroke-width="1.2" stroke-dasharray="4 4" '
        'opacity="0.65"/>'
    )
    label_on_left = selected_x > left + 0.86 * plot_width
    label_x = selected_x - 6 if label_on_left else selected_x + 6
    label_anchor = "end" if label_on_left else "start"
    parts.append(
        f'<text x="{label_x:.2f}" y="{top + 14}" '
        f'text-anchor="{label_anchor}" '
        'class="chart-note">selected</text>'
    )

    def add_series(
        series: pd.Series,
        colour: str,
        stroke_width: float,
        dash: Optional[str] = None,
    ) -> None:
        sampled = _downsample(series)
        if sampled.empty:
            return
        points = " ".join(
            f"{x_position(timestamp):.2f},{y_position(float(value)):.2f}"
            for timestamp, value in sampled.items()
            if np.isfinite(value)
        )
        if not points:
            return
        dash_attribute = (
            f' stroke-dasharray="{dash}"' if dash is not None else ""
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="{stroke_width}" stroke-linejoin="round" '
            f'stroke-linecap="round"{dash_attribute}/>'
        )

    add_series(history["market_1_price"], "#1d4ed8", 2.0)
    add_series(history["market_2_price"], "#b45309", 2.0)
    if "market_1_ma_7d" in history.columns:
        add_series(history["market_1_ma_7d"], "#60a5fa", 1.4, "5 4")
    if "market_2_ma_7d" in history.columns:
        add_series(history["market_2_ma_7d"], "#f59e0b", 1.4, "5 4")

    if forecast is not None and not forecast.empty:
        for column, colour in (
            ("market_1_price", "#1d4ed8"),
            ("market_2_price", "#b45309"),
        ):
            # Start at the forecast value for the selected bar.  Actual prices
            # after the selected time are deliberately not plotted or shown.
            add_series(forecast[column], colour, 2.0, "2 3")

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" '
        f'height="{plot_height}" fill="none" stroke="#cbd5e1"/>'
    )
    parts.append("</svg>")
    return "".join(parts)
