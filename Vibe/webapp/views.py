"""Single-page dashboard view."""

from __future__ import annotations

from typing import List

import pandas as pd
from django.core.paginator import Paginator
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from . import services


def _money(value, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if signed:
        sign = "+" if number >= 0 else "−"
        return f"{sign}£{abs(number):,.2f}"
    return f"£{number:,.2f}"


def _number(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:,.{digits}f}"


def _market_rows(snapshot: services.Snapshot) -> List[dict]:
    rows = []
    forecast_by_market = {}
    if snapshot.forecast is not None and not snapshot.forecast.empty:
        for key in ("market_1", "market_2"):
            value = snapshot.forecast.iloc[0].get(f"{key}_price")
            forecast_by_market[key] = (
                float(value) if pd.notna(value) else None
            )
    for key, label in (("market_1", "Market 1"), ("market_2", "Market 2")):
        price = snapshot.prices[key]
        average = snapshot.averages[key]
        forecast = forecast_by_market.get(key)
        if average is None:
            relation = "7-day average not ready"
            detail = "Full-size actions until seven days of history exist."
            tone = "muted"
            delta = "—"
        else:
            change = (price - average) / average * 100.0
            delta = f"{change:+.1f}%"
            if price < average:
                relation = "Below 7-day average"
                detail = (
                    "Favourable for charging: full power when buying, "
                    "half power when selling."
                )
                tone = "charge"
            elif price > average:
                relation = "Above 7-day average"
                detail = (
                    "Favourable for discharging: full power when selling, "
                    "half power when buying."
                )
                tone = "discharge"
            else:
                relation = "At the 7-day average"
                detail = "Half-size actions on both sides."
                tone = "muted"
        rows.append(
            {
                "name": label,
                "price": _money(price),
                "average": _money(average),
                "forecast": _money(forecast),
                "delta": delta,
                "relation": relation,
                "detail": detail,
                "tone": tone,
            }
        )
    return rows


def _battery_rows(snapshot: services.Snapshot) -> List[dict]:
    rows = []
    for battery in snapshot.battery_rows:
        last = battery["last_action"]
        last_label = (
            f"{last['time_label']} · {last['label']}"
            if last is not None
            else "No action yet"
        )
        rows.append(
            {
                "name": battery["name"],
                "soc": _number(battery["soc_mwh"]),
                "capacity": _number(battery["capacity_mwh"]),
                "soc_pct": _number(battery["soc_pct"], 1),
                "cycles": _number(battery["cycles"]),
                "average_cost": _money(battery["average_cost"]),
                "dischargeable": _number(battery["dischargeable_mwh"]),
                "room": _number(battery["room_mwh"]),
                "last_action": last_label,
            }
        )
    return rows


def _suggested(snapshot: services.Snapshot) -> dict:
    has_action = bool(snapshot.suggested_items)
    items = [
        {
            "battery": item["battery"],
            "action": item["action"],
            "energy": f"{item['energy']:.2f} MWh",
            "market": item["market"],
            "price": _money(item["price"]),
        }
        for item in snapshot.suggested_items
    ]
    return {
        "has_action": has_action,
        "label": (
            snapshot.suggested.action.replace("_", " ").title()
            if has_action
            else "No action"
        ),
        "reason": snapshot.suggested.reason or "No signal at this time.",
        "expected_profit": (
            _money(snapshot.suggested.expected_profit, signed=True)
            if has_action
            else "—"
        ),
        "items": items,
    }


def _forecast_rows(snapshot: services.Snapshot) -> List[dict]:
    rows = []
    for row in snapshot.forecast_rows:
        rows.append(
            {
                "time": row["time_label"],
                "market_1": _money(row["market_1"]),
                "market_2": _money(row["market_2"]),
            }
        )
    return rows


def _model_details(snapshot: services.Snapshot) -> dict:
    if not snapshot.forecast_available:
        return {"orders": [], "metrics": []}
    meta = services.get_forecast_meta()
    orders = meta.get("orders") or {}
    order_rows = []
    for market, order in orders.items():
        p, d, q = order
        order_rows.append(
            {
                "market": market.replace("_price", "").replace("_", " ").title(),
                "order": f"ARIMA({p}, {d}, {q})",
            }
        )
    metric_rows = []
    metrics = meta.get("metrics")
    if metrics is not None and not metrics.empty:
        for market, row in metrics.iterrows():
            metric_rows.append(
                {
                    "market": market.replace("_price", "")
                    .replace("_", " ")
                    .title(),
                    "mae": _number(row.get("MAE"), 3),
                    "skill": (
                        f"{row.get('skill_vs_naive'):+.1%}"
                        if pd.notna(row.get("skill_vs_naive"))
                        else "—"
                    ),
                }
            )
    return {"orders": order_rows, "metrics": metric_rows}


def health(request: HttpRequest) -> HttpResponse:
    """Lightweight readiness probe."""
    return HttpResponse("ok", content_type="text/plain")


def index(request: HttpRequest) -> HttpResponse:
    data = services.get_data()
    selected, time_note = services.resolve_time(data, request.GET.get("at"))
    train_end = services.get_train_end()

    forecasts = (
        services.get_forecasts() if selected >= train_end else {}
    )
    strategy = services.get_combined_strategy()
    snapshot = services.build_snapshot(data, selected, strategy, forecasts)
    chart_svg = services.build_chart_svg(data, selected, snapshot.forecast)

    display_records = list(reversed(snapshot.action_records))
    paginator = Paginator(display_records, 25)
    page_obj = paginator.get_page(request.GET.get("page"))

    fleet = snapshot.fleet
    fleet_capacity = fleet.capacity_mwh
    fleet_soc_pct = (
        100.0 * fleet.soc_mwh / fleet_capacity if fleet_capacity > 0 else 0.0
    )

    kpis = [
        {
            "label": "Realised profit",
            "value": _money(snapshot.realised_profit, signed=True),
            "hint": "Cash flow since 01 Jan 2018, before the selected time",
            "tone": "profit" if snapshot.realised_profit >= 0 else "loss",
        },
        {
            "label": "Economic profit",
            "value": _money(snapshot.economic_profit, signed=True),
            "hint": "Realised plus stored energy at market",
            "tone": "profit" if snapshot.economic_profit >= 0 else "loss",
        },
        {
            "label": "Fleet state of charge",
            "value": f"{_number(fleet.soc_mwh)} / {_number(fleet_capacity)} MWh",
            "hint": f"{_number(fleet_soc_pct, 1)}% full · "
            f"{_number(fleet.equivalent_cycles)} cycles",
            "tone": "neutral",
        },
        {
            "label": "Actions executed",
            "value": f"{len(snapshot.action_records):,}",
            "hint": "Charge or discharge intervals before this time",
            "tone": "neutral",
        },
    ]

    profit_rows = [
        {"label": "Revenue", "value": _money(snapshot.revenue)},
        {"label": "Energy purchase cost", "value": _money(snapshot.energy_cost)},
        {"label": "Cycle wear", "value": _money(snapshot.wear_cost)},
        {
            "label": "Stored energy value",
            "value": _money(snapshot.terminal_value),
        },
    ]

    if selected < train_end:
        forecast_note = (
            "ARIMA forecasts begin after the four-week training window, "
            f"from {train_end:%d %b %Y %H:%M}."
        )
    elif not snapshot.forecast_available:
        forecast_note = "No forecast is available at this bar."
    else:
        forecast_note = (
            f"Forecast origin {selected:%d %b %Y %H:%M}; "
            f"{len(snapshot.forecast_rows)} half-hour steps."
        )

    context = {
        "selected_label": selected.strftime("%d %b %Y, %H:%M"),
        "selected_input": selected.strftime("%Y-%m-%dT%H:%M"),
        "data_start_input": data.index.min().strftime("%Y-%m-%dT%H:%M"),
        "data_end_input": data.index.max().strftime("%Y-%m-%dT%H:%M"),
        "data_range_label": (
            f"{data.index.min():%d %b %Y %H:%M} – "
            f"{data.index.max():%d %b %Y %H:%M}"
        ),
        "train_end_label": train_end.strftime("%d %b %Y %H:%M"),
        "time_note": time_note,
        "kpis": kpis,
        "profit_rows": profit_rows,
        "market_rows": _market_rows(snapshot),
        "recent_prices": [
            {
                "time": row["time_label"],
                "market_1": _money(row["market_1"]),
                "market_2": _money(row["market_2"]),
                "market_1_ma": _money(row["market_1_ma"]),
                "market_2_ma": _money(row["market_2_ma"]),
            }
            for row in services.recent_price_rows(data, selected)
        ],
        "battery_rows": _battery_rows(snapshot),
        "fleet": {
            "soc": _number(fleet.soc_mwh),
            "capacity": _number(fleet_capacity),
            "soc_pct": _number(fleet_soc_pct, 1),
            "cycles": _number(fleet.equivalent_cycles),
            "average_cost": _money(fleet.average_stored_cost_per_mwh),
            "charge_mw": _number(
                sum(battery.max_charge_mw for battery in fleet.batteries)
            ),
            "discharge_mw": _number(
                sum(battery.max_discharge_mw for battery in fleet.batteries)
            ),
        },
        "suggested": _suggested(snapshot),
        "forecast_available": snapshot.forecast_available,
        "forecast_note": forecast_note,
        "forecast_rows": _forecast_rows(snapshot),
        "model": _model_details(snapshot),
        "chart_svg": chart_svg,
        "page_obj": page_obj,
        "action_count": len(snapshot.action_records),
        "query_at": selected.strftime("%Y-%m-%dT%H:%M"),
        "realised_profit_label": _money(
            snapshot.realised_profit, signed=True
        ),
    }
    return render(request, "webapp/index.html", context)
