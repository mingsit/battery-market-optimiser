"""Battery dispatch model and action optimisers.

The module now models the two batteries independently:

1. ``Battery`` - one 4 MWh / 2 MW unit with efficiencies, cycle wear and
   degradation.
2. ``BatteryFleet`` - the two units, which may charge/discharge at the same
   time as long as no single battery does both in the same interval.
3. ``ArbitrageStrategy`` / ``ForecastOptimiser`` / ``CombinedStrategy`` -
   decision rules that return a per-battery dispatch plan.
4. ``Backtester`` - runs a strategy over a price history and records cash flows.

Important modelling assumptions
-------------------------------
* Each battery is 4 MWh / 2 MW, so the fleet is 8 MWh / 4 MW.
* A full cycle is defined as charging one battery from empty to full and then
  discharging all of its stored energy, even when that happens in several
  segments (75% then 25% still counts as one cycle).  The £100 cycle cost
  (``£500,000 / 5,000``) is therefore charged when stored energy is discharged,
  in proportion to the energy removed.
* Each market carries a 168-hour moving average.  When a full week of history
  is available, prices on the favourable side of that average trade at full
  power and the other side at half power.
* Charge and discharge efficiencies are applied in the physical direction:
  grid -> battery loses 5%, battery -> grid loses 5%.
* Market 2 is hourly.  The backtest runs on a half-hourly grid and holds the
  hourly market-2 price constant for both half-hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

MARKET_1 = "market_1"
MARKET_2 = "market_2"
MARKETS = (MARKET_1, MARKET_2)
NO_ACTION = "no_action"
CHARGE = "charge"
DISCHARGE = "discharge"
CHARGE_AND_DISCHARGE = "charge+discharge"
EPS = 1e-9


# ---------------------------------------------------------------------------
# Single battery
# ---------------------------------------------------------------------------
@dataclass
class Execution:
    """Result of applying one action to one battery."""

    energy_mwh: float = 0.0
    cash_flow: float = 0.0
    energy_cost: float = 0.0
    revenue: float = 0.0
    wear_cost: float = 0.0
    removed_cost: float = 0.0
    soc_mwh: float = 0.0
    equivalent_cycles: float = 0.0


@dataclass
class Battery:
    """One battery unit built from ``Attachment.csv``."""

    rated_capacity_mwh: float
    max_charge_mw: float
    max_discharge_mw: float
    charge_efficiency: float
    discharge_efficiency: float
    cycle_cost_per_full_cycle: float
    degradation_rate_pct_per_cycle: float = 0.0
    # "cycle": £100 per full stored-energy cycle, paid when energy is discharged.
    # "throughput": £100 pro-rata over every MWh of grid throughput.
    # "per_action": a flat £100 for every charge or discharge call.
    cost_mode: str = "cycle"
    flat_action_cost: float = 0.0

    soc_mwh: float = 0.0
    equivalent_cycles: float = 0.0
    stored_cost: float = 0.0

    def reset(self, initial_soc_mwh: float = 0.0) -> None:
        self.equivalent_cycles = 0.0
        self.stored_cost = 0.0
        self.soc_mwh = float(np.clip(initial_soc_mwh, 0.0, self.capacity_mwh))

    # -- physical properties -------------------------------------------------
    @property
    def capacity_mwh(self) -> float:
        """Current usable capacity after cumulative cycle degradation."""
        loss = (
            self.degradation_rate_pct_per_cycle
            / 100.0
            * self.equivalent_cycles
        )
        return max(0.0, self.rated_capacity_mwh * (1.0 - loss))

    @property
    def full_cycle_throughput_mwh(self) -> float:
        """Grid-side throughput of one full empty -> full -> empty cycle."""
        return (
            self.rated_capacity_mwh / self.charge_efficiency
            + self.rated_capacity_mwh * self.discharge_efficiency
        )

    @property
    def cycle_cost_per_mwh_throughput(self) -> float:
        if self.cost_mode == "per_action":
            return 0.0
        return self.cycle_cost_per_full_cycle / self.full_cycle_throughput_mwh

    @property
    def action_cost(self) -> float:
        if self.cost_mode == "per_action":
            return self.flat_action_cost
        return 0.0

    @property
    def average_stored_cost_per_mwh(self) -> float:
        if self.soc_mwh <= EPS:
            return 0.0
        return self.stored_cost / self.soc_mwh

    def max_charge_grid_energy(self, dt_hours: float) -> float:
        room_mwh = max(0.0, self.capacity_mwh - self.soc_mwh)
        return min(
            self.max_charge_mw * dt_hours,
            room_mwh / self.charge_efficiency,
        )

    def max_discharge_grid_energy(self, dt_hours: float) -> float:
        available_mwh = max(0.0, self.soc_mwh * self.discharge_efficiency)
        return min(self.max_discharge_mw * dt_hours, available_mwh)

    # -- actions -------------------------------------------------------------
    def _charge_wear_cost(self, grid_energy_mwh: float) -> float:
        if grid_energy_mwh <= EPS:
            return 0.0
        if self.cost_mode == "per_action":
            return self.flat_action_cost
        if self.cost_mode == "throughput":
            return grid_energy_mwh * self.cycle_cost_per_mwh_throughput
        # The default "cycle" mode pays the full-cycle cost only when stored
        # energy is discharged.  Charging on its own therefore has no wear
        # charge; the cost is deferred until the energy leaves the battery.
        return 0.0

    def _discharge_wear_cost(
        self, grid_energy_mwh: float, removed_energy_mwh: float
    ) -> float:
        if grid_energy_mwh <= EPS:
            return 0.0
        if self.cost_mode == "per_action":
            return self.flat_action_cost
        if self.cost_mode == "throughput":
            return grid_energy_mwh * self.cycle_cost_per_mwh_throughput
        if self.rated_capacity_mwh <= EPS:
            return 0.0
        return (
            removed_energy_mwh / self.rated_capacity_mwh
        ) * self.cycle_cost_per_full_cycle

    def wear_cost_for_discharge(self, grid_energy_mwh: float) -> float:
        """Wear cost if this much grid energy were discharged now."""
        grid_energy = max(float(grid_energy_mwh), 0.0)
        if grid_energy <= EPS:
            return 0.0
        removed_energy = grid_energy / self.discharge_efficiency
        return self._discharge_wear_cost(grid_energy, removed_energy)

    def _register_discharge_cycles(self, removed_energy_mwh: float) -> None:
        """Count cycles from stored energy discharged, not grid throughput.

        A full 4 MWh discharge from full to empty counts as one cycle.  The
        Attachment.csv example - discharge 75% of capacity, then later charge
        and discharge the remaining 25% - therefore counts as 0.75 + 0.25 = 1.
        """
        if self.rated_capacity_mwh > EPS:
            self.equivalent_cycles += (
                removed_energy_mwh / self.rated_capacity_mwh
            )

    def charge(
        self, desired_grid_mwh: float, price: float, dt_hours: float
    ) -> Execution:
        grid_energy = min(
            max(float(desired_grid_mwh), 0.0),
            self.max_charge_grid_energy(dt_hours),
        )
        if grid_energy <= EPS:
            return self._zero_execution()

        stored_energy = grid_energy * self.charge_efficiency
        energy_cost = grid_energy * float(price)
        wear = self._charge_wear_cost(grid_energy)

        self.soc_mwh += stored_energy
        self.stored_cost += energy_cost + wear

        return Execution(
            energy_mwh=grid_energy,
            cash_flow=-(energy_cost + wear),
            energy_cost=energy_cost,
            revenue=0.0,
            wear_cost=wear,
            removed_cost=0.0,
            soc_mwh=self.soc_mwh,
            equivalent_cycles=self.equivalent_cycles,
        )

    def discharge(
        self, desired_grid_mwh: float, price: float, dt_hours: float
    ) -> Execution:
        grid_energy = min(
            max(float(desired_grid_mwh), 0.0),
            self.max_discharge_grid_energy(dt_hours),
        )
        if grid_energy <= EPS:
            return self._zero_execution()

        removed_energy = grid_energy / self.discharge_efficiency
        average_cost = self.average_stored_cost_per_mwh
        removed_cost = average_cost * removed_energy
        revenue = grid_energy * float(price)
        wear = self._discharge_wear_cost(grid_energy, removed_energy)

        self.soc_mwh = max(0.0, self.soc_mwh - removed_energy)
        self.stored_cost = max(0.0, self.stored_cost - removed_cost)
        if self.soc_mwh <= EPS:
            self.stored_cost = 0.0
        self._register_discharge_cycles(removed_energy)

        return Execution(
            energy_mwh=grid_energy,
            cash_flow=revenue - wear,
            energy_cost=0.0,
            revenue=revenue,
            wear_cost=wear,
            removed_cost=removed_cost,
            soc_mwh=self.soc_mwh,
            equivalent_cycles=self.equivalent_cycles,
        )

    def _zero_execution(self) -> Execution:
        return Execution(
            soc_mwh=self.soc_mwh,
            equivalent_cycles=self.equivalent_cycles,
        )


# ---------------------------------------------------------------------------
# Fleet of batteries
# ---------------------------------------------------------------------------
@dataclass
class BatteryAction:
    """One battery's instruction for a single time interval."""

    battery_index: int
    action: str = NO_ACTION
    energy_mwh: float = 0.0
    market: Optional[str] = None
    price: float = 0.0


@dataclass
class Dispatch:
    """A per-battery dispatch plan for one time interval."""

    actions: List[BatteryAction] = field(default_factory=list)
    expected_profit: float = 0.0
    reason: str = ""

    @classmethod
    def no_action(cls, reason: str = "") -> "Dispatch":
        return cls(actions=[], reason=reason)

    @property
    def action(self) -> str:
        directions = {
            action.action
            for action in self.actions
            if action.action != NO_ACTION and action.energy_mwh > EPS
        }
        if directions == {CHARGE}:
            return CHARGE
        if directions == {DISCHARGE}:
            return DISCHARGE
        if directions == {CHARGE, DISCHARGE}:
            return CHARGE_AND_DISCHARGE
        return NO_ACTION

    @property
    def charge_mwh(self) -> float:
        return sum(
            action.energy_mwh
            for action in self.actions
            if action.action == CHARGE
        )

    @property
    def discharge_mwh(self) -> float:
        return sum(
            action.energy_mwh
            for action in self.actions
            if action.action == DISCHARGE
        )


@dataclass
class FleetExecution:
    """Aggregated result of applying a dispatch to the fleet."""

    charge_mwh: float = 0.0
    discharge_mwh: float = 0.0
    cash_flow: float = 0.0
    energy_cost: float = 0.0
    revenue: float = 0.0
    wear_cost: float = 0.0
    soc_mwh: float = 0.0
    equivalent_cycles: float = 0.0
    actual_actions: List[BatteryAction] = field(default_factory=list)


@dataclass
class BatteryFleet:
    """Two independent batteries that may operate simultaneously."""

    batteries: List[Battery]

    def reset(self, initial_soc_mwh: float = 0.0) -> None:
        if not self.batteries:
            return
        per_battery = initial_soc_mwh / len(self.batteries)
        for battery in self.batteries:
            battery.reset(min(per_battery, battery.capacity_mwh))

    # -- fleet properties ----------------------------------------------------
    @property
    def capacity_mwh(self) -> float:
        return sum(battery.capacity_mwh for battery in self.batteries)

    @property
    def soc_mwh(self) -> float:
        return sum(battery.soc_mwh for battery in self.batteries)

    @property
    def equivalent_cycles(self) -> float:
        return sum(battery.equivalent_cycles for battery in self.batteries)

    @property
    def average_stored_cost_per_mwh(self) -> float:
        stored = sum(battery.stored_cost for battery in self.batteries)
        energy = self.soc_mwh
        return stored / energy if energy > EPS else 0.0

    @property
    def charge_efficiency(self) -> float:
        return self.batteries[0].charge_efficiency

    @property
    def discharge_efficiency(self) -> float:
        return self.batteries[0].discharge_efficiency

    @property
    def cycle_cost_per_mwh_throughput(self) -> float:
        return self.batteries[0].cycle_cost_per_mwh_throughput

    @property
    def action_cost(self) -> float:
        return self.batteries[0].action_cost

    def max_charge_grid_energy(self, dt_hours: float) -> float:
        return sum(
            battery.max_charge_grid_energy(dt_hours)
            for battery in self.batteries
        )

    def max_discharge_grid_energy(self, dt_hours: float) -> float:
        return sum(
            battery.max_discharge_grid_energy(dt_hours)
            for battery in self.batteries
        )

    # -- execution -----------------------------------------------------------
    def execute(self, dispatch: Dispatch, dt_hours: float) -> FleetExecution:
        result = FleetExecution()
        seen_batteries = set()

        for action in dispatch.actions:
            if action.action == NO_ACTION or action.energy_mwh <= EPS:
                continue
            if not (0 <= action.battery_index < len(self.batteries)):
                continue
            # A battery has one direction per interval.  If a strategy ever
            # returns two instructions for the same unit, only the first is
            # executed, so charge and discharge can never overlap on one unit.
            if action.battery_index in seen_batteries:
                continue
            seen_batteries.add(action.battery_index)

            battery = self.batteries[action.battery_index]
            if action.action == CHARGE:
                execution = battery.charge(
                    action.energy_mwh, action.price, dt_hours
                )
                result.charge_mwh += execution.energy_mwh
            elif action.action == DISCHARGE:
                execution = battery.discharge(
                    action.energy_mwh, action.price, dt_hours
                )
                result.discharge_mwh += execution.energy_mwh
            else:
                continue

            if execution.energy_mwh <= EPS:
                continue

            result.cash_flow += execution.cash_flow
            result.energy_cost += execution.energy_cost
            result.revenue += execution.revenue
            result.wear_cost += execution.wear_cost
            result.actual_actions.append(
                BatteryAction(
                    battery_index=action.battery_index,
                    action=action.action,
                    energy_mwh=execution.energy_mwh,
                    market=action.market,
                    price=action.price,
                )
            )

        result.soc_mwh = self.soc_mwh
        result.equivalent_cycles = self.equivalent_cycles
        return result


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
class Strategy(Protocol):
    def decide(
        self,
        timestamp: pd.Timestamp,
        prices: Dict[str, float],
        history: pd.DataFrame,
        fleet: BatteryFleet,
        dt_hours: float,
        forecast_path: Optional[np.ndarray] = None,
    ) -> Dispatch:
        ...


def _best_markets(prices: Dict[str, float]) -> Tuple[str, str]:
    buy_market = min(MARKETS, key=lambda market: prices[market])
    sell_market = max(MARKETS, key=lambda market: prices[market])
    return buy_market, sell_market


def _latest_moving_average(
    history: pd.DataFrame, market: str
) -> Optional[float]:
    """Latest 7-day moving average for ``market``, or None if not ready."""
    column = f"{market}_ma_7d"
    if column not in history.columns or history.empty:
        return None
    value = history[column].iloc[-1]
    if pd.isna(value) or float(value) <= 0.0:
        return None
    return float(value)


def _moving_average_size_multiplier(
    price: float,
    moving_average: Optional[float],
    charge: bool,
) -> float:
    """Use the 7-day average to size the action.

    Full available power is used when the current price is on the favourable
    side of its 7-day average (below it for charging, above it for
    discharging).  Otherwise the action is reduced to half size.  Before seven
    days of history exist, no adjustment is made.
    """
    if moving_average is None:
        return 1.0
    favourable = price < moving_average if charge else price > moving_average
    return 1.0 if favourable else 0.5


def _profitable_discharge_energy(
    battery: Battery,
    price: float,
    dt_hours: float,
    min_profit: float,
) -> float:
    """Grid-side energy this battery can profitably discharge at ``price``."""
    energy = battery.max_discharge_grid_energy(dt_hours)
    if energy <= EPS:
        return 0.0

    removed_energy = energy / battery.discharge_efficiency
    wear = battery.wear_cost_for_discharge(energy)
    book_cost = battery.average_stored_cost_per_mwh * removed_energy
    profit = energy * price - wear - book_cost
    return energy if profit > min_profit else 0.0


@dataclass
class ArbitrageStrategy:
    """Pure arbitrage using only prices known at the decision time.

    A battery buys when its market is in the bottom quantile of recent prices
    and sells when the most expensive market is in the top quantile.  The
    latest 7-day moving average is then used to size the action: if the price
    is on the favourable side of its average, the battery uses full power;
    otherwise it uses half power.  If charge and discharge conditions occur at
    once, one battery can charge while the other discharges.
    """

    lookback_steps: int = 96  # 48 hours of half-hourly bars
    lower_quantile: float = 0.10
    upper_quantile: float = 0.90
    min_history: int = 10
    min_profit: float = 0.0

    def decide(
        self,
        timestamp: pd.Timestamp,
        prices: Dict[str, float],
        history: pd.DataFrame,
        fleet: BatteryFleet,
        dt_hours: float,
        forecast_path: Optional[np.ndarray] = None,
    ) -> Dispatch:
        if len(history) <= self.min_history:
            return Dispatch.no_action("warming up")

        window = history.iloc[-(self.lookback_steps + 1) : -1]
        if len(window) < self.min_history:
            window = history.iloc[:-1]

        min_prices = window[["market_1_price", "market_2_price"]].min(axis=1)
        max_prices = window[["market_1_price", "market_2_price"]].max(axis=1)
        low_threshold = float(min_prices.quantile(self.lower_quantile))
        high_threshold = float(max_prices.quantile(self.upper_quantile))

        buy_market, sell_market = _best_markets(prices)
        buy_price = float(prices[buy_market])
        sell_price = float(prices[sell_market])

        ma_buy = _latest_moving_average(history, buy_market)
        ma_sell = _latest_moving_average(history, sell_market)
        charge_size = _moving_average_size_multiplier(
            buy_price, ma_buy, charge=True
        )
        discharge_size = _moving_average_size_multiplier(
            sell_price, ma_sell, charge=False
        )

        eta_c = fleet.charge_efficiency
        eta_d = fleet.discharge_efficiency
        eta2 = eta_c * eta_d
        cycle_cost = fleet.cycle_cost_per_mwh_throughput

        # A wide current cross-market spread is itself a pure arbitrage
        # opportunity: one battery can buy in the cheap market while another
        # sells into the expensive market.  The spread must cover a full
        # round-trip efficiency loss and the cycle wear on both legs.
        round_trip_cost = cycle_cost * (1.0 + eta2)
        cross_market_edge = sell_price * eta2 - buy_price - round_trip_cost
        spread_signal = cross_market_edge > self.min_profit

        charge_candidates = []
        if buy_price <= low_threshold or spread_signal:
            for index, battery in enumerate(fleet.batteries):
                energy = (
                    battery.max_charge_grid_energy(dt_hours) * charge_size
                )
                if energy <= EPS:
                    continue
                # If the current cross-market spread is the trigger, the exit
                # is the price available in the other market right now.
                expected_exit = max(high_threshold, sell_price)
                charge_profit = (
                    expected_exit * eta2 * energy
                    - buy_price * energy
                    - round_trip_cost * energy
                    - 2.0 * battery.action_cost
                )
                if charge_profit > self.min_profit:
                    charge_candidates.append(
                        (index, energy, charge_profit)
                    )

        discharge_candidates = []
        if sell_price >= high_threshold or spread_signal:
            for index, battery in enumerate(fleet.batteries):
                energy = (
                    _profitable_discharge_energy(
                        battery, sell_price, dt_hours, self.min_profit
                    )
                    * discharge_size
                )
                if energy <= EPS:
                    continue
                removed_energy = energy / eta_d
                wear = battery.wear_cost_for_discharge(energy)
                book_cost = (
                    battery.average_stored_cost_per_mwh * removed_energy
                )
                profit = energy * sell_price - wear - book_cost
                discharge_candidates.append((index, energy, profit))

        # Both signals: try to pair one charging battery with one discharging
        # battery.  A single battery is never asked to do both.
        if charge_candidates and discharge_candidates:
            best_pair = None
            for charge_index, charge_energy, charge_profit in charge_candidates:
                for discharge_index, discharge_energy, discharge_profit in discharge_candidates:
                    if charge_index == discharge_index:
                        continue
                    score = max(0.0, charge_profit) + discharge_profit
                    if best_pair is None or score > best_pair[0]:
                        best_pair = (
                            score,
                            charge_index,
                            charge_energy,
                            charge_profit,
                            discharge_index,
                            discharge_energy,
                            discharge_profit,
                        )

            if best_pair is not None and best_pair[0] > self.min_profit:
                (
                    score,
                    charge_index,
                    charge_energy,
                    charge_profit,
                    discharge_index,
                    discharge_energy,
                    discharge_profit,
                ) = best_pair
                return Dispatch(
                    actions=[
                        BatteryAction(
                            charge_index,
                            CHARGE,
                            charge_energy,
                            buy_market,
                            buy_price,
                        ),
                        BatteryAction(
                            discharge_index,
                            DISCHARGE,
                            discharge_energy,
                            sell_market,
                            sell_price,
                        ),
                    ],
                    expected_profit=score,
                    reason=(
                        f"simultaneous: charge battery {charge_index} at "
                        f"{buy_price:.2f}, discharge battery {discharge_index} "
                        f"at {sell_price:.2f}"
                    ),
                )

            # If no cross-battery pair is profitable, take the better single
            # direction below.

        # Charge-only case.
        if charge_candidates:
            total_profit = sum(
                max(0.0, profit) for _, _, profit in charge_candidates
            )
            if total_profit > self.min_profit:
                return Dispatch(
                    actions=[
                        BatteryAction(
                            index, CHARGE, energy, buy_market, buy_price
                        )
                        for index, energy, _ in charge_candidates
                    ],
                    expected_profit=total_profit,
                    reason=(
                        f"charge {len(charge_candidates)} battery(ies) at "
                        f"{buy_price:.2f} <= {low_threshold:.2f}"
                    ),
                )

        # Discharge-only case.
        if discharge_candidates:
            total_profit = sum(profit for _, _, profit in discharge_candidates)
            if total_profit > self.min_profit:
                return Dispatch(
                    actions=[
                        BatteryAction(
                            index,
                            DISCHARGE,
                            energy,
                            sell_market,
                            sell_price,
                        )
                        for index, energy, _ in discharge_candidates
                    ],
                    expected_profit=total_profit,
                    reason=(
                        f"discharge {len(discharge_candidates)} battery(ies) at "
                        f"{sell_price:.2f} >= {high_threshold:.2f}"
                    ),
                )

        return Dispatch.no_action("no profitable arbitrage")


@dataclass
class ForecastOptimiser:
    """Choose the best per-battery pattern under the ARIMA forecast.

    For two batteries there are only nine possible patterns: each battery can
    charge, discharge or stay idle.  This explicitly allows one battery to
    charge while the other discharges.
    """

    min_profit: float = 0.0

    def decide(
        self,
        timestamp: pd.Timestamp,
        prices: Dict[str, float],
        history: pd.DataFrame,
        fleet: BatteryFleet,
        dt_hours: float,
        forecast_path: Optional[np.ndarray] = None,
    ) -> Dispatch:
        if forecast_path is None or not np.isfinite(forecast_path).any():
            return Dispatch.no_action("no forecast available")

        buy_market, sell_market = _best_markets(prices)
        buy_price = float(prices[buy_market])
        sell_price = float(prices[sell_market])

        expected_sell = float(np.nanmax(forecast_path))
        expected_buy = float(np.nanmin(forecast_path))

        # The 7-day average sizes the action.  Buying below its average (or
        # selling above it) justifies full power; the less favourable side of
        # the average is traded at half power.  Before seven days of history
        # exist the multiplier is 1.0.
        ma_buy = _latest_moving_average(history, buy_market)
        ma_sell = _latest_moving_average(history, sell_market)
        charge_size = _moving_average_size_multiplier(
            buy_price, ma_buy, charge=True
        )
        discharge_size = _moving_average_size_multiplier(
            sell_price, ma_sell, charge=False
        )

        eta_c = fleet.charge_efficiency
        eta_d = fleet.discharge_efficiency
        eta2 = eta_c * eta_d
        cycle_cost = fleet.cycle_cost_per_mwh_throughput

        # Per MWh of grid-side energy.
        charge_edge = (
            expected_sell * eta2
            - buy_price
            - cycle_cost * (1.0 + eta2)
        )
        discharge_edge = (
            sell_price
            - expected_buy / eta2
            - cycle_cost * (1.0 + 1.0 / eta2)
        )

        best_dispatch: Optional[Dispatch] = None
        best_profit = self.min_profit

        for pattern in product(
            (NO_ACTION, CHARGE, DISCHARGE), repeat=len(fleet.batteries)
        ):
            actions: List[BatteryAction] = []
            profit = 0.0
            feasible = True

            for index, direction in enumerate(pattern):
                battery = fleet.batteries[index]
                if direction == CHARGE:
                    energy = (
                        battery.max_charge_grid_energy(dt_hours) * charge_size
                    )
                    if energy <= EPS:
                        feasible = False
                        break
                    profit += charge_edge * energy - battery.action_cost
                    actions.append(
                        BatteryAction(
                            index, CHARGE, energy, buy_market, buy_price
                        )
                    )
                elif direction == DISCHARGE:
                    energy = (
                        battery.max_discharge_grid_energy(dt_hours)
                        * discharge_size
                    )
                    if energy <= EPS:
                        feasible = False
                        break
                    profit += (
                        discharge_edge * energy - battery.action_cost
                    )
                    actions.append(
                        BatteryAction(
                            index, DISCHARGE, energy, sell_market, sell_price
                        )
                    )

            if feasible and profit > best_profit:
                best_profit = profit
                best_dispatch = Dispatch(
                    actions=actions,
                    expected_profit=profit,
                    reason=(
                        f"forecast pattern {pattern}; expected profit "
                        f"GBP {profit:.2f}"
                    ),
                )

        return best_dispatch or Dispatch.no_action(
            "forecast does not support a profitable action"
        )


@dataclass
class CombinedStrategy:
    """Use both signals and pick the higher expected-profit dispatch."""

    arbitrage: ArbitrageStrategy
    optimiser: ForecastOptimiser
    min_profit: float = 0.0

    def decide(
        self,
        timestamp: pd.Timestamp,
        prices: Dict[str, float],
        history: pd.DataFrame,
        fleet: BatteryFleet,
        dt_hours: float,
        forecast_path: Optional[np.ndarray] = None,
    ) -> Dispatch:
        arb_dispatch = self.arbitrage.decide(
            timestamp, prices, history, fleet, dt_hours, forecast_path
        )
        if forecast_path is None:
            return arb_dispatch

        opt_dispatch = self.optimiser.decide(
            timestamp, prices, history, fleet, dt_hours, forecast_path
        )

        if opt_dispatch.action == NO_ACTION:
            if arb_dispatch.action == NO_ACTION:
                return arb_dispatch
            score = self._score(arb_dispatch, forecast_path, fleet)
            if score > self.min_profit:
                arb_dispatch.expected_profit = score
                return arb_dispatch
            return Dispatch.no_action("no signal is profitable under forecast")

        if arb_dispatch.action == NO_ACTION:
            return opt_dispatch

        if arb_dispatch.action == opt_dispatch.action:
            return opt_dispatch

        arb_score = self._score(arb_dispatch, forecast_path, fleet)
        if arb_score > self.min_profit and arb_score > opt_dispatch.expected_profit:
            arb_dispatch.expected_profit = arb_score
            return arb_dispatch
        return opt_dispatch

    @staticmethod
    def _score(
        dispatch: Dispatch,
        forecast_path: np.ndarray,
        fleet: BatteryFleet,
    ) -> float:
        expected_sell = float(np.nanmax(forecast_path))
        expected_buy = float(np.nanmin(forecast_path))
        eta_c = fleet.charge_efficiency
        eta_d = fleet.discharge_efficiency
        eta2 = eta_c * eta_d
        cycle_cost = fleet.cycle_cost_per_mwh_throughput

        charge_edge = (
            expected_sell * eta2
            - 0.0
            - cycle_cost * (1.0 + eta2)
        )
        discharge_edge = (
            0.0
            - expected_buy / eta2
            - cycle_cost * (1.0 + 1.0 / eta2)
        )

        score = 0.0
        for action in dispatch.actions:
            if action.action == CHARGE:
                score += (
                    (charge_edge - action.price) * action.energy_mwh
                    - fleet.action_cost
                )
            elif action.action == DISCHARGE:
                score += (
                    (action.price + discharge_edge) * action.energy_mwh
                    - fleet.action_cost
                )
        return score


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    actions: pd.DataFrame
    summary: Dict[str, float]


class Backtester:
    """Run a strategy over a price history and record every decision."""

    def __init__(self, fleet: BatteryFleet, strategy: Strategy) -> None:
        self.fleet = fleet
        self.strategy = strategy

    def run(
        self,
        data: pd.DataFrame,
        forecast_paths: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
        initial_soc_mwh: float = 0.0,
        history_data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Backtest ``data``.

        ``history_data`` can contain the full price history (including the
        training period) when ``data`` is only the out-of-sample slice.  The
        strategy still only sees rows up to the current timestamp.
        """
        self.fleet.reset(initial_soc_mwh)
        history_source = data if history_data is None else history_data
        rows = []
        timestamps = data.index

        for i, (timestamp, row) in enumerate(data.iterrows()):
            if i + 1 < len(timestamps):
                dt_hours = (
                    timestamps[i + 1] - timestamp
                ).total_seconds() / 3600.0
            else:
                dt_hours = 0.5

            prices = {
                MARKET_1: float(row["market_1_price"]),
                MARKET_2: float(row["market_2_price"]),
            }
            history = history_source.loc[:timestamp]
            forecast_path = (
                forecast_paths.get(timestamp) if forecast_paths else None
            )

            dispatch = self.strategy.decide(
                timestamp=timestamp,
                prices=prices,
                history=history,
                fleet=self.fleet,
                dt_hours=dt_hours,
                forecast_path=forecast_path,
            )
            execution = self.fleet.execute(dispatch, dt_hours)

            executed_action = (
                dispatch.action
                if execution.charge_mwh + execution.discharge_mwh > EPS
                else NO_ACTION
            )
            actual_by_battery = {
                action.battery_index: action
                for action in execution.actual_actions
            }
            charge_actions = [
                action
                for action in execution.actual_actions
                if action.action == CHARGE
            ]
            discharge_actions = [
                action
                for action in execution.actual_actions
                if action.action == DISCHARGE
            ]

            rows.append(
                {
                    "time": timestamp,
                    "action": executed_action,
                    "charge_mwh": execution.charge_mwh,
                    "discharge_mwh": execution.discharge_mwh,
                    "energy_mwh": (
                        execution.charge_mwh + execution.discharge_mwh
                    ),
                    "power_mw": (
                        execution.charge_mwh + execution.discharge_mwh
                    )
                    / dt_hours
                    if dt_hours > 0
                    else 0.0,
                    "charge_market": (
                        charge_actions[0].market if charge_actions else None
                    ),
                    "discharge_market": (
                        discharge_actions[0].market
                        if discharge_actions
                        else None
                    ),
                    "charge_price": (
                        charge_actions[0].price if charge_actions else np.nan
                    ),
                    "discharge_price": (
                        discharge_actions[0].price
                        if discharge_actions
                        else np.nan
                    ),
                    "market_1_ma_7d": float(
                        row.get("market_1_ma_7d", np.nan)
                    ),
                    "market_2_ma_7d": float(
                        row.get("market_2_ma_7d", np.nan)
                    ),
                    "cash_flow": execution.cash_flow,
                    "energy_cost": execution.energy_cost,
                    "revenue": execution.revenue,
                    "wear_cost": execution.wear_cost,
                    "soc_mwh": execution.soc_mwh,
                    "soc_pct": 100.0
                    * execution.soc_mwh
                    / self.fleet.capacity_mwh
                    if self.fleet.capacity_mwh > 0
                    else 0.0,
                    "equivalent_cycles": execution.equivalent_cycles,
                    "battery_0_action": actual_by_battery.get(
                        0, BatteryAction(0)
                    ).action,
                    "battery_0_energy_mwh": actual_by_battery.get(
                        0, BatteryAction(0)
                    ).energy_mwh,
                    "battery_0_soc_mwh": self.fleet.batteries[0].soc_mwh,
                    "battery_1_action": actual_by_battery.get(
                        1, BatteryAction(1)
                    ).action,
                    "battery_1_energy_mwh": actual_by_battery.get(
                        1, BatteryAction(1)
                    ).energy_mwh,
                    "battery_1_soc_mwh": self.fleet.batteries[1].soc_mwh,
                    "expected_profit": dispatch.expected_profit,
                    "reason": dispatch.reason,
                }
            )

        actions = pd.DataFrame(rows)
        summary = self._summarise(actions)
        if summary:
            last_prices = {
                MARKET_1: float(data["market_1_price"].iloc[-1]),
                MARKET_2: float(data["market_2_price"].iloc[-1]),
            }
            terminal_sell_price = max(last_prices.values())
            terminal_value = 0.0
            for battery in self.fleet.batteries:
                grid_energy = battery.soc_mwh * battery.discharge_efficiency
                if grid_energy <= EPS:
                    continue
                # Energy still stored has not yet paid its cycle wear in
                # "cycle" mode, so deduct the deferred wear when marking the
                # inventory to market.
                wear = battery.wear_cost_for_discharge(grid_energy)
                terminal_value += grid_energy * terminal_sell_price - wear
            summary["terminal_storage_value"] = terminal_value
            summary["economic_profit"] = (
                summary["total_profit"] + terminal_value
            )
        return BacktestResult(actions=actions, summary=summary)

    def _summarise(self, actions: pd.DataFrame) -> Dict[str, float]:
        if actions.empty:
            return {}

        charging = actions["charge_mwh"] > EPS
        discharging = actions["discharge_mwh"] > EPS
        energy_charged = float(actions["charge_mwh"].sum())
        energy_discharged = float(actions["discharge_mwh"].sum())
        revenue = float(actions["revenue"].sum())
        energy_cost = float(actions["energy_cost"].sum())
        wear_cost = float(actions["wear_cost"].sum())
        total_profit = float(actions["cash_flow"].sum())

        n_battery_charge = int(
            (actions["battery_0_action"] == CHARGE).sum()
            + (actions["battery_1_action"] == CHARGE).sum()
        )
        n_battery_discharge = int(
            (actions["battery_0_action"] == DISCHARGE).sum()
            + (actions["battery_1_action"] == DISCHARGE).sum()
        )

        return {
            "total_profit": total_profit,
            "gross_revenue": revenue,
            "energy_cost": energy_cost,
            "wear_cost": wear_cost,
            "energy_charged_mwh": energy_charged,
            "energy_discharged_mwh": energy_discharged,
            "n_actions": n_battery_charge + n_battery_discharge,
            "n_charge": n_battery_charge,
            "n_discharge": n_battery_discharge,
            "n_charge_events": int(charging.sum()),
            "n_discharge_events": int(discharging.sum()),
            "n_simultaneous_events": int((charging & discharging).sum()),
            "average_buy_price": energy_cost / energy_charged
            if energy_charged > 0
            else 0.0,
            "average_sell_price": revenue / energy_discharged
            if energy_discharged > 0
            else 0.0,
            "final_soc_mwh": float(actions["soc_mwh"].iloc[-1]),
            "final_soc_pct": float(actions["soc_pct"].iloc[-1]),
            "equivalent_cycles": float(
                actions["equivalent_cycles"].iloc[-1]
            ),
            "final_capacity_mwh": self.fleet.capacity_mwh,
            "profit_per_cycle": total_profit
            / float(actions["equivalent_cycles"].iloc[-1])
            if actions["equivalent_cycles"].iloc[-1] > EPS
            else 0.0,
        }
