"""ARIMA price forecasting for the two wholesale electricity markets.

The model deliberately uses only calendar information as exogenous inputs:
  * time-of-day (hour dummies)
  * day-of-week (sin/cos)
  * is_holiday (weekend or configured bank holiday)

The ARIMA state is estimated on the first four weeks of data and then updated
with each new observation (``refit=False``).  This keeps the one-step forecast
fresh without re-estimating coefficients on every half-hour bar.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "statsmodels is required for the ARIMA model. "
        "Install it with: pip install statsmodels"
    ) from exc

warnings.simplefilter("ignore", ConvergenceWarning)

MARKETS = ("market_1_price", "market_2_price")
DEFAULT_HOLIDAYS = {date(2018, 1, 1)}


def _holiday_flags(index: pd.DatetimeIndex, holidays: Iterable[date]) -> np.ndarray:
    holidays = set(holidays)
    return np.array(
        [(ts.weekday() >= 5) or (ts.date() in holidays) for ts in index],
        dtype=float,
    )


def build_time_features(
    index: pd.DatetimeIndex,
    holidays: Iterable[date] = DEFAULT_HOLIDAYS,
) -> pd.DataFrame:
    """Create the only exogenous features allowed by the brief.

    Time of day is represented by hour dummies, which fit the sharp intraday
    shape better than a single sine/cosine pair.  Day of week is smooth, so it
    is represented by sine/cosine terms.  The remaining feature is the
    weekend/bank-holiday flag.

    Parameters
    ----------
    index:
        Timestamps to create features for.
    holidays:
        Dates that should be treated as holidays in addition to weekends.
    """
    index = pd.DatetimeIndex(index)
    day_of_week = index.dayofweek

    # Fixed category set keeps the exog columns identical for training and
    # forecasting, even when a short future window does not contain every hour.
    hour_categories = pd.Categorical(index.hour, categories=range(24))
    hour_dummies = pd.get_dummies(
        hour_categories, prefix="hour", dtype=float
    )
    # Drop one level: the ARIMA model already includes a constant term.
    hour_dummies = hour_dummies.iloc[:, 1:]
    hour_dummies.index = index

    features = pd.DataFrame(
        {
            "dow_sin": np.sin(2.0 * np.pi * day_of_week / 7.0),
            "dow_cos": np.cos(2.0 * np.pi * day_of_week / 7.0),
            "is_holiday": _holiday_flags(index, holidays),
        },
        index=index,
    )
    return pd.concat([features, hour_dummies], axis=1)


@dataclass
class ForecastResult:
    """Container for the rolling forecasts and their accuracy metrics."""

    paths: Dict[pd.Timestamp, np.ndarray]
    one_step: pd.DataFrame
    metrics: pd.DataFrame
    orders: Dict[str, Tuple[int, int, int]]
    horizon: int


class ARIMAPriceForecaster:
    """Small, repeatable ARIMAX wrapper around statsmodels.

    Parameters
    ----------
    horizon:
        Number of future half-hour steps stored for the dispatch optimiser.
    max_p, max_q:
        Maximum AR and MA orders considered during order selection.
    d:
        Differencing order.  Electricity prices are non-stationary in the
        sample, so ``d=1`` is a sensible default.
    refit_every_days:
        ``None`` keeps the coefficients estimated on the first four weeks and
        only updates the state with new observations.  Set to an integer to
        re-estimate the coefficients periodically.
    """

    def __init__(
        self,
        horizon: int = 8,
        max_p: int = 2,
        max_q: int = 2,
        d: int = 1,
        holidays: Iterable[date] = DEFAULT_HOLIDAYS,
        refit_every_days: Optional[int] = None,
        freq: str = "30min",
        maxiter: int = 100,
    ) -> None:
        self.horizon = horizon
        self.max_p = max_p
        self.max_q = max_q
        self.d = d
        self.holidays = holidays
        self.refit_every_days = refit_every_days
        self.freq = freq
        self.maxiter = maxiter

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------
    def _fit_one(
        self,
        y: np.ndarray,
        exog: np.ndarray,
        order: Tuple[int, int, int],
    ):
        model = SARIMAX(
            y,
            exog=exog,
            order=order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=self.maxiter)

    def select_order(
        self,
        y: np.ndarray,
        exog: np.ndarray,
    ) -> Tuple[int, int, int]:
        """Choose a small ARIMA order by AIC on the training sample."""
        best_order = (1, self.d, 1)
        best_aic = np.inf

        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                order = (p, self.d, q)
                try:
                    result = self._fit_one(y, exog, order)
                    aic = float(result.aic)
                except Exception:
                    continue
                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best_order = order

        return best_order

    def fit(
        self,
        history: pd.DataFrame,
        exog: pd.DataFrame,
    ) -> Tuple[Dict[str, object], Dict[str, Tuple[int, int, int]]]:
        """Fit one ARIMA model per market on the supplied history."""
        results: Dict[str, object] = {}
        orders: Dict[str, Tuple[int, int, int]] = {}

        for market in MARKETS:
            y = history[market].to_numpy(dtype=float)
            x = exog.loc[history.index].to_numpy(dtype=float)
            order = self.select_order(y, x)
            results[market] = self._fit_one(y, x, order)
            orders[market] = order

        return results, orders

    # ------------------------------------------------------------------
    # Rolling forecasts
    # ------------------------------------------------------------------
    def rolling_forecast(
        self,
        data: pd.DataFrame,
        train_end: pd.Timestamp,
        horizon: Optional[int] = None,
        progress: bool = True,
    ) -> ForecastResult:
        """Forecast every test timestamp using only information known then.

        ``data`` must contain the columns in :data:`MARKETS` and be indexed by
        a regular half-hourly ``DatetimeIndex``.  The first forecast origin is
        ``train_end``; observations before ``train_end`` are used for fitting.
        """
        horizon = horizon or self.horizon
        exog = build_time_features(data.index, self.holidays)

        train = data.loc[data.index < train_end]
        if len(train) < 50:
            raise ValueError("Not enough history to fit the ARIMA model.")

        test = data.loc[data.index >= train_end]
        if test.empty:
            raise ValueError("No test observations after train_end.")

        results, orders = self.fit(train, exog)
        last_price = {market: float(train[market].iloc[-1]) for market in MARKETS}
        paths: Dict[pd.Timestamp, np.ndarray] = {}
        one_step_rows = []
        steps_per_day = max(
            1, int(round(pd.Timedelta(days=1) / pd.Timedelta(self.freq)))
        )

        for i, (timestamp, row) in enumerate(test.iterrows()):
            # Optional periodic re-estimation.  The default is to keep the
            # coefficients fixed and only update the state below.
            if (
                self.refit_every_days
                and i > 0
                and i % (self.refit_every_days * steps_per_day) == 0
            ):
                history = data.loc[data.index < timestamp]
                try:
                    results, orders = self.fit(history, exog.loc[history.index])
                except Exception:
                    pass

            future_index = pd.date_range(
                timestamp, periods=horizon, freq=self.freq
            )
            future_exog = build_time_features(
                future_index, self.holidays
            ).to_numpy(dtype=float)

            path = np.full((horizon, len(MARKETS)), np.nan, dtype=float)
            for j, market in enumerate(MARKETS):
                try:
                    forecast = results[market].get_forecast(
                        steps=horizon, exog=future_exog
                    ).predicted_mean
                    forecast = np.asarray(forecast, dtype=float)
                    forecast = np.maximum(forecast, 0.0)
                    path[: len(forecast), j] = forecast
                except Exception:
                    path[:, j] = last_price[market]

                # Update the state with the observation we have actually seen.
                try:
                    results[market] = results[market].append(
                        endog=[float(row[market])],
                        exog=exog.loc[[timestamp]].to_numpy(dtype=float),
                        refit=False,
                    )
                except Exception:
                    # Fall back to a full re-fit if the state update fails.
                    try:
                        history = data.loc[data.index <= timestamp]
                        results[market] = self._fit_one(
                            history[market].to_numpy(dtype=float),
                            exog.loc[history.index].to_numpy(dtype=float),
                            orders[market],
                        )
                    except Exception:
                        pass

            # Fill any isolated NaNs with the latest observed price.
            for j, market in enumerate(MARKETS):
                column = path[:, j]
                if np.isnan(column).any():
                    column[np.isnan(column)] = last_price[market]
                path[:, j] = column

            paths[timestamp] = path
            one_step_rows.append(
                {
                    "time": timestamp,
                    "market_1_forecast": path[0, 0],
                    "market_2_forecast": path[0, 1],
                    "market_1_actual": float(row["market_1_price"]),
                    "market_2_actual": float(row["market_2_price"]),
                }
            )
            last_price = {
                market: float(row[market]) for market in MARKETS
            }

            if progress and i > 0 and i % 200 == 0:
                print(f"  ARIMA forecasts: {i}/{len(test)}")

        one_step = pd.DataFrame(one_step_rows).set_index("time")
        metrics = self._evaluate(test, one_step)
        return ForecastResult(
            paths=paths,
            one_step=one_step,
            metrics=metrics,
            orders=orders,
            horizon=horizon,
        )

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate(test: pd.DataFrame, one_step: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for market in MARKETS:
            forecast_column = market.replace("_price", "") + "_forecast"
            actual = test[market].to_numpy(dtype=float)
            forecast = one_step[forecast_column].to_numpy(dtype=float)
            mask = np.isfinite(actual) & np.isfinite(forecast)
            error = actual[mask] - forecast[mask]
            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(np.mean(error**2)))

            naive = np.full_like(actual, np.nan, dtype=float)
            naive[1:] = actual[:-1]
            naive_mask = np.isfinite(actual) & np.isfinite(naive)
            naive_mae = float(
                np.mean(np.abs(actual[naive_mask] - naive[naive_mask]))
            )

            nonzero = np.abs(actual[mask]) > 1e-9
            mape = (
                float(
                    np.mean(
                        np.abs(error[nonzero] / actual[mask][nonzero])
                    )
                    * 100.0
                )
                if nonzero.any()
                else np.nan
            )

            rows.append(
                {
                    "market": market,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE_%": mape,
                    "naive_MAE": naive_mae,
                    "skill_vs_naive": 1.0 - mae / naive_mae
                    if naive_mae > 0
                    else np.nan,
                }
            )

        return pd.DataFrame(rows).set_index("market")
