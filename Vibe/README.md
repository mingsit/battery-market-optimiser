# Battery Arbitrage Dashboard

This folder contains the battery-arbitrage model and a local Django dashboard
for exploring it.

## Run the dashboard

```powershell
cd C:\Users\85255\Documents\Python\Vibe
python manage.py runserver 8000
```

Then open <http://localhost:8000/>.

The first page load uses `webapp/cache/forecasts.pkl`, which is already
included. If you change the model or delete the cache, rebuild it with:

```powershell
python manage.py build_forecast_cache
```

## What the page shows

- A datetime picker for any half-hour bar in the sample.
- Market prices from the start of the sample up to the selected bar.
- Each market's 7-day (168-hour) moving average once a full week exists.
- The ARIMA forecast path when the selected bar is inside the out-of-sample
  period.
- Every executed battery action before the selected bar, paginated.
- Realised profit, stored-energy value and economic profit.
- Current fleet and per-battery state of charge, cycles and average cost.
- The combined strategy's suggested action for the selected bar, together
  with its reason and expected profit.

The dashboard runs the same modules as `main.py`: pure arbitrage in the first
four weeks, then the combined ARIMA + arbitrage strategy. It simulates
continuously from the start of the sample, so the battery state shown at a
given time is the state that the strategy would actually have reached.

## Re-run the command-line backtest

```powershell
python main.py
```

Results are written to `results/`.
