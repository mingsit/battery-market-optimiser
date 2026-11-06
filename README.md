# battery-market-optimiser
Optimise charging or discharging battery using market price to earn profit
In this example, I try to earn profit by mainly 2 strategies:
(1) By pure Arbitrage (2) By buying low and selling high according to moving average
## Assumption:
1. Actions of the battery have no impact on the market prices.
2. When the battery chooses to use some portion of its capacity to charge or discharge from one of the markets, it must commit that capacity for the entire time interval of that market.
3. The battery cannot charge and discharge at the same time.
4. Time cost is ignored for simplicity.
5. Having 2 half-filled battery initially.
6. See Attachment.csv
## Dependency
Only pandas.
## Run code
(1) python main.py
To run the whole script.
(2) from main import 
## Future ideas
Predict price of the markets with ARIMA model or Regression model, not just based on moving average.
I will upload a vibe coding example for a proof of concept.