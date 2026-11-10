import pandas
import os
from datetime import datetime, date, timedelta
from ortools.linear_solver import pywraplp


class BatteryMarketOptimiser():

    def __init__(self):    
        # Read Attachment 1 as config
        param_df = pandas.read_csv('Attachment.csv')
        self.param = {}
        for name, value in zip(param_df['Name'], param_df['Values']):
            self.param[name] = value
        self.price_history = []  # As we don't know the future, everything should only base on history price
        # list of dict: {time, is_holiday, market_1_price, market_2_price}
        self.action_history = []
        self.moving_average_days = 7
        self.market_ma = {1: 50, 2: 50}
        self.ma_profit_margin = 0.15  # Minimum profit margin for charging / discharging compared to moving average
        self.current_battery = [2, 2]  # assume we have 2 half-full battery initially, so that we can go for arbitrage profit if any
        self.battery_cycle = [0.25, 0.25]  # Cycle lifetime used (0.25 initially)

        # For OR Tools
        self.num_battery = 1

    def calculate_moving_average(self, days=7):
        sum_price = 0
        count = 0
        for market in [1, 2]:
            for item in reversed(self.price_history):
                if item['time'] > self.current_time - timedelta(days=days):
                    sum_price += item[f'market_{market}_price']
                    count += 1
                else:
                    break
            self.market_ma[market] = sum_price / count

    def decide_action(self):
        # Decide to charge / discharge based on arbitrage, moving average or ARIMA prediction
        use_ma = False
        if self.current_time - timedelta(days=self.moving_average_days) > self.start_time:
            self.calculate_moving_average(days=self.moving_average_days)
            use_ma = True
    
        # Calculate charging cost, If cost > expected gain, return no actions
        cycle_cost = self.param['Capex'] / self.param['Lifetime (2)']  # battery action cost by whole cycle, ignore time cost.
        charge_eff = (1 - self.param['Battery charging efficiency'])
        discharge_eff = (1 - self.param['Battery discharging efficiency'])
        total_efficiency = charge_eff * discharge_eff
        battery_size = self.param['Max storage volume']
        charge_rate = self.param['Max charging rate']
        discharge_rate = self.param['Max discharging rate']
        
        # Charge efficiency
        # Discharge efficiency

        # Calculate Arbitrage gain
        actions = []  # list of dict: {time, action:charge/discharge, battery=int, market=int, cash=float}
        
        market_1_price = self.price_history[-1]['market_1_price']
        market_2_price = self.price_history[-1]['market_2_price']
        if not self.market_1_freeze and not self.market_2_freeze:
            # Can do actions in both market, find arbitrage opportunity
            # Decide how much rate to charge and discharge so the amoumt is equal
            if market_1_price < market_2_price:
                max_charge_amt = charge_rate * self.market_1_interval
                max_discharge_amt = discharge_rate * self.market_2_interval
                target_amt = min(max_charge_amt, max_discharge_amt)
                charge_market = 1
                discharge_market = 2
            else:
                max_charge_amt = charge_rate * self.market_2_interval
                max_discharge_amt = discharge_rate * self.market_1_interval
                target_amt = min(max_charge_amt, max_discharge_amt)
                charge_market = 2
                discharge_market = 1
            if self.current_battery[0] > target_amt and self.current_battery[1] > target_amt and \
                self.current_battery[0] < battery_size - target_amt and self.current_battery[1] < battery_size - target_amt:
                # TODO improvement: should aggerate to charge / discharge until maximum or 0
                
                # Find actual revenue and cost by taking cycle cost and enery loss into account
                cycle_cost = cycle_cost / (battery_size * 2) * (target_amt * 2)
                # shorter market use full rate. longer market use proportional rate
                charge_cost = min(market_1_price, market_2_price) * target_amt
                actual_revenue = max(market_1_price, market_2_price) * target_amt * total_efficiency
                net_profit = actual_revenue - charge_cost - cycle_cost
                if net_profit > 0:  # Can earn arbitrage profit, do action
                    print(f'Arbitrage opportunity found: Market Price {market_1_price} vs {market_2_price}')
                    print(f'Gross profit for this action: {actual_revenue-charge_cost}')
                    print(f'Net profit for this action: {net_profit}')
                    charge_battery = 1
                    discharge_battery = 0
                    if self.current_battery[0] < self.current_battery[1]:
                        charge_battery = 0
                        discharge_battery = 1
                    actions.append({'time': self.current_time, 'battery': charge_battery, 'market': charge_market, 
                                    'action': 'charge', 'cash': - market_1_price * target_amt})
                    actions.append({'time': self.current_time, 'battery': discharge_battery, 'market': discharge_market, 
                                    'action': 'charge', 'cash': market_2_price * target_amt * total_efficiency})
                    self.current_battery[charge_battery] += target_amt * charge_eff
                    self.current_battery[discharge_battery] -= target_amt * charge_eff  # only charge_eff matters
                    for i in [0, 1]:
                        self.battery_cycle[i] += target_amt / (battery_size * 2)
        
        if use_ma and not actions:  # buy low, sell high
            # Charge if actual_charge_cost < moving average
            # Discharge if actual_revenue < moving average

            if self.current_battery[0] < battery_size and self.current_battery[1] < battery_size:
                # Find whether this is a charge opportunity
                buy_unit = 0
                charge_battery = []
                for i in [0, 1]:
                    if self.current_battery[i] < battery_size:
                        buy_unit += 1
                        charge_battery.append(i)
                if not self.market_1_freeze and not self.market_2_freeze:
                    buy_price = min(market_1_price, market_2_price)
                    compare_price = max(self.market_ma[1], self.market_ma[2])
                    period = self.market_1_interval if market_1_price < market_2_price else self.market_2_interval
                    market = 1 if market_1_price < market_2_price else 2
                elif not self.market_1_freeze:
                    buy_price = market_1_price
                    compare_price = self.market_ma[1]
                    period = self.market_1_interval
                    market = 1
                elif not self.market_2_freeze:
                    buy_price = market_2_price
                    compare_price = self.market_ma[2]
                    period = self.market_2_interval
                    market = 2
                # Naively assuming we can discharge enerygy later at average price, buy if profit under average price
                # Always use full power in profit scenario
                # Calculate cost and profit for EACH battery
                cycle_cost = cycle_cost / (battery_size * 2) * (charge_rate) * period
                charge_cost = buy_price * charge_rate * period
                revenue = compare_price * charge_rate * period * total_efficiency  # use charge_rate here
                potential_profit = revenue - charge_cost - cycle_cost
                if potential_profit > self.ma_profit_margin * charge_cost:  # buy low
                    print(f'Charge at price {buy_price} in market {market}')
                    print(f'Cost for this action: {charge_cost * buy_unit}')
                    for i in charge_battery:
                        if self.current_battery[i] > battery_size - charge_rate * period * charge_eff:
                            # If battery hits maximum, just charge to maximum
                            buy_amount = (battery_size - self.current_battery[i]) / charge_eff
                            charge_cost = buy_price * buy_amount
                            self.current_battery[i] = battery_size
                            self.battery_cycle[i] += (battery_size - self.current_battery[i]) / (battery_size * 2)
                            print(f'Charged battery to full, only cost {charge_cost} instead')
                        else:
                            self.current_battery[i] += charge_rate * charge_eff * period
                            self.battery_cycle[i] += charge_rate * period / (battery_size * 2)
                        actions.append({'time': self.current_time, 'battery': i, 'market': market, 
                                        'action': 'charge', 'cash': - charge_cost})

            if self.current_battery[0] > 0 or self.current_battery[1] > 0:
                # Find whether this is a discharge opportunity
                sell_unit = 0
                discharge_battery = []
                for i in [0, 1]:
                    if self.current_battery[i] > 0:
                        sell_unit += 1
                        discharge_battery.append(i)
                if not self.market_1_freeze and not self.market_2_freeze:
                    sell_price = max(market_1_price, market_2_price)
                    compare_price = min(self.market_ma[1], self.market_ma[2])
                    period = self.market_1_interval if market_1_price > market_2_price else self.market_2_interval
                    market = 1 if market_1_price > market_2_price else 2
                elif not self.market_1_freeze:
                    sell_price = market_1_price
                    compare_price = self.market_ma[1]
                    period = self.market_1_interval
                    market = 1
                elif not self.market_2_freeze:
                    sell_price = market_2_price
                    compare_price = self.market_ma[2]
                    period = self.market_2_interval
                    market = 2
                # Naively assuming we can charge enerygy back at average price, sell if profit over average price
                # Always use full power in profit scenario
                # Calculate cost and profit for EACH battery
                cycle_cost = cycle_cost / (battery_size * 2) * (discharge_rate) * period
                charge_cost = (compare_price * discharge_rate * period) / charge_eff
                actual_revenue = sell_price * discharge_rate * period * discharge_eff
                potential_profit = actual_revenue - charge_cost - cycle_cost
                if potential_profit > self.ma_profit_margin * charge_cost:  # sell high
                    print(f'Discharge at price {sell_price} in market {market}')
                    print(f'Revenue for this action: {actual_revenue * sell_unit}')
                    for i in discharge_battery:
                        if self.current_battery[i] <  discharge_rate * period:
                            # If battery hits 0, just discharge to 0
                            actual_revenue = sell_price * self.current_battery[i] * discharge_eff
                            self.battery_cycle[i] += self.current_battery[i] / (battery_size * 2)
                            self.current_battery[i] = 0
                            print(f'Discharged battery to 0, only receive {actual_revenue} instead')
                        else:
                            self.current_battery[i] -= discharge_rate * period
                            self.battery_cycle[i] += discharge_rate * period / (battery_size * 2)
                        actions.append({'time': self.current_time, 'battery': i, 'market': market, 
                                        'action': 'discharge', 'cash': actual_revenue})

        # TODO in the future: predict the price in the upcoming hours after having enough data (4-week?)
        # Maybe ARIMA model for time series analysis, then use weekday for regression testing
        # Further suggestion: also create an API to MetOffice to get current temperature, use ANOVA test for significance
        # Then compare predicted price with arbitrage opportunity / moving average, which has better gain
        
        if actions:
            print('Battery level after action:', self.current_battery)
            print('====================================')
            self.action_history += actions

    def simulate(self):
        # Read both market's data into data frame
        df_1 = pandas.read_csv(os.path.join('data', 'market_1.csv'))
        df_2 = pandas.read_csv(os.path.join('data', 'market_2.csv'))

        holiday = {date(2018, 1, 1)}  # Only 1 holiday from 2018-01-01 to 2018-03-01
        # use date().weekday() to find weekday. Both bank holiday and weekday = 5 or 6 defined as holiday
        
        market_1_ind = 0
        market_2_ind = 0
        self.current_time = datetime.strptime(df_1['time'][0], '%Y-%m-%d %H:%M:%S')
        self.start_time = self.current_time
        diff1 = datetime.strptime(df_1['time'][1], '%Y-%m-%d %H:%M:%S') - datetime.strptime(df_1['time'][0], '%Y-%m-%d %H:%M:%S')
        diff2 = datetime.strptime(df_2['time'][1], '%Y-%m-%d %H:%M:%S') - datetime.strptime(df_2['time'][0], '%Y-%m-%d %H:%M:%S')
        self.market_1_interval = diff1.seconds / 3600  # half hour interval
        self.market_2_interval = diff2.seconds / 3600  # 1 hour interval
        self.market_1_freeze = True
        self.market_2_freeze = True

        while market_1_ind < len(df_1) and market_2_ind < len(df_2):
            # No hard code, make it suitiable for flexible time as long as the time is orderred
            market_1_time = datetime.strptime(df_1['time'][market_1_ind], '%Y-%m-%d %H:%M:%S')
            market_2_time = datetime.strptime(df_2['time'][market_2_ind], '%Y-%m-%d %H:%M:%S')
            self.current_time = min(market_1_time, market_2_time)

            if self.current_time >= market_1_time:
                # Read market 1, get price, then move to the next row
                market_1_price = df_1['price'][market_1_ind]
                market_1_ind += 1
                self.market_1_freeze = False
            else:
                self.market_1_freeze = True

            if self.current_time >= market_2_time:
                # Read market 2, get price, then move to the next row
                market_2_price = df_2['price'][market_2_ind]
                market_2_ind += 1
                self.market_2_freeze = False
            else:
                self.market_2_freeze = True

            current_price_dict = {'time': self.current_time, 
                                  'is_holiday': self.current_time.weekday() in [5, 6] or self.current_time.date() in holiday,
                                  'market_1_price': market_1_price,
                                  'market_2_price': market_2_price}
            self.price_history.append(current_price_dict)

            self.decide_action()

    def summary(self):
        # Output profits and cycle costs
        cashflow = 0
        for item in self.action_history:
            cashflow += item['cash']
        print('Total Gross Profit (£):', cashflow)
        # Cycle cost
        cycle_cost = self.param['Capex'] / self.param['Lifetime (2)'] * sum(self.battery_cycle)
        print('Cost of battery cycle (£):', cycle_cost)
        # Battery level
        print('Total Battery energy left (MWh):', sum(self.current_battery))
        # Net profit, initally 4 MWh at ~£50
        net_profit = cashflow - cycle_cost + sum(self.current_battery) * (self.market_ma[1] + self.market_ma[2]) / 2 - 200
        print('Net Profit (£):', net_profit)

    def use_or_tools(self):
        ''' Assume we know the future, treat it as a pure optimising problem
        For simplicity, I only use market 1 '''
        df = pandas.read_csv(os.path.join('data', 'market_1.csv'), parse_dates=['time'])

        diff1 = df['time'][1] - df['time'][0]
        self.market_1_interval = diff1.seconds / 3600  # half hour interval

        charge_amt = self.param['Max charging rate'] * self.market_1_interval
        discharge_amt = self.param['Max discharging rate'] * self.market_1_interval
        battery_size = self.param['Max storage volume']
        charge_eff = (1 - self.param['Battery charging efficiency'])
        discharge_eff = (1 - self.param['Battery discharging efficiency'])
        cycle_cost = self.param['Capex'] / self.param['Lifetime (2)']
        cycle_cost_per_mwh = cycle_cost / (battery_size * discharge_eff)

        prices = df.set_index('time')['price'].astype(float)
        n = len(prices)
        price = prices.to_numpy()
        solver = pywraplp.Solver.CreateSolver('CBC')
        if solver is None:
            raise RuntimeError('OR-Tools CBC solver is not available.')
    
        charge, discharge, soc, charging = {}, {}, {}, {}
        for battery in range(self.num_battery):
            for t in range(n):
                charge[battery, t] = solver.NumVar(0, charge_amt, '')
                discharge[battery, t] = solver.NumVar(0, discharge_amt, '')
                soc[battery, t] = solver.NumVar(0, battery_size, '')
                charging[battery, t] = solver.BoolVar('')
    
                # A single battery cannot charge and discharge in the same bar.
                solver.Add(
                    charge[battery, t] <= charge_amt * charging[battery, t]
                )
                solver.Add(
                    discharge[battery, t]
                    <= discharge_amt * (1 - charging[battery, t])
                )
    
                previous_soc = soc[battery, t - 1] if t > 0 else 0.0
                solver.Add(
                    soc[battery, t]
                    == previous_soc
                    + charge_eff * charge[battery, t]
                    - discharge[battery, t] / discharge_eff
                )
    
            # End empty so no free energy is left in the battery.
            solver.Add(soc[battery, n - 1] == 0)
    
        revenue = sum(
            price[t] * discharge[battery, t]
            for battery in range(self.num_battery)
            for t in range(n)
        )
        purchase = sum(
            price[t] * charge[battery, t]
            for battery in range(self.num_battery)
            for t in range(n)
        )
        wear = sum(
            cycle_cost_per_mwh * discharge[battery, t]
            for battery in range(self.num_battery)
            for t in range(n)
        )
        solver.Maximize(revenue - purchase - wear)
    
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(f'Optimisation did not finish optimally: {status}')
    
        rows = []
        for t, timestamp in enumerate(prices.index):
            rows.append(
                {
                    'time': timestamp,
                    'price': price[t],
                    'charge_mwh': sum(
                        charge[battery, t].solution_value()
                        for battery in range(self.num_battery)
                    ),
                    'discharge_mwh': sum(
                        discharge[battery, t].solution_value()
                        for battery in range(self.num_battery)
                    ),
                    'soc_mwh': sum(
                        soc[battery, t].solution_value()
                        for battery in range(self.num_battery)
                    ),
                }
            )
    
        summary = pandas.DataFrame(rows).set_index('time')
        summary['profit'] = (
            summary['price'] * (summary['discharge_mwh'] - summary['charge_mwh'])
            - cycle_cost_per_mwh * summary['discharge_mwh']
        )

        revenue = (summary['price'] * summary['discharge_mwh']).sum()
        purchase = (summary['price'] * summary['charge_mwh']).sum()
        wear = cycle_cost_per_mwh * summary['discharge_mwh'].sum()
        discharged = summary['discharge_mwh'].sum()
        cycles = (discharged / discharge_eff) / battery_size

        print(f'For perfect buy/sell scenario (battery={self.num_battery}) optimised in market 1:')
        print(f'Revenue (£): {revenue:,.2f}')
        print(f'Energy purchase cost (£): {purchase:,.2f}')
        print(f'Cost of battery cycle (£): {wear:,.2f}')
        print(f'Net profit (£): {summary["profit"].sum():,.2f}')
        print(f'Equivalent full cycles: {cycles:,.2f}')

    
if __name__ == '__main__':
    bmo = BatteryMarketOptimiser()
    #bmo.simulate()
    #bmo.summary()
    bmo.use_or_tools()
