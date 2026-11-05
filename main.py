import pandas
import os
from datetime import datetime, date, timedelta
from copy import deepcopy


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
        self.market_ma = {1: 50, 2: 50}
        self.current_battery = (0, 0)  # assume we have 2 battery initially, so that we can go for arbitrage profit without prediction

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

    def decide_action(self) -> dict:
        # Decide to charge / discharge based on arbitrage, moving average or ARIMA prediction
        use_ma = False
        moving_average_days = 7
        if self.current_time - timedelta(days=moving_average_days) > self.start_time:
            self.calculate_moving_average(days=moving_average_days)
            use_ma = True
    
        # Calculate charging cost, If cost > expected gain, return no actions
        cycle_cost = self.param['Capex'] / self.param['Lifetime (2)']  # battery action cost by whole cycle, ignore time cost.
        total_efficiency = (1 - self.param['Battery charging efficiency']) * (1 - self.param['Battery discharging efficiency'])
        
        # Charge efficiency
        # Discharge efficiency

        # Calculate Arbitrage gain
        if use_ma:  # compare arbitrage gain with moving average
            pass

        # TODO in the future: predict the price in the upcoming hours after having enough data (4-week?)
        # Maybe ARIMA model for time series analysis, then use weekday for regression testing
        # Further suggestion: also create an API to MetOffice to get current temperature, use ANOVA test for significance
        # Then compare predicted price with arbitrage opportunity / moving average, which has better gain
        
        # list of dict: {time, action:[charge/discharge], cash=float}
        actions = []
        cashflow = 0  # positive: receive money by discharging, negative: pay money to charge
        action_dict = {'time': self.current_time, 'action': actions, 'cash': cashflow}
        if actions:
            self.action_history.append(action_dict)

    def simulate(self):
        # Read both market's data into data frame
        df_1 = pandas.read_csv(os.path.join('data', 'market_1.csv'))
        df_2 = pandas.read_csv(os.path.join('data', 'market_2.csv'))

        holiday = {date(2018, 1, 1)}  # Only 1 holiday from 2018-01-01 to 2018-03-01
        # use date().weekday() to find weekday. Both bank holiday and weekday = 5 or 6 defined as holiday

        market_1_ind = 0
        market_2_ind = 0
        self.current_time = datetime(2018, 1, 1)
        self.start_time = self.current_time
        self.market_1_interval = 0.5  # half hour interval
        self.market_2_interval = 1  # 1 hour interval
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

    
if __name__ == "__main__":
    BatteryMarketOptimiser().simulate()
