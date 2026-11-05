import pandas
import os
from datetime import datetime, date, timedelta
from copy import deepcopy


# Read Attachment 1 as config
param_df = pandas.read_csv('Attachment.csv')
PARAM = {}
for name, value in zip(param_df['Name'], param_df['Values']):
    PARAM[name] = value


def moving_average(history_data: list, market: int, days=7)  -> float:
    sum_price = 0
    count = 0
    current_time = history_data[-1]['time']
    for item in reversed(history_data):
        if item['time'] > current_time - timedelta(days=days):
            sum_price += item[f'market_{market}_price']
            count += 1
        else:
            break
    return sum_price / count


def decide_action() -> dict:
    # Decide to charge / discharge based on arbitrage, moving average or ARIMA prediction
    # Calculate charging cost, If cost > expected gain, return no actions
    cycle_cost = PARAM['Capex'] / PARAM['Lifetime (2)']  # battery action cost by whole cycle, ignore time cost.
    total_efficiency = (1 - PARAM['Battery charging efficiency']) * (1 - PARAM['Battery discharging efficiency'])
    action_dict = {'time': datetime(2018, 1, 1), 'action': [], 'cash': 0}
    # Charge efficiency
    # Discharge efficiency

    # TODO: predict the price in the upcoming hours after having enough data (4-week?)
    # Maybe ARIMA model for time series analysis, then use weekday for regression testing
    # Further suggestion: also create an API to MetOffice to get current temperature, use ANOVA test for significance
    # Then compare predicted price with arbitrage opportunity, which has better gain
    return action_dict


def main():
    # Read both market's data into data frame
    df_1 = pandas.read_csv(os.path.join('data', 'market_1.csv'))
    df_2 = pandas.read_csv(os.path.join('data', 'market_2.csv'))

    holiday = {date(2018, 1, 1)}  # Only 1 holiday from 2018-01-01 to 2018-03-01

    price_history = []  # As we don't know the future, everything should only base on history price
    # list of dict: {time, is_holiday, market_1_price, market_2_price}
    # use date().weekday() to find weekday. Both bank holiday and weekday = 5 or 6 defined as holiday
    price_dict = {'time': datetime(2018, 1, 1), 'is_holiday': False, 'market_1_price': 0, 'market_2_price': 0}

    action_history = []
    # list of dict: {time, action:[charge/discharge/no_action], cash=float}

    market_1_ind = 0
    market_2_ind = 0
    current_time = datetime(2018, 1, 1)
    current_battery = [0, 0]  # assume we have 2 battery initially, so that we can go for arbitrage profit without prediction

    while market_1_ind < len(df_1) and market_2_ind < len(df_2):
        current_price_dict = deepcopy(price_dict)
        # No hard code, make it suitiable for flexible time as long as the time is orderred
        market_1_time = datetime.strptime(df_1['time'][market_1_ind], '%Y-%m-%d %H:%M:%S')
        market_2_time = datetime.strptime(df_2['time'][market_2_ind], '%Y-%m-%d %H:%M:%S')
        current_time = min(market_1_time, market_2_time)

        if current_time >= market_1_time:
            # Read market 1, get price, then move to the next row
            market_1_price = df_1['price'][market_1_ind]
            market_1_ind += 1

        if current_time >= market_2_time:
            # Read market 2, get price, then move to the next row
            market_2_price = df_2['price'][market_2_ind]
            market_2_ind += 1

        current_price_dict['time'] = current_time
        current_price_dict['is_holiday'] = current_time.weekday() in [5, 6] or current_time.date() in holiday
        current_price_dict['market_1_price'] = market_1_price
        current_price_dict['market_2_price'] = market_2_price
        price_history.append(current_price_dict)

        current_action_dict = decide_action()
        action_history.append(current_action_dict)

    
if __name__ == "__main__":
    main()
