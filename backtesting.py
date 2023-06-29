import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Build a Backtesting Engine
# Implement the backtesting engine that simulates trades and calculates performance
class BacktestingEngine:
    def __init__(self, data, factor):
        self.data = data
        self.portfolio = {'cash': 1000000, 'positions': {}, 'total': [1000000]}
        self.trades = []
        self.factor = factor

    def run_backtest(self):
        # all in at per trade
        risk_per_trade = 1  # Risk 2% of portfolio per trade

        for index, row in self.data.iterrows():
            # Implement your trading logic for each data point
            close_price = row['close']
            signal = row['signal'] 

            if signal == 1:
                # Buy signal
                available_cash = self.portfolio['cash']
                position_size = available_cash * risk_per_trade / close_price
                self.execute_trade(symbol=self.data.name, quantity=position_size, price=close_price)

                positions_value = sum(self.portfolio['positions'].get(symbol, 0) * close_price for symbol in self.portfolio['positions'])
                self.portfolio['total'].append(self.portfolio['cash'] + positions_value)
            elif signal == -1:
                # Sell signal
                position_size = self.portfolio['positions'].get(self.data.name, 0)
                if position_size > 0:
                    self.execute_trade(symbol=self.data.name, quantity=-position_size, price=close_price)
                    positions_value = sum(self.portfolio['positions'].get(symbol, 0) * close_price for symbol in self.portfolio['positions'])
                    self.portfolio['total'].append(self.portfolio['cash'] + positions_value)
            else:
                # No trade
                value = self.portfolio['total'][-1]
                self.portfolio['total'].append(value)


    def calculate_performance(self):
        # Calculate performance metrics based on executed trades
        trade_prices = np.array([trade['price'] for trade in self.trades])
        trade_quantities = np.array([trade['quantity'] for trade in self.trades])

        trade_returns = np.diff(trade_prices) / trade_prices[:-1]
        trade_pnl = trade_returns * trade_quantities[:-1]

        total_pnl = np.sum(trade_pnl)
        average_trade_return = np.mean(trade_returns)
        win_ratio = np.sum(trade_pnl > 0) / len(trade_pnl)

        return total_pnl, average_trade_return, win_ratio

    def execute_trade(self, symbol, quantity, price):
        # Update portfolio and execute trade
        self.portfolio['cash'] -= quantity * price

        if symbol in self.portfolio['positions']:
            self.portfolio['positions'][symbol] += quantity
        else:
            self.portfolio['positions'][symbol] = quantity
        
        self.trades.append({'symbol': symbol, 'quantity': quantity, 'price': price})

    # def get_portfolio_value(self, price):
    #     # Calculate the current value of the portfolio
    #     positions_value = sum(self.portfolio['positions'].get(symbol, 0) * price for symbol in self.portfolio['positions'])
    #     return self.portfolio['cash'] + positions_value

    def get_portfolio_returns(self):
        # Calculate the daily portfolio returns
        # portfolio_value = [self.get_portfolio_value(row['close']) for _, row in self.data.iterrows()]
        portfolio_value = self.portfolio['total']
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        return returns

    def print_portfolio_summary(self):
        print('--- Portfolio Summary ---')
        print('Cash:', self.portfolio['cash'])
        print('Positions:')
        for symbol, quantity in self.portfolio['positions'].items():
            print(symbol + ':', quantity)
        print('Trades:')
        for i in self.trades:
            print(i)

    def plot_portfolio_value(self):
        # portfolio_value = [self.get_portfolio_value(row['close']) for _, row in self.data.iterrows()]
        portfolio_value = self.portfolio['total']
        dates = self.data['date']
        signals = self.data['signal']

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot portfolio value
        ax1.plot(dates, portfolio_value, 'b-', label='Portfolio Value')
        ax1.set_xlabel('Date') 
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')

        # Plot buy/sell signals
        # ax2 = ax1.twinx()
        # ax2.plot(dates, signals, 'r-', label='Buy/Sell Signal')
        # ax2.set_ylabel('Signal')
        # ax2.grid(None)

        fig.tight_layout()
        plt.show()

class SetTradingSignalEngine:
    def __init__(self, data, factor):
        self.data = data.pivot(index='date', columns='symbol', values='close')
        self.date = data.date.unique()
        self.factor = factor.set_index('date')
        self.bound = self.cal_bound().set_index('date')
    
    def check_trend(self, date):

        hist = self.factor[self.factor.index == date-datetime.timedelta(days = 2)].values
        cur = self.factor[self.factor.index == date-datetime.timedelta(days = 1)].values
        if hist < cur:
            tend = True
        else:
            tend = False

        return tend
    
    def cal_bound(self, type=1):
        ub = [0]
        lb = [0]
        match type:
            case 0:

                for i in self.date[1:]:
                    mean = self.factor[self.factor.index < i].values.mean()
                    std = self.factor[self.factor.index < i].values.std()
                    ub.append(mean+std)
                    lb.append(mean-std)
            case 1:
                ub = [0.01] * len(self.factor)
                lb = [-0.01] * len(self.factor)

        
        bound = pd.DataFrame({'date': self.date,
                           'upper_bound': ub,
                            'lower_bound': lb})
        
        bound.set_index('date')
        
        return bound

    def set_signal_init(self, type=0):

        signal = [0] * 2
        ub = self.bound['upper_bound']
        lb = self.bound['lower_bound']
        threshold = 0.005
        
        match type:
            case 0:

                for i in self.date[2:]:
                    tend = self.check_trend(i)
                    ub_abs = abs(self.factor[self.factor.index == i-datetime.timedelta(days=1)]['factor'].values - ub[ub.index == i].values)
                    lb_abs = abs(self.factor[self.factor.index == i-datetime.timedelta(days=1)]['factor'].values - lb[lb.index == i].values)
                    if (tend & (ub_abs <= threshold)):
                        tmp = 1
                    elif ((not tend) & (lb_abs <= threshold)):
                        tmp = -1
                    else:
                        tmp = 0
                    signal.append(tmp)
        
        return signal
    
    def set_signal(self):
        signal = self.set_signal_init()
        tag = signal[np.nonzero(signal)[0][0]]
        for i in range(np.nonzero(signal)[0][0]+1, len(signal)):
            if signal[i] == tag:
                signal[i] = 0
            elif signal[i] == -tag:
                tag = signal[i]
        
        return signal
    
    def plot_signal(self, signal):
        factor = self.factor['factor']
        ub = self.bound['upper_bound']
        lb = self.bound['lower_bound']

        fig, ax1 = plt.subplots(figsize=(10, 6))


        # Plot portfolio value
        ax1.plot(self.date, factor, 'b-', label='factor')
        ax1.plot(self.date, ub, 'k-', label='upper bound')
        ax1.plot(self.date, lb, 'k-', label='lower bound')
        ax1.set_xlabel('Date') 
        ax1.set_ylabel('factor Value')
        ax1.set_title('factor Value Over Time')

        # Plot buy/sell signals
        ax2 = ax1.twinx()
        ax2.plot(self.date, signal, 'r-', label='Buy/Sell Signal')
        ax2.set_ylabel('Signal')
        ax2.grid(None)

        fig.tight_layout()
        plt.show()


def backtest_main(data, factor):

    # Define Trading Strategies
    # Implement your trading strategies using the data
    # sma_indicator is the simple moving average
    # sma_short = ta.trend.sma_indicator(data['Close'], window=50)
    # sma_long = ta.trend.sma_indicator(data['Close'], window=200)

    signalengine = SetTradingSignalEngine(data, factor)
    data['signal'] = signalengine.set_signal()
    signalengine.plot_signal(data['signal'])
    data.name = data['symbol'][0]  # Set the name attribute for the data DataFrame

    # Start back testing
    engine = BacktestingEngine(data, factor)
    # initial_portfolio_value = engine.get_portfolio_value(data.iloc[0]['close'])
    engine.run_backtest()
    engine.print_portfolio_summary()


    # Evaluate Performance
    # final_portfolio_value = engine.get_portfolio_value(data.iloc[-1]['close'])
    returns = engine.get_portfolio_returns()
    # total_returns = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
    # annualized_returns = (1 + total_returns) ** (252 / len(data)) - 1  # Assuming 252 trading days in a year
    # volatility = np.std(returns) * np.sqrt(252)
    # sharpe_ratio = (annualized_returns - 0.02) / volatility  # Assuming risk-free rate of 2%

    # Calculate and print performance metrics
    total_pnl, average_trade_return, win_ratio = engine.calculate_performance()
    print('--- Performance Metrics ---')
    # print(f'Total Returns: {total_returns:.2%}')
    # print(f'Annualized Returns: {annualized_returns:.2%}')
    # print(f'Volatility: {volatility:.2%}')
    # print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
    print(f'Total P&L: {total_pnl:.2f}')
    print(f'Average Trade Return: {average_trade_return:.2%}')
    print(f'Win Ratio: {win_ratio:.2%}')

    # Visualize Results
    engine.plot_portfolio_value()

def download_index_data(code, start_time, end_time):
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol=code)
    
    # select the time
    startdate = pd.to_datetime(start_time).date()
    enddate = pd.to_datetime(end_time).date()
    index_data = df[(df['date'] >= startdate) & (df['date'] <= enddate)].reset_index()
    
    # add the index code at the first
    duration = len(index_data)
    symbol_column = pd.DataFrame(np.array([code]*duration), columns=['symbol'])
    data = pd.merge(symbol_column,index_data,left_index=True, right_index=True)
    
    return data

def augur_001(df):
    '''
    计算动荡指数Turbulence Index
    ref: https://zhuanlan.zhihu.com/p/629602283?utm_medium=social&utm_oi=38417056399360&utm_psn=1641600863545217024&utm_source=wechat_session

    Params
    ------
    2021~2022 的中证500指数
    columns: date      symbol      open      high       low     close       volume

    Returns
    -------
    动荡指数
    columns: date              turbulence

    '''
    
    df_price_pivot=df.pivot(index='date', columns='symbol', values='close')
    unique_date = df.date.unique()
    
    start = 252
    turbulence_index = [0]*start

    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        turbulence_temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        turbulence_index.append(turbulence_temp)
    
    factor = pd.DataFrame({'date':df_price_pivot.index,
                                     'factor':turbulence_index})
    
    return factor

def augur_002(df):
    import math
    v1 = df.pivot(index='date', columns='symbol', values='close')
    v2 = df.pivot(index='date', columns='symbol', values='open')
    unique_date = df.date.unique()
    
    skew = []
    for i in range(len(unique_date)):
        v3 = v2.iloc[v2.index == unique_date[i],0].values
        v4 = v1.iloc[v1.index == unique_date[i],0].values
        tmp = math.log(v3)-math.log(v4)
        skew.append(tmp)
    
    factor = pd.DataFrame({'date':v1.index,
                        'factor':skew})
    
    return factor

if __name__ == '__main__':

    code = "sz399905"
    start_time = '2018-01-01'
    end_time = '2023-05-31'
    data = download_index_data(code, start_time, end_time)
    
    factor = augur_002(data)

    backtest_main(data, factor)
