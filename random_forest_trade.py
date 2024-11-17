import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pylab import mpl

# 改变plot字体，适应中文
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']


class Portfolio_Trading:
    def __init__(self, stock_codes, start_date, end_date):
        self.stock_codes = stock_codes
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = {}
        self.models = {}
        self.initial_capital = 1000000  # 初始资金
        self.cash = self.initial_capital
        self.positions = {stock: 0 for stock in stock_codes}
        self.total_values = []
        self.total_values_buysell = []
        self.buy_points = []  # 用于存储买入点 (日期和价格)
        self.sell_points = []  # 用于存储卖出点 (日期和价格)
        self.hsi_return = 0  # 恒生指数的收益率
        self.total_return = 0  # 投资组合的收益率
        self.alpha = 0  # 跑赢指数的收益率

        # 下载股票数据
        for stock_code in stock_codes:
            data = yf.download(stock_code, start=start_date, end=end_date)
            data['Stock'] = stock_code
            self.stock_data[stock_code] = data

    def preprocess_and_train(self):
        for stock_code, data in self.stock_data.items():
            # Calculate technical indicators
            data['MA20'] = data['Close'].rolling(window=20).mean()  # 20-day moving average
            data['STD20'] = data['Close'].rolling(window=20).std()  # 20-day standard deviation
            data['Upper'] = data['MA20'] + (2 * data['STD20'])  # Upper Bollinger Band
            data['Lower'] = data['MA20'] - (2 * data['STD20'])  # Lower Bollinger Band
            data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(window=14).apply(
                lambda x: (x[x > 0].sum() / abs(x[x < 0].sum())) if abs(x[x < 0].sum()) > 0 else 0)))  # 14-day RSI

            # Label generation: Buy/sell signals based on future returns prediction
            data['Return'] = data['Close'].pct_change().shift(-1)  # Next day return
            data['Label'] = np.where(data['Return'] > 0, 1, 0)  # 1 for buy signal, 0 for sell signal

            # Data cleaning
            data.dropna(inplace=True)  # Drop rows with missing values

            # Select features and labels
            features = data[['MA20', 'STD20', 'RSI', 'Close']]
            labels = data['Label']
            # Data standardization
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
            # Train the Random Forest model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            # Save the model
            self.models[stock_code] = model
            # Make predictions and save to data
            data['Prediction'] = model.predict(features_scaled)
            self.stock_data[stock_code] = data

    def backtest(self):
        for date in self.stock_data[self.stock_codes[0]].index:  # 使用第一个股票的日期索引
            daily_cash = self.cash
            daily_position_value = 0

            for stock_code, data in self.stock_data.items():
                if date not in data.index:
                    continue

                df_row = data.loc[date]
                price = df_row['Close']

                # 初始化买入参数
                initial_buy_percentage = 0.01  # 初始买入使用总资金的10%
                buy_multiple = 2  # 每次加仓时增加的倍数
                max_buy_times = 10  # 最大买入次数
                last_buy_price = None  # 上次买入价格

                # Initial purchase
                if 'Prediction' in df_row and df_row['Prediction'] > 0 and self.cash >= price:
                    initial_buy_amount = int(
                        (self.cash * initial_buy_percentage) // price)  # Initial number of shares to buy
                    if initial_buy_amount > 0:  # Ensure the number of shares bought is greater than zero
                        self.cash -= initial_buy_amount * price  # Deduct funds
                        self.positions[stock_code] += initial_buy_amount  # Increase position
                        last_buy_price = price  # Update last buy price
                        print(
                            f"{date}: Bought {stock_code}, Amount: {initial_buy_amount}, Price: {price:.2f}, Current cash: {self.cash:.2f}")
                        self.buy_points.append((date, price))  # Record the buy point

                # Subsequent purchases (Martingale strategy)
                elif 'Prediction' in df_row and df_row['Prediction'] > 0 and self.cash >= price:
                    # Check if conditions are met for additional purchase: price drops by a certain percentage (e.g., 10%)
                    if last_buy_price and price <= last_buy_price * (1 - 0.1):  # Current price is 10% lower than the last buy price
                        buy_amount = int((
                                                     self.cash * initial_buy_percentage * buy_multiple) // price)  # Number of shares to buy (increased by a multiplier)
                        if buy_amount > 0:  # Ensure the number of shares bought is greater than zero
                            self.cash -= buy_amount * price  # Deduct funds
                            self.positions[stock_code] += buy_amount  # Increase position
                            last_buy_price = price  # Update last buy price
                            print(
                                f"{date}: Added position {stock_code}, Amount: {buy_amount}, Price: {price:.2f}, Current cash: {self.cash:.2f}")
                            self.buy_points.append((date, price))  # Record the buy point
                # 卖出信号
                elif 'Prediction' in df_row and df_row['Prediction'] < 0 and self.positions[stock_code] > 0:
                    sell_amount = self.positions[stock_code]
                    self.cash += sell_amount * price
                    self.positions[stock_code] -= sell_amount
                    print(
                        f"{date}: 卖出 {stock_code}, 数量: {sell_amount}, 价格: {price:.2f}, 当前现金: {self.cash:.2f}")
                    self.sell_points.append((date, price))

                # 计算持仓市值
                daily_position_value += self.positions[stock_code] * price

            # 记录总资产
            self.total_values.append(self.cash + daily_position_value)

        # 最后计算收益率
        final_value = self.total_values[-1]
        self.total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # 获取恒生指数数据
        hsi_data = yf.download('^HSI', start=self.start_date, end=self.end_date)['Close']

        # 保证恒生指数数据和回测数据对齐
        hsi_data = hsi_data.reindex(self.stock_data[self.stock_codes[0]].index, method='ffill')

        # 按初始资金标准化恒生指数数据
        hsi_data = hsi_data / hsi_data.iloc[0] * self.initial_capital

        # 计算恒生指数的收益率
        hsi_final_value = hsi_data.iloc[-1]  # 使用恒生指数数据的最后一日价值
        self.hsi_return = (hsi_final_value - self.initial_capital) / self.initial_capital * 100

        # 计算相对跑赢指数收益率
        self.alpha = self.total_return - self.hsi_return

        print(f"初始资金: {self.initial_capital:.2f}, 最终资产: {final_value:.2f}, 收益率: {self.total_return:.2f}%")
        print(f"恒生指数收益率: {self.hsi_return:.2f}%")
        print(f"跑赢恒生指数: {self.alpha:.2f}%")

    def backtest_buysell(self):
        for date in self.stock_data[self.stock_codes[0]].index:  # 使用第一个股票的日期索引
            daily_cash = self.cash
            daily_position_value = 0

            for stock_code, data in self.stock_data.items():
                if date not in data.index:
                    continue

                df_row = data.loc[date]
                price = df_row['Close']

                # 初始化买入参数
                initial_buy_percentage = 0.01  # 初始买入使用总资金的10%
                buy_multiple = 2  # 每次加仓时增加的倍数
                max_buy_times = 10  # 最大买入次数
                last_buy_price = None  # 上次买入价格

                # 初始买入
                if 'Prediction' in df_row and df_row['Prediction'] > 0 and self.cash >= price:
                    initial_buy_amount = int((self.cash * initial_buy_percentage) // price)  # 初始买入股数
                    if initial_buy_amount > 0:  # 确保买入股数大于零
                        self.cash -= initial_buy_amount * price  # 扣除资金
                        self.positions[stock_code] += initial_buy_amount  # 增加持仓
                        last_buy_price = price  # 更新上次买入价格
                        print(
                            f"{date}: 买入 {stock_code}, 数量: {initial_buy_amount}, 价格: {price:.2f}, 当前现金: {self.cash:.2f}")
                        self.buy_points.append((date, price))  # 记录买入点

                # 后续买入（马丁策略）
                elif 'Prediction' in df_row and df_row['Prediction'] > 0 and self.cash >= price:
                    # 判断是否符合加仓条件：价格下跌了一定比例（例如10%）
                    if last_buy_price and price <= last_buy_price * (1 - 0.1):  # 当前价格低于上次买入价格的90%
                        buy_amount = int((self.cash * initial_buy_percentage * buy_multiple) // price)  # 加仓股数（倍数递增）
                        if buy_amount > 0:  # 确保买入股数大于零
                            self.cash -= buy_amount * price  # 扣除资金
                            self.positions[stock_code] += buy_amount  # 增加持仓
                            last_buy_price = price  # 更新上elif 'Prediction' in df_row and df_row['Prediction'] <= 0 and self.positions[stock_code] > 0:次买入价格
                            print(
                                f"{date}: 加仓 {stock_code}, 数量: {buy_amount}, 价格: {price:.2f}, 当前现金: {self.cash:.2f}")
                            self.buy_points.append((date, price))  # 记录买入点

                # 卖出信号
                elif 'Prediction' in df_row and df_row['Prediction'] <= 0 and self.positions[stock_code] > 0:
                    sell_amount = self.positions[stock_code]
                    self.cash += sell_amount * price  # 卖出股票，获取现金
                    self.positions[stock_code] -= sell_amount  # 减少持仓
                    print(
                        f"{date}: 卖出 {stock_code}, 数量: {sell_amount}, 价格: {price:.2f}, 当前现金: {self.cash:.2f}")
                    self.sell_points.append((date, price))  # 记录卖出点

                # 计算持仓市值
                daily_position_value += self.positions[stock_code] * price

            # 记录总资产
            self.total_values.append(self.cash + daily_position_value)

        # 最后计算收益率
        final_value = self.total_values[-1]
        self.total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # 获取恒生指数数据
        hsi_data = yf.download('^HSI', start=self.start_date, end=self.end_date)['Close']

        # 保证恒生指数数据和回测数据对齐
        hsi_data = hsi_data.reindex(self.stock_data[self.stock_codes[0]].index, method='ffill')

        # 按初始资金标准化恒生指数数据
        hsi_data = hsi_data / hsi_data.iloc[0] * self.initial_capital

        # 计算恒生指数的收益率
        hsi_final_value = hsi_data.iloc[-1]  # 使用恒生指数数据的最后一日价值
        self.hsi_return = (hsi_final_value - self.initial_capital) / self.initial_capital * 100

        # 计算相对跑赢指数收益率
        self.alpha = self.total_return - self.hsi_return

        print(f"初始资金: {self.initial_capital:.2f}, 最终资产: {final_value:.2f}, 收益率: {self.total_return:.2f}%")
        print(f"恒生指数收益率: {self.hsi_return:.2f}%")
        print(f"跑赢恒生指数: {self.alpha:.2f}%")

    def plot_results(self):
        # Download Hang Seng Index data
        hsi_data = yf.download('^HSI', start=self.start_date, end=self.end_date)['Close']

        # Align Hang Seng Index data with backtest data
        hsi_data = hsi_data.reindex(self.stock_data[self.stock_codes[0]].index, method='ffill')

        # Normalize Hang Seng Index data by initial capital
        hsi_data = hsi_data / hsi_data.iloc[0] * self.initial_capital

        # Plot asset change curve
        plt.figure(figsize=(12, 6))
        # Use the date index from stock_codes[0] as the x-axis
        plt.plot(self.stock_data[self.stock_codes[0]].index, self.total_values, label='Portfolio Total Asset',
                 color='blue', linewidth=3)
        plt.plot(hsi_data.index, hsi_data, label='Hang Seng Index (Normalized by Initial Capital)', color='red',
                 linestyle='-', linewidth=3, markersize=8, markerfacecolor='red')

        # Mark buy and sell signals
        if self.buy_points:  # Ensure buy_points is not empty
            buy_dates, _ = zip(*self.buy_points)
            buy_prices = [self.total_values[self.stock_data[self.stock_codes[0]].index.get_loc(date)] for date in
                          buy_dates]
            plt.scatter(buy_dates, buy_prices, label='Buy Signal', marker='^', color='green')

        if self.sell_points:  # Ensure sell_points is not empty
            sell_dates, _ = zip(*self.sell_points)
            sell_prices = [self.total_values[self.stock_data[self.stock_codes[0]].index.get_loc(date)] for date in
                           sell_dates]
            plt.scatter(sell_dates, sell_prices, label='Sell Signal', marker='v', color='red')

        # Display information on the chart
        result_text = f"Return: {self.total_return:.2f}%\n" \
                      f"Hang Seng Index Return: {self.hsi_return:.2f}%\n" \
                      f"Outperform Hang Seng Index: {self.alpha:.2f}%"

        # Display result text on the chart, set position and font
        plt.text(0.05, 0.95, result_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8))

        # Set title and labels
        plt.title('Portfolio vs Hang Seng Index')
        plt.xlabel('Time')
        plt.ylabel('Total Asset (Million USD)')
        plt.legend()
        plt.grid()
        plt.figure(figsize=(12, 6))
        # Iterate over all stock codes, get each stock's price data, and normalize by initial capital
        for stock_code in self.stock_codes:
            stock_data = self.stock_data[stock_code]['Close']

            # Normalize the selected stock's price data by initial capital
            stock_data = stock_data / stock_data.iloc[0] * self.initial_capital

            # Plot the stock's total asset curve
            plt.plot(stock_data.index, stock_data, label=f'{stock_code} Total Asset')
        plt.title('5 Stocks Total Asset')
        plt.xlabel('Time')
        plt.ylabel('Total Asset (Million USD)')
        plt.legend()
        plt.grid()


        plt.show()


if __name__ == '__main__':
    stock_codes = ['1810.HK', '3690.HK', '0700.HK', '9988.HK', '9618.HK']

    trading = Portfolio_Trading(stock_codes, start_date='2019-01-01', end_date='2024-01-01')
    trading.preprocess_and_train()
    trading.backtest()
    trading.plot_results()
    trading2 = Portfolio_Trading(stock_codes, start_date='2019-01-01', end_date='2024-01-01')
    trading2.preprocess_and_train()
    trading2.backtest_buysell()
    trading2.plot_results()
