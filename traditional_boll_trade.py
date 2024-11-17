import yfinance as yf
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']


class Boll_Predict:
    def __init__(self, stock_data):
        self.stock_data = stock_data

    def make_predict(self, day=20):
        self.stock_data['MA'] = self.stock_data['Close'].rolling(window=day).mean()
        self.stock_data['STD'] = self.stock_data['Close'].rolling(window=day).std()
        self.stock_data['Upper'] = self.stock_data['MA'] + (2 * self.stock_data['STD'])
        self.stock_data['Lower'] = self.stock_data['MA'] - (2 * self.stock_data['STD'])

        self.stock_data['Signal'] = 0
        self.stock_data.loc[self.stock_data['Close'] < self.stock_data['Lower'], 'Signal'] = 1
        self.stock_data.loc[self.stock_data['Close'] > self.stock_data['Upper'], 'Signal'] = -1
        self.stock_data.dropna(inplace=True)

        return self.stock_data


class Portfolio_Trading:
    def __init__(self, stock_codes, start_date, end_date):
        self.stock_codes = stock_codes
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = {}
        self.models = {}
        self.initial_capital = 1000000  # Initial capital
        self.cash = self.initial_capital
        self.positions = {stock: 0 for stock in stock_codes}
        self.total_values = []
        self.buy_points = []  # To store buy points (date and price)
        self.sell_points = []  # To store sell points (date and price)
        self.hsi_return = 0  # 恒生指数的收益率
        self.total_return = 0  # 投资组合的收益率
        self.alpha = 0  # 跑赢指数的收益率

        # Download stock data
        for stock_code in stock_codes:
            data = yf.download(stock_code, start=start_date, end=end_date)
            # Save data to an Excel file
            excel_filename = f"{stock_code}_data.xlsx"
            data.to_excel(excel_filename)
            print(f"Data for {stock_code} saved to {excel_filename}")
            data['Stock'] = stock_code
            self.stock_data[stock_code] = data

        # Apply Bollinger Band strategy
        for stock_code in stock_codes:
            boll_predict = Boll_Predict(self.stock_data[stock_code])  # Pass stock data directly
            boll_predict_data = boll_predict.make_predict()
            self.stock_data[stock_code] = boll_predict_data

    def backtest(self):
        for date in self.stock_data[self.stock_codes[0]].index:  # Use the date index of the first stock
            daily_cash = self.cash
            daily_position_value = 0

            for stock_code, data in self.stock_data.items():
                if date not in data.index:
                    continue

                df_row = data.loc[date]
                price = df_row['Close']

                # Buy signal
                if df_row['Signal'] == 1 and self.cash >= price:
                    buy_amount = int(self.cash // price)  # Buy with available funds
                    if buy_amount > 0:
                        self.cash -= buy_amount * price
                        self.positions[stock_code] += buy_amount
                        print(
                            f"{date}: Buy {stock_code}, Amount: {buy_amount}, Price: {price:.2f}, Current Cash: {self.cash:.2f}")
                        self.buy_points.append((date, price))

                # Sell signal
                elif df_row['Signal'] == -1 and self.positions[stock_code] > 0:
                    sell_amount = self.positions[stock_code]
                    self.cash += sell_amount * price  # Sell stocks and get cash
                    self.positions[stock_code] -= sell_amount  # Decrease position
                    print(
                        f"{date}: Sell {stock_code}, Amount: {sell_amount}, Price: {price:.2f}, Current Cash: {self.cash:.2f}")
                    self.sell_points.append((date, price))

                # Calculate position value
                daily_position_value += self.positions[stock_code] * price

            # Record total value
            self.total_values.append(self.cash + daily_position_value)

        # Finally calculate the return
        final_value = self.total_values[-1]
        self.total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        # Get Hang Seng Index data
        hsi_data = yf.download('^HSI', start=self.start_date, end=self.end_date)['Close']


        # Align Hang Seng Index data with the backtest data
        hsi_data = hsi_data.reindex(self.stock_data[self.stock_codes[0]].index, method='ffill')

        # Normalize Hang Seng Index data based on initial capital
        hsi_data = hsi_data / hsi_data.iloc[0] * self.initial_capital

        # Calculate the return of Hang Seng Index
        hsi_final_value = hsi_data.iloc[-1]  # Use the last day's value of Hang Seng Index
        self.hsi_return = (hsi_final_value - self.initial_capital) / self.initial_capital * 100

        # Calculate relative outperformance
        self.alpha = self.total_return - self.hsi_return

        print(
            f"Initial Capital: {self.initial_capital:.2f}, Final Value: {final_value:.2f}, Total Return: {self.total_return:.2f}%")
        print(f"Hang Seng Index Return: {self.hsi_return:.2f}%")
        print(f"Outperformance over Hang Seng Index: {self.alpha:.2f}%")

    def plot_results(self):
        # Plot portfolio vs Hang Seng Index comparison chart
        plt.figure(figsize=(12, 6))
        # Plot total asset curve
        plt.plot(self.stock_data[self.stock_codes[0]].index, self.total_values, label='Portfolio Total Asset',
                 color='blue',
                 linewidth=3)
        # Download Hang Seng Index data
        hsi_data = yf.download('^HSI', start=self.start_date, end=self.end_date)['Close']
        # Align Hang Seng Index data with the backtest data
        hsi_data = hsi_data.reindex(self.stock_data[self.stock_codes[0]].index, method='ffill')
        # Normalize Hang Seng Index data based on initial capital
        hsi_data = hsi_data / hsi_data.iloc[0] * self.initial_capital
        plt.plot(hsi_data.index, hsi_data, label='Hang Seng Index',
                 color='red', linestyle='-', linewidth=3, markersize=8, markerfacecolor='red')

        # Mark buy and sell signals
        if self.buy_points:
            buy_dates, _ = zip(*self.buy_points)
            buy_prices = [self.total_values[self.stock_data[self.stock_codes[0]].index.get_loc(date)] for date in
                          buy_dates]
            plt.scatter(buy_dates, buy_prices, label='Buy Signal', marker='^', color='green')

        if self.sell_points:
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


if __name__ == "__main__":
    # Set stock codes and backtest start and end dates
    stock_codes = ['1810.HK', '3690.HK', '0700.HK', '9988.HK', '9618.HK']

    # Create a Portfolio_Trading object
    portfolio = Portfolio_Trading(stock_codes, start_date='2019-01-01', end_date='2024-01-01')

    # Perform backtest
    portfolio.backtest()

    # Plot backtest results
    portfolio.plot_results()
