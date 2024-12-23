from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import json
import argparse


def save_dataset(symbol, time_window):
    try:
        # Load credentials
        credentials = json.load(open('creds.json', 'r'))
        api_key = credentials.get('av_api_key')
        if not api_key:
            raise ValueError("API key not found in creds.json")

        print(f"Fetching data for symbol: {symbol}, time window: {time_window}")
        ts = TimeSeries(key=api_key, output_format='pandas')

        # Fetch data based on the time window
        if time_window == 'intraday':
            data, meta_data = ts.get_intraday(
                symbol=symbol, interval='1min', outputsize='full')
        elif time_window == 'daily':
            data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        elif time_window == 'daily_adj':
            data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize='full')

        # Display and save data
        pprint(data.head(10))
        data.to_csv(f'./{symbol}_{time_window}.csv', index=True)
        print(f"Data saved to ./{symbol}_{time_window}.csv")

    except FileNotFoundError:
        print("Error: creds.json file not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download historical stock data for a given symbol and time window.")

    parser.add_argument('symbol', type=str, help="The stock symbol you want to download.")
    parser.add_argument(
        'time_window',
        type=str,
        choices=['intraday', 'daily', 'daily_adj'],
        help="The time period you want to download the stock history for.")

    args = parser.parse_args()
    save_dataset(args.symbol, args.time_window)