import pandas_datareader.data as web
import datetime

start_date = datetime.datetime(2003, 1, 1)
end_date = datetime.datetime(2023, 12, 31)

# Fetching Case-Shiller Index data
cs_index = web.DataReader('CSUSHPISA', 'fred', start_date, end_date)
cs_index.to_csv('case_shiller_index.csv')  # Save locally

# Fetching Federal Funds Rate data
fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)
fed_funds_rate.to_csv('fed_funds_rate.csv')  # Save locally
