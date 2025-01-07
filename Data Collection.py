import pandas as pd
import requests
import datetime as dt
import pandas_datareader.data as web

# Define time range for data
start_date = dt.datetime(2003, 1, 1)
end_date = dt.datetime(2023, 12, 31)

# 1. S&P Case-Shiller Home Price Index
cs_index = web.DataReader('CSUSHPISA', 'fred', start_date, end_date)
cs_index.to_csv('case_shiller_index.csv')
print("S&P Case-Shiller Home Price Index collected.")

# 2. Federal Funds Rate
fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)
fed_funds_rate.to_csv('fed_funds_rate.csv')
print("Federal Funds Rate collected.")

# 4. Consumer Price Index (CPI)
cpi_data = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi_data.to_csv('cpi_data.csv')
print("CPI data collected.")

# 6. Housing Starts
housing_starts = web.DataReader('HOUST', 'fred', start_date, end_date)
housing_starts.to_csv('housing_starts.csv')
print("Housing Starts data collected.")


