import requests
import json
import pandas as pd

headers = {'Content-type': 'application/json'}
data = json.dumps({
    "seriesid": ['LNS14000000'],
    "startyear": "2003",
    "endyear": "2023"
})

response = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)

if response.status_code == 200:
    json_data = json.loads(response.text)
    
    if 'Results' in json_data:
        all_data = []
        for series in json_data['Results']['series']:
            for item in series['data']:
                all_data.append({
                    "Year": item['year'],
                    "Period": item['period'],
                    "Value": float(item['value'])
                })
        
        df = pd.DataFrame(all_data)
        
        # Filter for monthly data and create 'DATE' column
        df['PeriodMonth'] = df['Period'].str[1:].astype(int)
        df['DATE'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['PeriodMonth'].astype(str) + '-01')
        df = df[['DATE', 'Value']].rename(columns={'Value': 'UnemploymentRate'})
        
        df.to_csv('processed_unemployment_rate.csv', index=False)
        print("Processed Unemployment Rate data saved to 'processed_unemployment_rate.csv'.")
        print(df.head())
    else:
        print("Error in BLS API response: No results found.")
else:
    print(f"Failed to connect to BLS API. Status code: {response.status_code}")

