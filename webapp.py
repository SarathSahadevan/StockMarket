# !pip install fuzzywuzzy
import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import datetime
import numpy as np
import requests
from bs4 import BeautifulSoup

def find_best_match(input_string, string_list):
    best_match, score = process.extractOne(input_string, string_list)
    return best_match, score

def create_future_dates(model, start_date_nse, periods=40):
    business_days_until_today = pd.date_range(start=start_date_nse, end=pd.to_datetime("today"), freq='B')
    future_dates = pd.date_range(start=start_date_nse, periods=(len(business_days_until_today) + periods), freq='B')
    future = pd.DataFrame({'ds': future_dates})
    return future

def get_historical_stock_data_nse(ticker, start_date, end_date):
    try:
        ticker_nse = f"{ticker}.NS"
        data = yf.download(ticker_nse, start=start_date, end=end_date)

        st.write("Historical Data:")
        st.write(data.tail())

        return data

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def prepare_data_for_prophet(historical_data):
    df_prophet = historical_data.reset_index()
    df_prophet = df_prophet[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    return df_prophet

def fit_prophet_model(data, seasonality_prior_scale=0.1, yearly_seasonality=20, weekly_seasonality=15):
    model = Prophet(seasonality_prior_scale=seasonality_prior_scale, yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=False)
    model.fit(data)
    return model

def make_predictions_and_plot(model, future):
    forecast = model.predict(future)
    st.write("PREDICTION:")

    fig = model.plot(forecast)
    st.pyplot(fig)

    # st.write("Forecast Data:")
    # st.write(forecast)

def crawl_screener(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all tables on the page
        tables = soup.find_all('table', {'class': 'data-table'})

        # Print data from specified tables
        if len(tables) >= 3:
            st.write("\n\n Quarterly Results Table:")
            display_table(tables[0])

            st.write("\n\n Profit and Loss Table:")
            display_table(tables[1])

            # st.write("\nPeer Comparison Table:")
            # display_table(tables[2])

            # Extract data from the "quarterly-shp" div
            shp_div = soup.find('div', {'id': 'quarterly-shp'})
            if shp_div:
                st.write("\n\nShare Holding Pattern Table:")
                extract_shp_data(shp_div)
            else:
                st.warning("Share Holding Pattern data not found on the page.")
        else:
            st.warning("Insufficient tables found on the page.")
    else:
        st.error(f"Failed to retrieve the page. Status code: {response.status_code}")

# !pip install plotly
import plotly.express as px

def extract_shp_data(shp_div):
    # Find all tables within the div
    tables = shp_div.find_all('table', {'class': 'data-table'})

    # Display data from the tables
    for idx, table in enumerate(tables, 1):
        # Convert the table data to a DataFrame
        table_data = []
        rows = table.find_all('tr')

        # Add the header
        header = [th.text.strip() for th in rows[0].find_all('th')]
        table_data.append(header)

        # Add the data
        for row in rows[1:]:
            data = [td.text.strip() for td in row.find_all('td')]
            table_data.append(data)

        # Convert the table data to a DataFrame
        df = pd.DataFrame(table_data[1:-2], columns=table_data[0])

        # Exclude 'No of Share Holder'
        df = df[df[df.columns[0]] != 'No of Share Holder']

        # Display the DataFrame with custom CSS for freezing the first column
        st.markdown(
            f"""
            <style>
                .freeze-table {{
                    overflow-x: auto;
                }}
                .freeze-table table {{
                    table-layout: fixed;
                    width: auto;
                    border-collapse: collapse;
                }}
                .freeze-table th, .freeze-table td:first-child {{
                    position: sticky;
                    left: 0;
                    background-color: white;
                    z-index: 1;
                    border: 1px solid #ddd; /* Add border to left side */
                }}
                .freeze-table th, .freeze-table td {{
                    background-color: white;
                    color: black;  /* Set text color to black */
                    border: 1px solid #ddd; /* Add border to top, right, and bottom sides */
                    padding: 8px;
                    text-align: left;
                }}
            </style>
            """
        , unsafe_allow_html=True)

        # st.markdown(f"**Share Holding Pattern Table {idx}:**")
        st.dataframe(df)

        # Create a horizontal bar chart using Plotly Express
        fig = px.bar(df, y=df.columns[0], x=df.columns[1], orientation='h',
                     title=f'Share Holding Pattern Table {idx}', labels={df.columns[0]: 'Category', df.columns[1]: 'Percentage'})
        st.plotly_chart(fig)


def display_table(table):
    # Extract data from the table
    rows = table.find_all('tr')

    # Create a list to hold table data
    table_data = []

    # Add the header
    header = [th.text.strip() for th in rows[0].find_all('th')]
    table_data.append(header)

    # Add the data
    for row in rows[1:]:
        data = [td.text.strip() for td in row.find_all('td')]
        table_data.append(data)

    # Convert the table data to a DataFrame
    df = pd.DataFrame(table_data[1:], columns=table_data[0])

    # Display the DataFrame with custom CSS for freezing the first column
    st.markdown(
        f"""
        <style>
            .freeze-table {{
                overflow-x: auto;
            }}
            .freeze-table table {{
                table-layout: fixed;
                width: auto;
                border-collapse: collapse;
            }}
            .freeze-table th, .freeze-table td:first-child {{
                position: sticky;
                left: 0;
                background-color: white;
                z-index: 1;
                border: 1px solid #ddd; /* Add border to left side */
            }}
            .freeze-table th, .freeze-table td {{
                background-color: white;
                color: black;  /* Set text color to black */
                border: 1px solid #ddd; /* Add border to top, right, and bottom sides */
                padding: 8px;
                text-align: left;
            }}
        </style>
        """
    , unsafe_allow_html=True)

    st.markdown(df.to_html(classes='freeze-table', escape=False, index=False), unsafe_allow_html=True)



def main():
    st.title("Stock Analysis App")

    ticker_symbol_nse = st.text_input("Enter NSE Ticker Symbol:", "Ujjivansfb")

    start_date_nse = "2021-01-01"
    end_date_nse = str(datetime.datetime.today().date())
    
    best_match, score = find_best_match(ticker_symbol_nse, lookup)
    historical_data_nse = get_historical_stock_data_nse(best_match, start_date_nse, end_date_nse)

    if historical_data_nse is not None:
        df_prophet = prepare_data_for_prophet(historical_data_nse)

        model = fit_prophet_model(df_prophet)

        future = create_future_dates(model, start_date_nse, periods=40)

        make_predictions_and_plot(model, future)

        crawl_screener(f"https://www.screener.in/company/{best_match}/")

if __name__ == "__main__":
    lookup = list(pd.read_excel("lookup.xlsx")["Symbol"])

    main()
