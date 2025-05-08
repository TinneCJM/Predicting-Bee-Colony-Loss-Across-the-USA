import requests
import pandas as pd
import os
import numpy as np
import time
import openmeteo_requests
import requests_cache
from retry_requests import retry
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from matplotlib.colors import Normalize
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import statsmodels.api as sm

def interactive_choropleth_by_year(df, value_column, title_prefix='Number of Colonies'):
    """
    Creates interactive choropleth maps by year and quarter for a specified column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'year', 'quarter', 'state', 'state_code', and the value column.
    - value_column (str): Column name to visualize on the map.
    - title_prefix (str): Prefix for the map title.
    """

    # Create the dropdown widget
    year_dropdown = widgets.Dropdown(
        options=sorted(df['year'].unique()),
        value=df['year'].min(),
        description='Year:',
        style={'description_width': 'initial'}
    )

    # Define the update function
    def update_map(year):
        for quarter in sorted(df['quarter'].unique()):
            quarter_data = df[(df['quarter'] == quarter) & (df['year'] == year)]

            fig = px.choropleth(
                quarter_data,
                locations='state_code',
                locationmode='USA-states',
                color=value_column,
                hover_name='state',
                title=f'{title_prefix} per State on US Map (Year {year}, Quarter {quarter})',
                scope='usa',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(
                title_x=0.5,
                title_font=dict(size=20)
            )

            fig.show()

    # Link the dropdown to the update function
    year_dropdown.observe(lambda change: update_map(change['new']), names='value')

    # Display the dropdown and render initial map
    display(year_dropdown)
    update_map(year_dropdown.value)


def fetch_weather_data(locations_df, start_date="2015-01-01", end_date="2023-01-01"):
    """
    Fetch weather data for multiple locations using the Open-Meteo API.

    Args:
        locations_df (pd.DataFrame): A DataFrame with columns 'latitude' and 'longitude'.
        start_date (str): Start date for the weather data (format: YYYY-MM-DD).
        end_date (str): End date for the weather data (format: YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame containing weather data for all locations.
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Base URL for the Open-Meteo API
    url = "https://archive-api.open-meteo.com/v1/archive"

    # List to store weather data for all locations
    all_weather_data = []

    # Iterate over each location in the DataFrame
    for _, row in locations_df.iterrows():
        latitude = row['latitude']
        longitude = row['longitude']

        # Define API parameters
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "wind_speed_10m_max", "weather_code", "temperature_2m_mean",
                "temperature_2m_max", "temperature_2m_min", "precipitation_hours",
                "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min"
            ]
        }

        # Fetch weather data for the current location
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Process daily data
        daily = response.Daily()
        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "latitude": latitude,
            "longitude": longitude,
            "wind_speed_10m_max": daily.Variables(0).ValuesAsNumpy(),
            "weather_code": daily.Variables(1).ValuesAsNumpy(),
            "temperature_2m_mean": daily.Variables(2).ValuesAsNumpy(),
            "temperature_2m_max": daily.Variables(3).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(4).ValuesAsNumpy(),
            "precipitation_hours": daily.Variables(5).ValuesAsNumpy(),
            "relative_humidity_2m_mean": daily.Variables(6).ValuesAsNumpy(),
            "relative_humidity_2m_max": daily.Variables(7).ValuesAsNumpy(),
            "relative_humidity_2m_min": daily.Variables(8).ValuesAsNumpy(),
        }

        # Convert to DataFrame and append to the list
        location_weather_df = pd.DataFrame(data=daily_data)
        all_weather_data.append(location_weather_df)

    # Combine all location DataFrames into a single DataFrame
    combined_weather_data = pd.concat(all_weather_data, ignore_index=True)
    return combined_weather_data

# Bee/nature-inspired color palettes
bee_colony_palette = ['#F4A300', '#A66E00', '#654321']  # Colony features
bee_stressor_palette = ['#D1B000', '#8B8C00', '#4F6F00', '#CBBF7A', '#A58F5D'] # Stressors

def get_quarter_start_date(row):
    # Return the first date of the quarter based on the year and quarter
    quarter_map = {
        1: '01-01',  # Q1 -> January 1st
        2: '04-01',  # Q2 -> April 1st
        3: '07-01',  # Q3 -> July 1st
        4: '10-01'   # Q4 -> October 1st
    }
    return pd.to_datetime(f"{row['year']}-{quarter_map[row['quarter']]}")

# Function to analyze the time series for a specific state
def analyze_state_data(bees, state_name):
    # Apply the function to create a new datetime column
    bees['date'] = bees.apply(get_quarter_start_date, axis=1)

    # Set the datetime column as the index
    bees.set_index('date', inplace=True)

    # Ensure data is sorted by date
    bees = bees.sort_index()

    # Filter for the specified state and drop rows with missing values in 'num_colonies'
    bees_state = bees[bees['state'] == state_name].dropna(subset=['num_colonies'])

    # Explicitly set the frequency to quarterly data (quarterly start)
    bees_state.index = pd.to_datetime(bees_state.index)

    # Apply Simple Exponential Smoothing (SES)
    ses_model = SimpleExpSmoothing(bees_state['num_colonies'])
    ses_fit = ses_model.fit()

    # Apply Exponential Smoothing (Holt-Winters) with additive trend and seasonality
    holt_winters_model_additive = ExponentialSmoothing(bees_state['num_colonies'], trend='add', seasonal='add', seasonal_periods=4)
    holt_winters_fit_additive = holt_winters_model_additive.fit()

    # Apply Exponential Smoothing (Holt-Winters) with multiplicative trend and additive seasonality
    holt_winters_model_multiplicative_trend = ExponentialSmoothing(bees_state['num_colonies'], trend='mul', seasonal='add', seasonal_periods=4)
    holt_winters_fit_multiplicative_trend = holt_winters_model_multiplicative_trend.fit()

    # Apply Exponential Smoothing (Holt-Winters) with additive trend and multiplicative seasonality
    holt_winters_model_additive_seasonality = ExponentialSmoothing(bees_state['num_colonies'], trend='add', seasonal='mul', seasonal_periods=4)
    holt_winters_fit_additive_seasonality = holt_winters_model_additive_seasonality.fit()

    # Apply Exponential Smoothing (Holt-Winters) with multiplicative trend and seasonality
    holt_winters_model_multiplicative_seasonality = ExponentialSmoothing(bees_state['num_colonies'], trend='mul', seasonal='mul', seasonal_periods=4)
    holt_winters_fit_multiplicative_seasonality = holt_winters_model_multiplicative_seasonality.fit()

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Plot the original data alongside all smoothing methods
    plt.figure(figsize=(12, 6))

    # Use sns.lineplot for each series
    sns.lineplot(x=bees_state.index, y=bees_state['num_colonies'], label='Original Data', color='black')
    sns.lineplot(x=bees_state.index, y=ses_fit.fittedvalues, label='Simple Exponential Smoothing (SES)', color='blue')
    sns.lineplot(x=bees_state.index, y=holt_winters_fit_additive.fittedvalues, label='Exponential Smoothing (Holt-Winters - Additive)', color='green')
    sns.lineplot(x=bees_state.index, y=holt_winters_fit_multiplicative_trend.fittedvalues, label='Exponential Smoothing (Holt-Winters - Mul Trend)', color='red')
    sns.lineplot(x=bees_state.index, y=holt_winters_fit_additive_seasonality.fittedvalues, label='Exponential Smoothing (Holt-Winters - Additive Seasonality)', color='purple')
    sns.lineplot(x=bees_state.index, y=holt_winters_fit_multiplicative_seasonality.fittedvalues, label='Exponential Smoothing (Holt-Winters - Mul Seasonality)', color='orange')

    # Customize plot
    plt.title(f'Time Series with Smoothing Methods for "num_colonies" ({state_name})', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('num_colonies', fontsize=12)

    # Set x-axis limits to zoom from 2015-Q1 to 2018-Q4
    plt.xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2018-12-31'))

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add a legend
    plt.legend(loc='best')

    # Display grid
    plt.grid(True)

    # Show plot
    plt.show()

# Example usage: analyze data for 'New York'
# analyze_state_data(bees, 'New York')

def plot_num_colonies_over_time(dataset, state):
    plt.figure(figsize=(12, 6))
    plt.plot(dataset[dataset['state'] == state].index, dataset[dataset['state'] == state]['num_colonies'], label='Num Colonies')
    plt.title(f'Number of Bee Colonies Over Time in {state}')
    plt.xlabel('Date')
    plt.ylabel('Number of Bee Colonies')
    plt.legend()
    plt.show()


def heatmap_colonies_over_time(bees, state):
    """
    Creates a heatmap for num_colonies over time for a specific US state,
    with dynamically colored annotations for legibility.
    """

    # Filter for the selected state
    state_data = bees[bees['state'] == state].copy()

    # Create a 'Quarter' label
    state_data['Quarter'] = 'Q' + state_data['quarter'].astype(str)

    # Pivot table: rows = years, columns = quarters
    cm_data = state_data.pivot(index='year', columns='Quarter', values='num_colonies')

    # Reorder columns to Q1–Q4
    cm_data = cm_data.reindex(columns=['Q1', 'Q2', 'Q3', 'Q4'])

    # Create figure and heatmap
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap('YlGnBu')
    norm = Normalize(vmin=np.nanmin(cm_data.values), vmax=np.nanmax(cm_data.values))

    ax = sns.heatmap(
        cm_data,
        annot=False,
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={'label': 'Number of Colonies'}
    )

    # Add annotations with dynamic text color
    for i in range(cm_data.shape[0]):
        for j in range(cm_data.shape[1]):
            val = cm_data.iloc[i, j]
            if pd.notnull(val):
                # Normalize value and get background color
                bg_color = cmap(norm(val))
                # Compute luminance (perceived brightness)
                luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                text_color = 'black' if luminance > 0.5 else 'white'
                plt.text(j + 0.5, i + 0.5, f"{int(val):,}",
                         ha='center', va='center',
                         fontsize=12, color=text_color)

    # Titles and labels
    plt.title(f'Number of Bee Colonies Over Time in {state}', fontsize=14)
    plt.xlabel('Quarter')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.show()
