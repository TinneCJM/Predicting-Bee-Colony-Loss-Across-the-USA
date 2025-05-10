import requests
import pandas as pd
import os
import numpy as np
from datetime import datetime, date
import time
import openmeteo_requests
import requests_cache
from retry_requests import retry
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import cm as mpl_cm
from matplotlib.colors import Normalize
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import statsmodels.api as sm
from matplotlib.ticker import ScalarFormatter
from statsmodels.tsa.stattools import adfuller

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

    # Filter for the specified state and drop rows with missing values in 'percent_lost'
    bees_state = bees[bees['state'] == state_name].dropna(subset=['percent_lost'])

    # Explicitly set the frequency to quarterly data (quarterly start)
    bees_state.index = pd.to_datetime(bees_state.index)

    # Apply Simple Exponential Smoothing (SES)
    ses_model = SimpleExpSmoothing(bees_state['percent_lost'])
    ses_fit = ses_model.fit()

    # Apply Exponential Smoothing (Holt-Winters) with additive trend and seasonality
    holt_winters_model_additive = ExponentialSmoothing(bees_state['percent_lost'], trend='add', seasonal='add', seasonal_periods=4)
    holt_winters_fit_additive = holt_winters_model_additive.fit()

    # Apply Exponential Smoothing (Holt-Winters) with multiplicative trend and additive seasonality
    holt_winters_model_multiplicative_trend = ExponentialSmoothing(bees_state['percent_lost'], trend='mul', seasonal='add', seasonal_periods=4)
    holt_winters_fit_multiplicative_trend = holt_winters_model_multiplicative_trend.fit()

    # Apply Exponential Smoothing (Holt-Winters) with additive trend and multiplicative seasonality
    holt_winters_model_additive_seasonality = ExponentialSmoothing(bees_state['percent_lost'], trend='add', seasonal='mul', seasonal_periods=4)
    holt_winters_fit_additive_seasonality = holt_winters_model_additive_seasonality.fit()

    # Apply Exponential Smoothing (Holt-Winters) with multiplicative trend and seasonality
    holt_winters_model_multiplicative_seasonality = ExponentialSmoothing(bees_state['percent_lost'], trend='mul', seasonal='mul', seasonal_periods=4)
    holt_winters_fit_multiplicative_seasonality = holt_winters_model_multiplicative_seasonality.fit()

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Plot the original data alongside all smoothing methods
    plt.figure(figsize=(12, 6))

    # Use sns.lineplot for each series
    sns.lineplot(x=bees_state.index, y=bees_state['percent_lost'], label='Original Data', color='black')
    sns.lineplot(x=bees_state.index, y=ses_fit.fittedvalues, label='Simple Exponential Smoothing (SES)', color='blue')
    sns.lineplot(x=bees_state.index, y=holt_winters_fit_additive.fittedvalues, label='Exponential Smoothing (Holt-Winters - Additive)', color='green')
    sns.lineplot(x=bees_state.index, y=holt_winters_fit_multiplicative_trend.fittedvalues, label='Exponential Smoothing (Holt-Winters - Mul Trend)', color='red')
    sns.lineplot(x=bees_state.index, y=holt_winters_fit_additive_seasonality.fittedvalues, label='Exponential Smoothing (Holt-Winters - Additive Seasonality)', color='purple')
    sns.lineplot(x=bees_state.index, y=holt_winters_fit_multiplicative_seasonality.fittedvalues, label='Exponential Smoothing (Holt-Winters - Mul Seasonality)', color='orange')

    # Customize plot
    plt.title(f'Time Series with Smoothing Methods for "percent_lost" ({state_name})', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('percent_lost', fontsize=12)

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

def plot_percent_lost_over_time(dataset, state):
    plt.figure(figsize=(12, 6))
    plt.plot(dataset[dataset['state'] == state].index, dataset[dataset['state'] == state]['percent_lost'], label='Num Colonies')
    plt.title(f'Number of Bee Colonies Over Time in {state}')
    plt.xlabel('Date')
    plt.ylabel('Number of Bee Colonies')
    plt.legend()
    plt.show()


def heatmap_colonies_over_time(bees, state):
    """
    Creates a heatmap for percent_lost over time for a specific US state,
    with dynamically colored annotations for legibility.
    """

    # Filter for the selected state
    state_data = bees[bees['state'] == state].copy()

    # Create a 'Quarter' label
    state_data['Quarter'] = 'Q' + state_data['quarter'].astype(str)

    # Pivot table: rows = years, columns = quarters
    cm_data = state_data.pivot(index='year', columns='Quarter', values='percent_lost')

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


def plot_bee_colony_trends(bees, state):
    # Filter data for the selected state and create time column
    state_data = bees[bees['state'] == state].copy()
    state_data['time'] = state_data['year'].astype(str) + ' Q' + state_data['quarter'].astype(str)

    # Sort time
    state_data = state_data.sort_values(by=['year', 'quarter'])

    # Get middle color from Viridis colormap
    viridis_middle_color = cm.viridis(0.5)  # RGBA tuple
    hex_color = mcolors.to_hex(viridis_middle_color)  # Convert RGBA to Hex

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(state_data['time'], state_data['percent_lost'], marker='o', color=hex_color, label='Percent Lost Colonies')

    # Customize plot
    plt.title(f'Percent Lost Bee Colonies Over Time in {state}', fontsize=14)
    plt.xlabel('Time (Year - Quarter)', labelpad=15)
    plt.ylabel('Percent Lost Colonies', labelpad=15)
    plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels
    plt.ticklabel_format(style='plain', axis='y')  # Set y-axis to plain notation
    plt.grid(True, color='lightgray', linestyle='--', linewidth=0.6)  # Lighten gridlines
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_percent_loss_by_quarter(bees, state):
    """
    Plots percent colony loss by quarter for a given state.

    Parameters:
        bees (pd.DataFrame): The full bees dataset.
        state (str): The state to filter for.
    """
    state_data = bees[bees['state'] == state]

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='quarter', y='percent_lost', data=state_data, color='yellow')
    plt.title(f'Percent Lost Bee Colonies by Quarter in {state}')
    plt.xlabel('Quarter')
    plt.ylabel('Percent Lost Bee Colonies')
    plt.tight_layout()
    plt.show()


def plot_stressors_by_quarter(bees, state, bee_stressor_palette=None):
    """
    Plots stressors affecting bee colonies by quarter for a given state.

    Parameters:
        bees (pd.DataFrame): The full bees dataset.
        state (str): The state to filter for.
        bee_stressor_palette (dict, optional): Custom color palette for stressors.
    """
    stressors = ['varroa_mites', 'other_pests_and_parasites', 'diseases', 'pesticides', 'other_or_unknown']
    state_data = bees[bees['state'] == state]

    melted_stressors = state_data.melt(
        id_vars=['quarter'],
        value_vars=stressors,
        var_name='Stressor',
        value_name='Value'
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='quarter', y='Value', hue='Stressor', data=melted_stressors, palette=bee_stressor_palette)
    plt.title(f'Stressors by Quarter in {state}')
    plt.xlabel('Quarter')
    plt.ylabel('Percentage of Colonies Affected')
    plt.legend(title='Stressor', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_drought_by_quarter(bees, state):
    """
    Plots drought severity levels by quarter for a given state.

    Parameters:
        bees (pd.DataFrame): The full bees dataset.
        state (str): The state to filter for.
    """
    drought_cols = [
        'D0_mean', 'D1_mean', 'D2_mean', 'D3_mean', 'D4_mean',
        'D0_max', 'D1_max', 'D2_max', 'D3_max', 'D4_max'
    ]

    # Check for presence of required columns
    missing = [col for col in drought_cols if col not in bees.columns]
    if missing:
        raise ValueError(f"The following drought columns are missing: {missing}")

    # Filter and reshape
    state_data = bees[bees['state'] == state]
    melted_drought = state_data.melt(
        id_vars='quarter',
        value_vars=drought_cols,
        var_name='Drought_Level',
        value_name='Value'
    )

    # Plot
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='quarter', y='Value', hue='Drought_Level', data=melted_drought)
    plt.title(f'Drought Severity by Quarter in {state}')
    plt.xlabel('Quarter')
    plt.ylabel('Severity Index')
    plt.legend(title='Drought Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_weather_conditions_by_quarter(bees, state):
    """
    Plots weather condition sums by quarter for a given state.

    Parameters:
        bees (pd.DataFrame): The full bees dataset.
        state (str): The state to filter for.
    """
    weather_cols = [
        'moderate_drizzle_sum', 'moderate_rain_sum', 'light_rain_sum', 'heavy_rain_sum',
        'overcast_sum', 'partly_cloudy_sum', 'clear_sky_sum', 'light_drizzle_sum',
        'mainly_clear_sum', 'heavy_drizzle_sum', 'light_snow_sum',
        'heavy_snow_sum', 'moderate_snow_sum'
    ]

    # Check for presence of columns
    missing = [col for col in weather_cols if col not in bees.columns]
    if missing:
        raise ValueError(f"The following required columns are missing: {missing}")

    # Filter and reshape data
    state_data = bees[bees['state'] == state]
    melted_weather = state_data.melt(
        id_vars='quarter',
        value_vars=weather_cols,
        var_name='Condition',
        value_name='Sum_Hours'
    )

    # Plot
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='quarter', y='Sum_Hours', hue='Condition', data=melted_weather)
    plt.title(f'Weather Conditions by Quarter in {state}')
    plt.xlabel('Quarter')
    plt.ylabel('Sum of Hours')
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


   
def plot_precipitation_by_quarter(bees, state):
    """
    Plots precipitation hours by quarter for a given state.

    Parameters:
        bees (pd.DataFrame): The full bees dataset.
        state (str): The state to filter for.
    """
    if 'precipitation_hours_sum' not in bees.columns:
        raise ValueError("'precipitation_hours_sum' column not found in dataset.")

    state_data = bees[bees['state'] == state]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='quarter', y='precipitation_hours_sum', data=state_data, color='lightblue')
    plt.title(f'Precipitation Hours by Quarter in {state}')
    plt.xlabel('Quarter')
    plt.ylabel('Precipitation Hours (Sum)')
    plt.tight_layout()
    plt.show()


def plot_temperature_features_by_quarter(bees, state):
    """
    Plots temperature-related features by quarter for a given state.

    Parameters:
        bees (pd.DataFrame): The full bees dataset.
        state (str): The state to filter for.
    """
    temp_features = [
        'temperature_2m_meanmean', 'temperature_2m_meansum',
        'temperature_2m_maxmax', 'temperature_2m_minmin'
    ]
    state_data = bees[bees['state'] == state]

    melted_temp = state_data.melt(
        id_vars=['quarter'],
        value_vars=[col for col in temp_features if col in state_data.columns],
        var_name='Temperature Metric',
        value_name='Value'
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='quarter', y='Value', hue='Temperature Metric', data=melted_temp)
    plt.title(f'Temperature Metrics by Quarter in {state}')
    plt.xlabel('Quarter')
    plt.ylabel('Temperature (°C)')
    plt.legend(title='Temperature Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_colony_timeseries(state_bees, column, state, rolling_window=4):
    """
    Plots time series data with rolling mean and standard deviation.

    Parameters:
        state_bees (DataFrame): Data containing 'date' and the column to analyze.
        column (str): The name of the column to plot (e.g., 'num_colonies').
        state (str): The name of the state for the plot title.
        rolling_window (int): Number of periods for the rolling window. Default is 4.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    sns.lineplot(x=state_bees['date'], y=state_bees[column], ax=ax, color='dodgerblue', label='raw data')
    sns.lineplot(x=state_bees['date'], y=state_bees[column].rolling(rolling_window).mean(), ax=ax, color='black', label='rolling mean')
    sns.lineplot(x=state_bees['date'], y=state_bees[column].rolling(rolling_window).std(), ax=ax, color='orange', label='rolling std')

    ax.set_title(f'{column.replace("_", " ").title()} for {state}: Non-stationary \nCheck if data is stationary or not', fontsize=14)
    ax.set_ylabel(column.replace("_", " ").title(), fontsize=14)
    ax.set_xlim([date(2015, 1, 1), date(2022, 10, 1)])

    # Disable scientific notation on y-axis
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.show()    

def adf_test(series, state):
    """
    Perform Augmented Dickey-Fuller test and return a Series with results.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pd.Series(result[0:4], index=labels)

    # Add critical values
    for key, val in result[4].items():
        out[f'critical value ({key})'] = val

    # Add stationary flag
    out['stationary'] = result[1] <= 0.05
    
    return out


# subset dataframe to only have rows of state of interest
def subset_by_state(state_bees, state):
    """
    Returns rows from the DataFrame corresponding to a given U.S. state.
    If the state is not found, prints a message and returns None.
    """
    if state not in state_bees['state'].unique():
        print(f"Information for the state '{state}' is unavailable.")
        return None
    return state_bees[state_bees['state'] == state]


def create_quarterly_index(state_bees):
    """
    Converts 'year' and 'quarter' columns in a DataFrame to a datetime index
    representing the start of each quarter.

    Returns a DataFrame with the new index.
    """
    if not {'year', 'quarter'}.issubset(state_bees.columns):
        raise ValueError("DataFrame must contain 'year' and 'quarter' columns.")
    
    # Create a quarterly period index
    state_bees = state_bees.copy()
    state_bees['date'] = pd.to_datetime(state_bees['year'].astype(str) + 'Q' + state_bees['quarter'].astype(str))
    return state_bees