from datetime import date

import numpy as np
import pandas as pd
import requests_cache
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from cycler import cycler

import openmeteo_requests
from retry_requests import retry
import plotly.express as px
import ipywidgets as widgets
from IPython.display import clear_output, display
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller


def set_bee_style():
    """
    Configure Matplotlib global rcParams with a bee-inspired palette and grid styling.
    """
    bee_colors = [
        "#F2C94C", "#E91E63", "#3498DB", "#7B3F00", "#e39db4",
        "#2ECC71", "#2C3E50", "#9B59B6", "#027031", "#E67E22",
        "#FF6F61", "#F39C12", "#F4D03F", "#8E44AD", "#95A5A6"
    ]
    mpl.rcParams['axes.prop_cycle'] = cycler('color', bee_colors)
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.color'] = '0.5'
    mpl.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.frameon'] = False




def interactive_choropleth_by_year(
    df: pd.DataFrame,
    value_column: str,
    title_prefix: str = 'Number of Colonies'
) -> None:
    """
    Display a choropleth grid by quarter for a selected year.

    Args:
        df: DataFrame with columns ['year', 'quarter', 'state_code', value_column].
        value_column: Column to color by.
        title_prefix: Prefix for the map title.
    """
    years = sorted(df['year'].dropna().unique())

    year_dropdown = widgets.Dropdown(
        options=years,
        value=years[0],
        description='Year:',
        style={'description_width': 'initial'}
    )

    out = widgets.Output()

    def update(year):
        with out:
            clear_output(wait=True)
            sub = df.loc[df['year'] == year].copy() 
            sub['Q'] = sub['quarter'].apply(lambda q: f"Q{q}")

            fig = px.choropleth(
                sub,
                locations='state_code',
                locationmode='USA-states',
                color=value_column,
                facet_col='Q',
                facet_col_wrap=2,
                scope='usa',
                color_continuous_scale='Viridis',
                title=f"{title_prefix} — {year}",
                labels={value_column: value_column}
            )
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig.show()

    # Redraw whenever the dropdown changes
    year_dropdown.observe(lambda change: update(change.new), names='value')

    display(year_dropdown, out)
    update(years[0])





def fetch_weather_data(
    locations_df: pd.DataFrame,
    start_date: str = "2015-01-01",
    end_date: str = "2023-01-01"
) -> pd.DataFrame:
    """
    Fetch daily weather data for multiple locations via Open-Meteo.

    Args:
        locations_df: DataFrame with ['latitude', 'longitude'] columns.
        start_date: ISO date string for start.
        end_date: ISO date string for end.

    Returns:
        Combined weather DataFrame for all locations.
    """

    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"

    all_weather_data = []

    for _, row in locations_df.iterrows():
        latitude = row['latitude']
        longitude = row['longitude']

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

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

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

        location_weather_df = pd.DataFrame(data=daily_data)
        all_weather_data.append(location_weather_df)

    combined_weather_data = pd.concat(all_weather_data, ignore_index=True)
    return combined_weather_data




def get_quarter_start_date(row: pd.Series) -> pd.Timestamp:
    """
    Return the Timestamp for the first day of the quarter given 'year' and 'quarter'.

    Args:
        row: Series with 'year' and 'quarter' ints.

    Returns:
        pandas.Timestamp for quarter start.
    """

    quarter_map = {
        1: '01-01',  # Q1 -> January 1st
        2: '04-01',  # Q2 -> April 1st
        3: '07-01',  # Q3 -> July 1st
        4: '10-01'   # Q4 -> October 1st
    }
    return pd.to_datetime(f"{row['year']}-{quarter_map[row['quarter']]}")




def analyze_state_data(bees: pd.DataFrame, state_name: str) -> None:
    """
    Analyze and visualize the 'percent_lost' time series for a given state using smoothing methods.

    This function:
      1. Converts 'year' and 'quarter' into a datetime index.
      2. Filters the data for the specified state.
      3. Fits multiple smoothing models:
         - Simple Exponential Smoothing (SES)
         - Holt-Winters (additive trend & seasonality)
         - Holt-Winters (multiplicative trend, additive seasonality)
         - Holt-Winters (additive trend, multiplicative seasonality)
         - Holt-Winters (multiplicative trend & seasonality)
      4. Plots the original 'percent_lost' series alongside each model's fitted values.

    Parameters
    ----------
    bees : pd.DataFrame
        Must contain columns 'year', 'quarter', 'state', and 'percent_lost'.
    state_name : str
        The state to analyze (filters bees['state'] == state_name).

    Returns
    -------
    Displays a Matplotlib plot of the original and smoothed series.
    """
    bees['date'] = bees.apply(get_quarter_start_date, axis=1)
    bees.set_index('date', inplace=True)
    bees = bees.sort_index()

    bees_state = bees[bees['state'] == state_name].dropna(subset=['percent_lost'])

    ses_model = SimpleExpSmoothing(bees_state['percent_lost']).fit()
    hw_add  = ExponentialSmoothing(bees_state['percent_lost'], trend='add', seasonal='add', seasonal_periods=4).fit()
    hw_mul_trend = ExponentialSmoothing(bees_state['percent_lost'], trend='mul', seasonal='add', seasonal_periods=4).fit()
    hw_add_seas  = ExponentialSmoothing(bees_state['percent_lost'], trend='add', seasonal='mul', seasonal_periods=4).fit()
    hw_mul_seas  = ExponentialSmoothing(bees_state['percent_lost'], trend='mul', seasonal='mul', seasonal_periods=4).fit()

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.plot(bees_state.index, bees_state['percent_lost'], label='Original Data', color='black')
    plt.plot(bees_state.index, ses_model.fittedvalues, label='SES', color='blue')
    plt.plot(bees_state.index, hw_add.fittedvalues, label='Holt-Winters Add', color='green')
    plt.plot(bees_state.index, hw_mul_trend.fittedvalues, label='HW Mul Trend', color='red')
    plt.plot(bees_state.index, hw_add_seas.fittedvalues, label='HW Add Season', color='purple')
    plt.plot(bees_state.index, hw_mul_seas.fittedvalues, label='HW Mul Season', color='orange')

    plt.title(f'Time Series Smoothing Comparison for "{state_name}"', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('percent_lost', fontsize=12)
    plt.xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2018-12-31'))
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()






def plot_percent_lost_over_time(dataset, state):
    plt.figure(figsize=(12, 6))
    plt.plot(dataset[dataset['state'] == state].index, dataset[dataset['state'] == state]['percent_lost'], label='Num Colonies')
    plt.title(f'Number of Bee Colonies Over Time in {state}')
    plt.xlabel('Date')
    plt.ylabel('Number of Bee Colonies')
    plt.legend()
    plt.show()




def heatmap_percent_lost_over_time(
    bees: pd.DataFrame,
    state: str
) -> None:
    """
    Plot a heatmap of the 'percent_lost' metric over years and quarters for a given state.

    This function:
      1. Filters the DataFrame for the specified state.
      2. Creates a 'Quarter' label (Q1–Q4) from the 'quarter' column.
      3. Pivots the data so rows are years and columns are quarters, values are 'percent_lost'.
      4. Draws a heatmap with colorbar labeled 'Percent Lost (%)'.
      5. Annotates each cell with the percent lost, adjusting text color for legibility.

    Parameters
    ----------
    bees : pd.DataFrame
        Must contain columns 'state', 'year', 'quarter', and 'percent_lost'.
    state : str
        The two-letter state code or full state name to filter the data.

    Returns
    -------
    Displays the heatmap directly.
    """
    state_data = bees[bees['state'] == state].copy()

    state_data['Quarter'] = 'Q' + state_data['quarter'].astype(str)

    cm_data = state_data.pivot(index='year', columns='Quarter', values='percent_lost')
    cm_data = cm_data.reindex(columns=['Q1', 'Q2', 'Q3', 'Q4'])

    vmin = np.nanmin(cm_data.values)
    vmax = np.nanmax(cm_data.values)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('YlGnBu')

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm_data,
        annot=False,
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={'label': 'Percent Lost (%)'}
    )

    for i in range(cm_data.shape[0]):
        for j in range(cm_data.shape[1]):
            val = cm_data.iloc[i, j]
            if pd.notnull(val):
                bg = cmap(norm(val))
                luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                text_color = 'black' if luminance > 0.5 else 'white'
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{val:.1f}%",
                    ha='center', va='center',
                    color=text_color, fontsize=10
                )

    plt.title(f'Percent Lost Over Time in {state}', fontsize=14)
    plt.xlabel('Quarter')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.show()




def plot_bee_colony_trends(bees: pd.DataFrame, state: str) -> None:
    """
    Plot the time-series trend of percent lost bee colonies for a given state.

    This function:
      1. Filters the full `bees` DataFrame to the specified state.
      2. Constructs a human-readable `time` column in "YYYY Q#" format.
      3. Sorts by chronological order (year, then quarter).
      4. Uses the midpoint of the Viridis colormap for the line color.
      5. Plots `percent_lost` against `time` with customization for readability.

    Parameters
    ----------
    bees : pd.DataFrame
        Must contain columns:
          - 'state' (str)               : state identifier
          - 'year' (int)                : calendar year
          - 'quarter' (int, 1–4)        : quarter number
          - 'percent_lost' (float/int)  : percent of colonies lost
    state : str
        The state to visualize (matches values in `bees['state']`).

    Returns
    -------
    Displays a Matplotlib line plot of percent lost over time.
    """

    state_data = bees[bees['state'] == state].copy()
    state_data['time'] = (
        state_data['year'].astype(str)
        + ' Q'
        + state_data['quarter'].astype(str)
    )

    state_data = state_data.sort_values(by=['year', 'quarter'])

    viridis_middle_color = cm.viridis(0.5)           
    hex_color = mcolors.to_hex(viridis_middle_color) 

    plt.figure(figsize=(15, 6))
    plt.plot(
        state_data['time'],
        state_data['percent_lost'],
        marker='o',
        color=hex_color,
        label='Percent Lost Colonies'
    )

    plt.title(f'Percent Lost Bee Colonies Over Time in {state}', fontsize=14)
    plt.xlabel('Time (Year - Quarter)', labelpad=15)
    plt.ylabel('Percent Lost Colonies', labelpad=15)
    plt.xticks(rotation=90, fontsize=10)
    plt.ticklabel_format(style='plain', axis='y')
    plt.grid(True, color='lightgray', linestyle='--', linewidth=0.6)
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

    missing = [col for col in drought_cols if col not in bees.columns]
    if missing:
        raise ValueError(f"The following drought columns are missing: {missing}")

    state_data = bees[bees['state'] == state]
    melted_drought = state_data.melt(
        id_vars='quarter',
        value_vars=drought_cols,
        var_name='Drought_Level',
        value_name='Value'
    )

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

    missing = [col for col in weather_cols if col not in bees.columns]
    if missing:
        raise ValueError(f"The following required columns are missing: {missing}")

    state_data = bees[bees['state'] == state]
    melted_weather = state_data.melt(
        id_vars='quarter',
        value_vars=weather_cols,
        var_name='Condition',
        value_name='Sum_Hours'
    )


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
        'temperature_2m_mean', 
        'temperature_2m_max', 
        'temperature_2m_min'
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





def plot_humidity_features_by_quarter(bees: pd.DataFrame, state: str) -> None:
    """
    Plots humidity-related features by quarter for a given state.

    Parameters:
    -----------
    bees : pd.DataFrame
        The full dataset, must contain columns 'state' and 'quarter', 
        as well as the humidity features.
    state : str
        The state to filter for.
    """
    humidity_features = [
        'relative_humidity_2m_mean',
        'relative_humidity_2m_max',
        'relative_humidity_2m_min'
    ]

    state_data = bees[bees['state'] == state]

    features_present = [col for col in humidity_features if col in state_data.columns]

    melted_hum = state_data.melt(
        id_vars=['quarter'],
        value_vars=features_present,
        var_name='Humidity Metric',
        value_name='Value'
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='quarter',
        y='Value',
        hue='Humidity Metric',
        data=melted_hum
    )
    plt.title(f'Humidity Metrics by Quarter in {state}')
    plt.xlabel('Quarter')
    plt.ylabel('Relative Humidity (%)')
    plt.legend(title='Humidity Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
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




def subset_by_state(df: pd.DataFrame, state: str) -> pd.DataFrame | None:
    """
    Return rows of DataFrame where df['state']==state or None if missing.
    """
    if state not in df['state'].unique():
        print(f"State '{state}' not found.")
        return None
    return df[df['state'] == state]





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