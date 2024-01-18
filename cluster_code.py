# -*- coding: utf-8 -*-
"""
A python program that create a poster that showcases interesting clusters of 
data using clustering methods and fits models to datasets using curve fitting
"""

# Import the neccessary libraries
import numpy as np                     # 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm                     # Statistics
from sklearn.cluster import KMeans               # Clustering 
from scipy.optimize import curve_fit             # Fitting

from cluster_tools import map_corr, scaler     # Tools to support clustering
from errors import error_prop     # error functions from errors.py

# Defining a function that loads, clean and return two dataframes
def read_data(filename):
    """
    This Function reads the csv file, cleans the data, transpose and returns 
    two dataframes as output
    
    Example:
    
        df1, df2 = read_data('mydata.csv')
    
    """
    # Load the data
    wdi_data = pd.read_csv(filename)
    
    # Clean data by replace non-number with pd.NA
    wdi_data = wdi_data.replace('..', pd.NA)
    
    # Handle Missing Values
    wdi_data = wdi_data.fillna(method='ffill').fillna(method='bfill')

    # Drop Duplicates
    wdi_data = wdi_data.drop_duplicates()

    # Remove Meta Data Rows
    wdi_data.drop(wdi_data.index[1519:], inplace=True)

    # Drop Un-needed Column
    wdi_data.drop(["Country Code", "Series Code"], axis=1, inplace=True)

    # Convert the Year Columns to float data type
    wdi_data[['1980', '1985', '1990', '1995', 
              '2000', '2005', '2010', '2015', '2020']] = wdi_data[['1980', '1985', '1990', '1995', 
              '2000', '2005', '2010', '2015', '2020']].astype(float)

    # Transpose the Data to have indicators as columns 
    # Melt the = dataframe to have 'YEAR' as a column
    wdi_transposed = pd.melt(wdi_data,
                             id_vars=['Series Name','Country Name'],
                             var_name='Year',
                             value_name='Value')

    # Pivot the melted dataframe to have individual columns
    # for each country
    wdi_transposed = wdi_transposed.pivot_table(
        index=['Year', 'Country Name'],
        columns='Series Name',
        values='Value',
        aggfunc='first')

    # Reset index for a clean structure
    wdi_transposed.reset_index(inplace=True)

    return wdi_data, wdi_transposed

df1, df2 = read_data('cluster_data.csv')

indicator_list = df1['Series Name'].unique()

# Function to generate visualizations telling a visual story of the data
def visualize(data_frame, indicators):
    """
    Data Visualization
  
    Args:
      data_frame: The dataframe to be visualized.
      indicators: The list of indicators to be explored.
      
    """
    # Plot bar chart for top 10 countries for each indicator in 2020
    for indicator in indicators:
        plot_data = data_frame[data_frame['Year'] == '2020'].copy()

        # Assign ranks based on CO2 emissions (kt)
        plot_data.loc[:, 'Rank'] = plot_data[indicator].rank(ascending=False)
        
        # Select the top 10 Emitting Countries
        plot_data = plot_data.sort_values('Rank').head(10)
        
        ax = sns.barplot(x=indicator, y='Country Name', data=plot_data,
                          order=plot_data.sort_values('Rank')['Country Name'],
                          palette='Blues')
        
        title_text = indicator.split('(')[0]
        
        ax.set_title(f'Top 10 Countries: {title_text}',
                             color='#0f484f', fontsize=15, fontweight='bold')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        
        plt.show()  
        
    # Plot Line

visualize(df2, indicator_list)

# Correlation using the map_corr function from cluster_tools
correlation_map = map_corr(df2.select_dtypes(include=np.number))

# Function to Cluster data 
def cluster_data(df, n_clusters=4):
    """
    K-means clustering.
  
    Args:
      df: The dataframe to be clustered.
      n_clusters: The number of clusters to generate.
  
    Returns:
      A dataframe with the cluster labels.
  """
    # Create a copy of the Dataframe to use
    cluster_df = df.copy()
  
    # Normalize data using the scaler function from cluster tools
    normalized_data, data_min, data_max = scaler(cluster_df.select_dtypes(include=np.number))
    
    # Create a KMeans model.
    model = KMeans(n_clusters=n_clusters, random_state=0)
  
    # Fit the model to the data.
    model.fit(normalized_data)
    
    # Get the cluster labels.
    cluster_labels = model.labels_
    
    # Add the cluster labels to the dataset
    cluster_df["Cluster"] = cluster_labels
    
    return cluster_df, normalized_data
    
c_data, n_data = cluster_data(df2)

# Function to visualize the relationship between clusters
def visualize_clusters(clustered_data, cluster_column, indicators):
    """
    Create visualizations for the clusters.

    Args:
        clustered_data: The dataframe with cluster labels.
        cluster_column: The name of the column containing cluster labels.
        indicators: The list of indicators to be explored.

    Returns:
        None
    """
    # Set the style for the plots
    sns.set(style="darkgrid")

    # Loop through each indicator to create visualizations
    for indicator in indicators:
        plt.figure(figsize=(12, 8))
        
        # Create a boxplot for each cluster
        sns.barplot(x=cluster_column, y=indicator, 
                    data=clustered_data, errorbar=None)
        
        # Set plot labels and title
        title_text = indicator.split('(')[0]
        plt.xlabel("Cluster", fontsize=14)
        plt.ylabel(f"{title_text}", fontsize=14)
        plt.title(f'Distribution of {title_text} across Clusters', 
                  fontsize=16, fontweight='bold')
        
        # Show the plot
        plt.show()

        
    

visualize_clusters(c_data, "Cluster", indicator_list)

# Compare Indicator Distribution in each cluster
c_data.groupby('Cluster').last().plot(kind='bar', figsize=(12,8))
plt.title("Indicator Distribution Across CLusters")
plt.ylabel("Value")
plt.show()

# Function to fit model
def fit_model_with_errors(x, y, func, initial_guess=None, alpha=0.05):
    """
    Fit a model to the dataset using both curve fitting and OLS and include error propagation.

    Args:
        x: Independent variable values.
        y: Dependent variable values.
        func: Function to fit the data using curve fitting.
        initial_guess: Initial guess for the parameters (default is None).
        alpha: Significance level for confidence intervals (default is 0.05).

    Returns:
        popt: Optimal parameters of the fitted curve.
        pcov: Covariance matrix of the parameters.
        predictions_curve_fit: Fitted curve values using curve_fit.
        predictions_ols: Fitted curve values using OLS.
        conf_int_low_curve_fit: Lower bounds of confidence intervals for curve_fit.
        conf_int_high_curve_fit: Upper bounds of confidence intervals for curve_fit.
        conf_int_low_ols: Lower bounds of confidence intervals for OLS.
        conf_int_high_ols: Upper bounds of confidence intervals for OLS.
    """
    # Perform curve fitting
    popt, pcov = curve_fit(func, x, y, p0=initial_guess)

    # Generate predictions using the fitted model (curve_fit)
    predictions_curve_fit = func(x, *popt)

    # Ordinary Least Squares (OLS) model
    X_ols = sm.add_constant(x)  # Add a constant term
    model_ols = sm.OLS(y, X_ols).fit()

    # Generate predictions using OLS model
    predictions_ols = model_ols.predict(X_ols)

    # Calculate error ranges using error propagation for curve_fit
    sigma_curve_fit = error_prop(x, func, popt, pcov)

    # Calculate confidence intervals for curve_fit
    lowerbound_curve_fit = predictions_curve_fit - sigma_curve_fit
    upperbound_curve_fit = predictions_curve_fit + sigma_curve_fit

    # Calculate confidence intervals for OLS
    conf_int_ols = model_ols.get_prediction(X_ols).conf_int(alpha=alpha)
    lowerbound_ols, upperbound_ols = conf_int_ols[:, 0], conf_int_ols[:, 1]

    return popt, pcov, predictions_curve_fit, predictions_ols, \
           lowerbound_curve_fit, upperbound_curve_fit, \
           lowerbound_ols, upperbound_ols

# Define a linear function for fitting
def linear_function(x, a, b):
    return a * x + b

# Implementation
x_data = df2[('Year')].astype('int')
y_data = df2['CO2 emissions (kg per PPP $ of GDP)']

# Fit the model with error propagation
optimal_params, covariance_matrix, \
    predictions_curve_fit, predictions_ols, \
        conf_int_low_curve_fit, \
            conf_int_high_curve_fit, \
            lower_bound, \
                upper_bound = fit_model_with_errors(x_data, y_data, linear_function)

# Plot the results
sns.lineplot(x=x_data, y=y_data, label='Original Data', color='blue')
sns.lineplot(x=x_data, y=predictions_ols, label='OLS Predictions', color='red')
plt.fill_between(x_data, lower_bound, upper_bound, 
                 color='black', alpha=0.2, label='Confidence Interval')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (kg per PPP $ of GDP)')
plt.legend()
plt.show()

g_data = c_data.groupby('Cluster').last()
