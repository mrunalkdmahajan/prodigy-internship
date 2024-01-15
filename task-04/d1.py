# Download data from https://www.kaggle.com/code/harshalbhamare/us-accident-eda

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

# Load the dataset (provide the correct file path)
df_USA = pd.read_csv(r'task-05\US_Accidents_March23.csv', nrows=1000)  # Read the first 1000 rows

# Display the first few rows of the dataset
df_USA.head()

# Display the column names
df_USA.columns

# Display the count of each data type in the dataset
df_USA.dtypes.value_counts()

# Display the shape of the dataset (number of rows and columns)
df_USA.shape

# Display basic statistics of the dataset
df_USA.describe()

# Display unique values of the 'State' column
df_USA['State'].unique()

# Create a new dataframe for accidents in Ohio (OH)
df1 = df_USA[df_USA['State'] == 'OH']

# Extract numerical values from the 'ID' column and create a new column 'IDD'
# Convert 'ID' column to string and create a copy of the DataFrame
df1 = df1.copy()
df1['ID_str'] = df1['ID'].astype(str)

# Extract numerical values from 'ID' column using regex
id_extracted = df1['ID_str'].str.extractall(r'(\d+)').unstack().fillna('')

# Sum the extracted values along rows and convert to integer
df1['IDD'] = id_extracted.sum(axis=1).astype(int)

# Drop the intermediate 'ID_str' column
df1.drop('ID_str', axis=1, inplace=True)

df1.head()

# Display the shape and columns of the new dataframe
df1.shape
df1.columns

# Display the number of duplicated rows in the dataframe
df1.duplicated().sum()

# Drop rows with missing values in the 'Precipitation(in)' column
df1 = df1.dropna(subset=['Precipitation(in)'])

# Drop rows with missing values in specified columns
columns_to_drop_na = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Weather_Condition']
df1 = df1.dropna(subset=columns_to_drop_na)

# Display the percentage of missing values in each column
df1.isna().sum() / len(df1) * 100

# Drop rows with missing values in specified columns
columns_to_drop_na = ['City', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
df1 = df1.dropna(subset=columns_to_drop_na)

# Display the count of unique values in the 'Weather_Condition' column
df1['Weather_Condition'].value_counts()

# Separate categorical and numerical columns
df_cat = df1.select_dtypes('object')
df_num = df1.select_dtypes(np.number)
df_cat = df_cat.drop('ID', axis=1)

# Display unique values count for each categorical column
df_cat = df1.select_dtypes('object')

col_name = []
length = []

# Iterate through each categorical column
for i in df_cat.columns:
    col_name.append(i)
    length.append(len(df_cat[i].unique()))

# Create a new dataframe 'df_2' with column names and their respective counts of unique values
df_2 = pd.DataFrame(zip(col_name, length), columns=['feature', 'count_of_unique_values'])
df_2

# Drop unnecessary columns
columns_to_drop = ['Description', 'Zipcode', 'Weather_Timestamp', 'Airport_Code', ]
df1.drop(columns_to_drop, axis=1, inplace=True)

# Check the loaded dataset (df_USA)
"Original dataset shape:", df_USA.shape

# Review any filtering or preprocessing steps leading to df1
"Shape of df1:", df1.shape

# Print the unique values in the 'City' column of df1
"Unique cities in df1:", df1['City'].unique()

# Display the count of each state in the dataset
state_counts = df_USA['State'].value_counts()
"Count of accidents by state:\n", state_counts

df_USA
df1
len(df1['City'].unique())

# Plot heatmap for correlation matrix of numerical columns (excluding 'End_Lat' and 'End_Lng')
plt.figure(figsize=(10, 6))
columns_to_exclude = ['End_Lat', 'End_Lng']
sns.heatmap(df_num.drop(columns_to_exclude, axis=1).corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
plt.show()

# Analyze accidents by cities
cities = df1['City'].unique()
len(cities)

# Count accidents by cities
accidents_by_cities = df1['City'].value_counts()

# Count accidents by severity
accidents_severity = df1.groupby('Severity').count()['ID']

# Plot pie chart for accidents by severity
unique_severity_values = df1['Severity'].unique()
label = [str(value) for value in unique_severity_values]

fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))
plt.pie(accidents_severity, labels=label, autopct='%1.1f%%', pctdistance=0.85)
circle = plt.Circle((0, 0), 0.5, color='white')
p = plt.gcf()
p.gca().add_artist(circle)
ax.set_title("Accident by Severity", fontdict={'fontsize': 16})
plt.tight_layout()
plt.show()

# Convert 'Start_Time' and 'End_Time' to datetime
df1 = df1.astype({'Start_Time': 'datetime64[ns]', 'End_Time': 'datetime64[ns]'})

# Extract date and time components
df1['start_date'] = [d.date() for d in df1['Start_Time']]
df1['start_time'] = [d.time() for d in df1['Start_Time']]
df1['end_date'] = [d.date() for d in df1['End_Time']]
df1['end_time'] = [d.time() for d in df1['End_Time']]

# Display details for a specific index
index_to_check = 469
if index_to_check in df1.index:
    print(df1['Start_Time'][index_to_check])
    print(df1['End_Time'][index_to_check])
else:
    print(f"Index {index_to_check} does not exist in the DataFrame.")

# Remove 'Start_Time' and 'End_Time' columns
del df1['Start_Time']
del df1['End_Time']

# Plot bar chart for top 20 weather conditions
weather_conditions = df1['Weather_Condition'].value_counts()
plt.figure(figsize=(10, 6))
weather_conditions.sort_values(ascending=False)[:20].plot(kind='bar')
plt.title('Weather Conditions at Time of Accident Occurrence')
plt.xlabel('Weather')
plt.ylabel('Accidents Count')
plt.show()

# Plot scatter plot for Severity vs Start_Lat
df_num.plot(kind='scatter', y='Start_Lat', x='Severity', figsize=(10, 6))
plt.title('Scatter Plot: Severity vs Start_Lat')
plt.xlabel('Severity')
plt.ylabel('Start_Lat')
plt.show()

# Scatter plot for Start_Lat vs Start_Lng
scatter_plot = sns.jointplot(x=df_num.Start_Lat.values, y=df_num.Start_Lng.values, height=10)
scatter_plot.set_axis_labels('Start Latitude', 'Start Longitude', fontsize=12)
plt.suptitle('Scatter Plot for Start Location', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
plt.show()


# Bar chart for Top 10 Cities by Number of Accidents
fig_cities = go.Figure()
fig_cities.add_trace(go.Bar(x=accidents_by_cities[:10].index, y=accidents_by_cities[:10].values,
                            hovertext=[f"City: {city}<br>No. of Accidents: {count}<br>Percentage of Accidents: {count / len(df1) * 100:.2f}%"
                                       for city, count in zip(accidents_by_cities[:10].index, accidents_by_cities[:10].values)],
                            hoverinfo='text',
                            marker_color='blue'))

fig_cities.update_layout(title='Top 10 Cities by Number of Accidents',
                         xaxis_title='City',
                         yaxis_title='Accidents Count')

fig_cities.show()


# Scatter plot for Severity vs Start_Lat
fig_severity_start_lat = go.Figure()
fig_severity_start_lat.add_trace(go.Scatter(x=df_num['Severity'], y=df_num['Start_Lat'],
                                           mode='markers',
                                           hovertext=[f"Severity: {severity}<br>Start Latitude: {lat:.6f}"
                                                      for severity, lat in zip(df_num['Severity'], df_num['Start_Lat'])],
                                           hoverinfo='text',
                                           marker=dict(color='red', size=8)))

fig_severity_start_lat.update_layout(title='Scatter Plot: Severity vs Start_Lat',
                                     xaxis_title='Severity',
                                     yaxis_title='Start_Lat')

fig_severity_start_lat.show()


# Pie chart for accidents by severity
fig_severity_pie = go.Figure()

fig_severity_pie.add_trace(go.Pie(
    labels=label,
    values=accidents_severity,
    textinfo='label+percent',
    hoverinfo='label+percent',
    hole=0.3,
    marker=dict(colors=['#ff3333', '#ffd633', '#ff9933', '#33cc33'], line=dict(color='white', width=2))
))

fig_severity_pie.update_layout(
    title='Accidents by Severity',
    annotations=[dict(text='Severity', x=0.5, y=0.5, font_size=20, showarrow=False)],
    showlegend=False  # Hide legend for this plot
)

# Add details to each slice
fig_severity_pie.update_traces(
    textposition='inside',
    textinfo='label+percent',
    insidetextorientation='radial'
)

fig_severity_pie.show()


# Heatmap for correlation matrix of numerical columns (excluding 'End_Lat' and 'End_Lng')
columns_to_exclude = ['End_Lat', 'End_Lng']
corr_matrix = df_num.drop(columns_to_exclude, axis=1).corr()

fig_heatmap = go.Figure()

fig_heatmap.add_trace(go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale='Viridis',
    hoverinfo='z+text',
    text=[[f"{col1} to {col2}<br>Correlation: {corr:.2f}" for col1, corr in zip(corr_matrix.columns, row)]
          for row in corr_matrix.values]
))

fig_heatmap.update_layout(
    title='Correlation Matrix of Numerical Columns',
    xaxis_title='Numerical Columns',
    yaxis_title='Numerical Columns',
    showlegend=False  # Hide legend for this plot
)

fig_heatmap.show()

# Histogram for Start Time
fig_start_time_hist = go.Figure()

fig_start_time_hist.add_trace(go.Histogram(
    x=df1['start_time'],
    nbinsx=50,
    marker_color='green',
    text=df1['start_time'].apply(lambda x: f"Start Time: {x}")
))

fig_start_time_hist.update_layout(
    title='Histogram for Start Time',
    xaxis_title='Start Time',
    yaxis_title='Frequency',
    showlegend=False  # Hide legend for this plot
)

fig_start_time_hist.show()



