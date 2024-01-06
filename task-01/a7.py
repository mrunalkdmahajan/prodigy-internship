import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from tkinter import Tk, Toplevel, Label, Canvas, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go

# Load data from CSVs
data = pd.read_csv(r'task-01\Book2.csv')
female_data = pd.read_csv(r'task-01\female.csv', nrows=14)
male_data = pd.read_csv(r'task-01\male.csv', nrows=14)

# Assuming your CSV file has columns 'YEAR', 'MALE', 'FEMALE'
year = data['YEAR']
values1 = data['MALE']
values2 = data['FEMALE']


# Calculate total population
total_population = values1 + values2

# Create a grouped bar chart using go.Figure
fig = go.Figure()

# Add bar traces for male and female populations
fig.add_trace(go.Bar(x=year, y=values1, name='MALE', marker_color='dodgerblue'))
fig.add_trace(go.Bar(x=year, y=values2, name='FEMALE', marker_color='darkmagenta'))

# Customize the layout
fig.update_layout(
    title='India Male-Female Distribution Bar Chart',
    xaxis=dict(title='Year'),
    yaxis=dict(title='Values'),
    barmode='group',
    hovermode='x',  # Show hover information for each bar on the x-axis
    showlegend=True,
)

# Add hover information with total population
fig.update_traces(
    hovertemplate='%{x}:<br>Total Population: %{customdata[2]:,.0f}<br>Male: %{y:,.0f}<br>Female: %{customdata[1]:,.0f}',
    customdata=list(zip(values1, values2, total_population)),
)

# Show the interactive plot
fig.show()


# Merge female and male data for the year 2022
merged_data = pd.merge(
    female_data[['Country Name', '2022']],
    male_data[['Country Name', '2022']],
    on='Country Name',
    suffixes=('_female', '_male')
)

# Create a bar chart for the year 2022
fig = px.bar(
    merged_data,
    x='Country Name',
    y=['2022_female', '2022_male'],
    title='Female and Male Population in 2022',
    labels={'value': 'Population', 'variable': 'Gender'},
)
fig.update_layout(barmode='group')

# Create a function to handle bar chart click event
def on_bar_click(trace, points, selector):
    # Extract the clicked bar index
    bar_index = points.point_inds[0]

    # Extract the country name for the clicked bar
    clicked_country = merged_data.iloc[bar_index]['Country Name']

    # Filter data for the specific country over the last 10 years (2014 to 2022)
    country_data = pd.merge(
        female_data[['Country Name', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']],
        male_data[['Country Name', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']],
        on='Country Name',
        suffixes=('_female', '_male')
    )
    country_data = country_data[country_data['Country Name'] == clicked_country]

    # Create and show a new window with a bar graph for the specific country
    create_new_window(clicked_country, country_data)

# Add the click event callback to the initial chart
fig.data[0].on_click(on_bar_click)

# Show the initial chart
fig.show()

def create_new_window(country, data):
    # Create a new Tkinter window
    new_window = Toplevel()
    new_window.title(f'Population Distribution for {country}')

    # Create a matplotlib figure
    fig = plt.Figure(figsize=(8, 6), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Plot female population over the last 10 years
    ax1 = plt.subplot(gs[0])
    ax1.bar(data.columns[1:10], data.iloc[:, 1:10].values.flatten(), color='lightblue')
    ax1.set_title('Female Population')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Population')

    # Plot male population over the last 10 years
    ax2 = plt.subplot(gs[1])
    ax2.bar(data.columns[10:], data.iloc[:, 10:].values.flatten(), color='lightgreen')
    ax2.set_title('Male Population')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Population')

    # Embed the matplotlib figure into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

    # Add a button to close the new window
    close_button = Button(new_window, text='Close', command=new_window.destroy)
    close_button.pack(side='bottom')

    # Show the Tkinter window
    new_window.mainloop()
