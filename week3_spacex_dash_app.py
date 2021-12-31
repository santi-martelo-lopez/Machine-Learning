
# lets install pandas dash
#pip install pandas dash

# lets download the data set
# wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"


# lets download the template
# wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/spacex_dash_app.py"

# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
#from dash.dependencies import Input, Output
from dash.dependencies import Input, Output, State
import plotly.express as px
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
print(spacex_df.head(20))
data = spacex_df.groupby("Launch Site").mean().reset_index()
print(data[["Launch Site","class"]])
data1 = spacex_df.groupby("Launch Site").sum().reset_index()
print(data1[["Launch Site","class"]])
data1 = spacex_df.groupby("Booster Version Category").mean().reset_index()
print(data1[["Booster Version Category","class"]])
data1 = spacex_df.groupby("Booster Version Category").sum().reset_index()
print(data1[["Booster Version Category","class"]])

launch = []
launchSitesList = []
for ii,val in enumerate(spacex_df["Launch Site"].unique()) :
    launch.append(val)
    launchSitesList.append({'label': launch[ii], 'value': launch[ii]})
    print(ii,val,launchSitesList[ii])

max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                dcc.Dropdown(id='site-dropdown',options=[{'label': 'All Sites', 'value': 'ALL'},#launchSitesList
                                {'label': launch[0], 'value': launch[0]},
                                {'label': launch[1], 'value': launch[1]},
                                {'label': launch[2], 'value': launch[2]},
                                {'label': launch[3], 'value': launch[3]}
                                ],
                                                value='ALL',
                                                placeholder="Select a launch site here",
                                                searchable=True
                                                ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                #dcc.RangeSlider(id='payload-slider',...)
                                dcc.RangeSlider(id='payload-slider',
                                    min=0, max=10000, step=1000,
                                    marks={0: '0',100: '100'},
                                    value=[min_payload,max_payload]),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
# Function decorator to specify function input and output
@app.callback(
              Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))


def get_pie_chart(entered_site):
    filtered_df = spacex_df
    # success rates for all sites
    data = spacex_df.groupby("Launch Site").mean().reset_index()
    if entered_site == 'ALL':
        title_message = "Total Sucess Launches By Site"
        fig = px.pie(data, values='class', 
        names = launch, 
        title=title_message)
        return fig
    else:
        title_message = f"Total Sucess Launches for site {entered_site}"
        sr = data.loc[data["Launch Site"]==entered_site,"class"]
        sr_dict = {"Success Rate":[float(1-sr),float(sr)]} # you need to cast the values to float
        data1 = pd.DataFrame(sr_dict)
        fig = px.pie(data1, values='Success Rate', 
        names = [0,1], 
        title=title_message)
        return fig
        
# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(
                Output(component_id='success-payload-scatter-chart', component_property='figure'),
            [
                Input(component_id='site-dropdown', component_property='value'), 
                Input(component_id="payload-slider", component_property="value")
            ]
            )
def get_scatter_plot(entered_site,selected_payload):
    # success rates for all sites
    df = spacex_df.loc[ ( (spacex_df["Payload Mass (kg)"] >= selected_payload[0]) & (spacex_df["Payload Mass (kg)"] <= selected_payload[1]) ),:]
    if entered_site == 'ALL':
        title_message = "Correlation between Payload and Success for all sites"
        # Plot a scatter point chart with x axis to be Flight Number and y axis to be the launch site, and hue to be the class value
        #fig = sns.catplot(y="class", x="Payload Mass (kg)", hue="Booster Version", data=df, aspect = 5)
        #plt.xlabel("Payload Mass (kg)",fontsize=20)
        #plt.ylabel("class",fontsize=20)
        #plt.show()
        fig = px.scatter(df,x="Payload Mass (kg)",y="class",color="Booster Version Category",
        title=title_message)
        return fig
    else:
        df = spacex_df.loc[ ( (spacex_df["Payload Mass (kg)"] >= selected_payload[0]) & (spacex_df["Payload Mass (kg)"] <= selected_payload[1]) ),:]
        df1 = df.loc[ spacex_df["Launch Site"]==entered_site,:]
        title_message = f"Correlation between Payload and Success for site {entered_site}"
        fig = px.scatter(df1,x="Payload Mass (kg)",y="class",color="Booster Version Category",
        title=title_message)
        return fig

# Run the app
if __name__ == '__main__':
    #app.run_server()
    app.run_server(debug=True)
    # http://127.0.0.1:8050/
    #app.run_server(port=8050)
    #app.run_server(port=8050,debug=True)