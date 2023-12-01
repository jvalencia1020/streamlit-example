from distutils import errors
from distutils.log import error
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from itertools import cycle

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
from pyecharts.charts import Bar, Scatter
from pyecharts import options as opts
from streamlit_echarts import st_echarts, st_pyecharts

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

indices = np.linspace(0, 1, num_points)
theta = 2 * np.pi * num_turns * indices
radius = indices

x = radius * np.cos(theta)
y = radius * np.sin(theta)

df = pd.DataFrame({
    "x": x,
    "y": y,
    "idx": indices,
    "rand": np.random.randn(num_points),
})

st.altair_chart(alt.Chart(df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
    x=alt.X("x", axis=None),
    y=alt.Y("y", axis=None),
    color=alt.Color("idx", legend=None, scale=alt.Scale()),
    size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
))

np.random.seed(42)

st.button("Generate Plan")

############################################################################################
############################################################################################
############################       Scatter Plots                ############################
############################################################################################
############################################################################################

data = [
    [3.275154, 2.957587],
    [-3.344465, 2.603513],
    [0.355083, -3.376585],
    [1.852435, 3.547351],
    [-2.078973, 2.552013],
    [-0.993756, -0.884433],
    [2.682252, 4.007573],
    [-3.087776, 2.878713],
    [-1.565978, -1.256985],
    [2.441611, 0.444826],
    [-0.659487, 3.111284],
    [-0.459601, -2.618005],
    [2.17768, 2.387793],
    [-2.920969, 2.917485],
    [-0.028814, -4.168078],
    [3.625746, 2.119041],
    [-3.912363, 1.325108],
    [-0.551694, -2.814223],
    [2.855808, 3.483301],
    [-3.594448, 2.856651],
    [0.421993, -2.372646],
    [1.650821, 3.407572],
    [-2.082902, 3.384412],
    [-0.718809, -2.492514],
    [4.513623, 3.841029],
    [-4.822011, 4.607049],
    [-0.656297, -1.449872],
    [1.919901, 4.439368],
    [-3.287749, 3.918836],
    [-1.576936, -2.977622],
    [3.598143, 1.97597],
    [-3.977329, 4.900932],
    [-1.79108, -2.184517],
    [3.914654, 3.559303],
    [-1.910108, 4.166946],
    [-1.226597, -3.317889],
    [1.148946, 3.345138],
    [-2.113864, 3.548172],
    [0.845762, -3.589788],
    [2.629062, 3.535831],
    [-1.640717, 2.990517],
    [-1.881012, -2.485405],
    [4.606999, 3.510312],
    [-4.366462, 4.023316],
    [0.765015, -3.00127],
    [3.121904, 2.173988],
    [-4.025139, 4.65231],
    [-0.559558, -3.840539],
    [4.376754, 4.863579],
    [-1.874308, 4.032237],
    [-0.089337, -3.026809],
    [3.997787, 2.518662],
    [-3.082978, 2.884822],
    [0.845235, -3.454465],
    [1.327224, 3.358778],
    [-2.889949, 3.596178],
    [-0.966018, -2.839827],
    [2.960769, 3.079555],
    [-3.275518, 1.577068],
    [0.639276, -3.41284]
]

line_opt = {
    "animation": False,
    "label": {"formatter": "y = 0.5 * x + 3", "align": "right"},
    "lineStyle": {"type": "solid"},
    "tooltip": {"formatter": "y = 0.5 * x + 3"},
    "data": [
        [{"coord": [0, 3], "symbol": None}, {"coord": [20, 13], "symbol": None}]
    ],
}

CLUSTER_COUNT = 6
DIENSIION_CLUSTER_INDEX = 2
COLOR_ALL = [
    '#37A2DA',
    '#e06343',
    '#37a354',
    '#b55dba',
    '#b5bd48',
    '#8378EA',
    '#96BFFF'
]
pieces = []
for i in range(0, CLUSTER_COUNT):
    pieces.append({
        'value': i,
        'label': 'cluster ' + str(i),
        'color': COLOR_ALL[i]
    })

option = {
    "dataset": [
        {
            "source": data,
        },
        {
            "transform": {
                "type": 'ecStat:clustering',
                "config": {
                    "clusterCount": CLUSTER_COUNT,
                    "outputType": 'single',
                    "outputClusterIndexDimension": DIENSIION_CLUSTER_INDEX
                }
            }
        }
    ],
    "title": {"text": "Plan Performance", "left": "center", "top": 0},
    "tooltip": {"position": 'top'},
    "visualMap": {
        "type": 'piecewise',
        "top": 'middle',
        "min": 0,
        "max": CLUSTER_COUNT,
        "left": 10,
        "splitNumber": CLUSTER_COUNT,
        "dimension": DIENSIION_CLUSTER_INDEX,
        "pieces": pieces
      },
    "grid": {
        "left": 120
    },
    "xAxis": {},
    "yAxis": {},
    "series": [
        {
            "name": "I",
            "type": "scatter",
            "encode": {
                "tooltip": [0, 1]
            },
            "symbolSize": 15,
            "itemStyle": {
                "borderControl": '#555'
            },
            "datasetIndex": 1
        },
    ],
}

# data = [
#     [
#         [10.0, 8.04],
#         [8.0, 6.95],
#         [13.0, 7.58],
#         [9.0, 8.81],
#         [11.0, 8.33],
#         [14.0, 9.96],
#         [6.0, 7.24],
#         [4.0, 4.26],
#         [12.0, 10.84],
#         [7.0, 4.82],
#         [5.0, 5.68],
#     ],
#     [
#         [10.0, 9.14],
#         [8.0, 8.14],
#         [13.0, 8.74],
#         [9.0, 8.77],
#         [11.0, 9.26],
#         [14.0, 8.10],
#         [6.0, 6.13],
#         [4.0, 3.10],
#         [12.0, 9.13],
#         [7.0, 7.26],
#         [5.0, 4.74],
#     ],
#     [
#         [10.0, 7.46],
#         [8.0, 6.77],
#         [13.0, 12.74],
#         [9.0, 7.11],
#         [11.0, 7.81],
#         [14.0, 8.84],
#         [6.0, 6.08],
#         [4.0, 5.39],
#         [12.0, 8.15],
#         [7.0, 6.42],
#         [5.0, 5.73],
#     ],
#     [
#         [8.0, 6.58],
#         [8.0, 5.76],
#         [8.0, 7.71],
#         [8.0, 8.84],
#         [8.0, 8.47],
#         [8.0, 7.04],
#         [8.0, 5.25],
#         [19.0, 12.50],
#         [8.0, 5.56],
#         [8.0, 7.91],
#         [8.0, 6.89],
#     ],
# ]
#
# line_opt = {
#     "animation": False,
#     "label": {"formatter": "y = 0.5 * x + 3", "align": "right"},
#     "lineStyle": {"type": "solid"},
#     "tooltip": {"formatter": "y = 0.5 * x + 3"},
#     "data": [
#         [{"coord": [0, 3], "symbol": None}, {"coord": [20, 13], "symbol": None}]
#     ],
# }
#
# option = {
#     "title": {"text": "Anscombe's quartet", "left": "center", "top": 0},
#     "grid": [
#         {"left": "7%", "top": "7%", "width": "38%", "height": "38%"},
#         {"right": "7%", "top": "7%", "width": "38%", "height": "38%"},
#         {"left": "7%", "bottom": "7%", "width": "38%", "height": "38%"},
#         {"right": "7%", "bottom": "7%", "width": "38%", "height": "38%"},
#     ],
#     "tooltip": {"formatter": "Group {a}: ({c})"},
#     "xAxis": [
#         {"gridIndex": 0, "min": 0, "max": 20},
#         {"gridIndex": 1, "min": 0, "max": 20},
#         {"gridIndex": 2, "min": 0, "max": 20},
#         {"gridIndex": 3, "min": 0, "max": 20},
#     ],
#     "yAxis": [
#         {"gridIndex": 0, "min": 0, "max": 15},
#         {"gridIndex": 1, "min": 0, "max": 15},
#         {"gridIndex": 2, "min": 0, "max": 15},
#         {"gridIndex": 3, "min": 0, "max": 15},
#     ],
#     "series": [
#         {
#             "name": "I",
#             "type": "scatter",
#             "xAxisIndex": 0,
#             "yAxisIndex": 0,
#             "data": data[0],
#             "markLine": line_opt,
#         },
#         {
#             "name": "II",
#             "type": "scatter",
#             "xAxisIndex": 1,
#             "yAxisIndex": 1,
#             "data": data[1],
#             "markLine": line_opt,
#         },
#         {
#             "name": "III",
#             "type": "scatter",
#             "xAxisIndex": 2,
#             "yAxisIndex": 2,
#             "data": data[2],
#             "markLine": line_opt,
#         },
#         {
#             "name": "IV",
#             "type": "scatter",
#             "xAxisIndex": 3,
#             "yAxisIndex": 3,
#             "data": data[3],
#             "markLine": line_opt,
#         },
#     ],
# }
st_echarts(options=option, height="600px")


############################################################################################
############################################################################################
############################       Spreadsheet Grid             ############################
############################################################################################
############################################################################################

@st.cache(allow_output_mutation=True)
def fetch_data(samples):
    deltas = cycle([
        pd.Timedelta(weeks=-2),
        pd.Timedelta(days=-1),
        pd.Timedelta(hours=-1),
        pd.Timedelta(0),
        pd.Timedelta(minutes=5),
        pd.Timedelta(seconds=10),
        pd.Timedelta(microseconds=50),
        pd.Timedelta(microseconds=10)
    ])
    dummy_data = {
        "date_time_naive": pd.date_range('2021-01-01', periods=samples),
        "apple": np.random.randint(0, 100, samples) / 3.0,
        "banana": np.random.randint(0, 100, samples) / 5.0,
        "chocolate": np.random.randint(0, 100, samples),
        "group": np.random.choice(['A', 'B'], size=samples),
        "date_only": pd.date_range('2020-01-01', periods=samples).date,
        "timedelta": [next(deltas) for i in range(samples)],
        "date_tz_aware": pd.date_range('2022-01-01', periods=samples, tz="Asia/Katmandu")
    }
    return pd.DataFrame(dummy_data)


# Example controlers
st.sidebar.subheader("St-AgGrid example options")

sample_size = st.sidebar.number_input("rows", min_value=10, value=30)
grid_height = st.sidebar.number_input("Grid height", min_value=200, max_value=800, value=300)

return_mode = st.sidebar.selectbox("Return Mode", list(DataReturnMode.__members__), index=1)
return_mode_value = DataReturnMode.__members__[return_mode]

update_mode = st.sidebar.selectbox("Update Mode", list(GridUpdateMode.__members__),
                                   index=len(GridUpdateMode.__members__) - 1)
update_mode_value = GridUpdateMode.__members__[update_mode]

# enterprise modules
enable_enterprise_modules = st.sidebar.checkbox("Enable Enterprise Modules")
if enable_enterprise_modules:
    enable_sidebar = st.sidebar.checkbox("Enable grid sidebar", value=False)
else:
    enable_sidebar = False

# features
fit_columns_on_grid_load = st.sidebar.checkbox("Fit Grid Columns on Load")

enable_selection = st.sidebar.checkbox("Enable row selection", value=True)
if enable_selection:
    st.sidebar.subheader("Selection options")
    selection_mode = st.sidebar.radio("Selection Mode", ['single', 'multiple'], index=1)

    use_checkbox = st.sidebar.checkbox("Use check box for selection", value=True)
    if use_checkbox:
        groupSelectsChildren = st.sidebar.checkbox("Group checkbox select children", value=True)
        groupSelectsFiltered = st.sidebar.checkbox("Group checkbox includes filtered", value=True)

    if ((selection_mode == 'multiple') & (not use_checkbox)):
        rowMultiSelectWithClick = st.sidebar.checkbox("Multiselect with click (instead of holding CTRL)", value=False)
        if not rowMultiSelectWithClick:
            suppressRowDeselection = st.sidebar.checkbox("Suppress deselection (while holding CTRL)", value=False)
        else:
            suppressRowDeselection = False
    st.sidebar.text("___")

enable_pagination = st.sidebar.checkbox("Enable pagination", value=False)
if enable_pagination:
    st.sidebar.subheader("Pagination options")
    paginationAutoSize = st.sidebar.checkbox("Auto pagination size", value=True)
    if not paginationAutoSize:
        paginationPageSize = st.sidebar.number_input("Page size", value=5, min_value=0, max_value=sample_size)
    st.sidebar.text("___")

# df = fetch_data(sample_size)

df = pd.read_csv('media_plan_sample.csv')
df = df[['station', 'start_date', 'end_date', 'weeks_affected', 'hiatus_dates', 'weeks_left', 'days', 'daypart',
         'length', 'quantity', 'rate', 'booked']]

# Infer basic colDefs from dataframe types
gb = GridOptionsBuilder.from_dataframe(df)

# customize gridOptions
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
gb.configure_column("start_date", type=["dateColumnFilter", "customDateTimeFormat"], custom_format_string='yyyy-MM-dd',
                    pivot=True)
gb.configure_column("end_date", type=["dateColumnFilter", "customDateTimeFormat"], custom_format_string='yyyy-MM-dd',
                    pivot=True)
gb.configure_column("weeks_affected", type=["dateColumnFilter", "customDateTimeFormat"],
                    custom_format_string='yyyy-MM-dd', pivot=True)
gb.configure_column("hiatus_dates", type=["dateColumnFilter", "customDateTimeFormat"],
                    custom_format_string='yyyy-MM-dd', pivot=True)
# gb.configure_column("hiatus_dates", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='yyyy-MM-dd HH:mm zzz', pivot=True)

gb.configure_column("quantity", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=2,
                    aggFunc='sum')
gb.configure_column("booked", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=1,
                    aggFunc='avg')
# gb.configure_column("chocolate", type=["numericColumn", "numberColumnFilter", "customCurrencyFormat"], custom_currency_symbol="R$", aggFunc='max')

# configures last row to use custom styles based on cell's value, injecting JsCode on components front end
cellsytle_jscode = JsCode("""
function(params) {
    if (params.value == 0) {
        return {
            'color': 'white',
            'backgroundColor': 'darkred'
        }
    } else {
        return {
            'color': 'black',
            'backgroundColor': 'white'
        }
    }
};
""")
gb.configure_column("quantity", cellStyle=cellsytle_jscode)

if enable_sidebar:
    gb.configure_side_bar()

if enable_selection:
    gb.configure_selection(selection_mode)
    if use_checkbox:
        gb.configure_selection(selection_mode, use_checkbox=True, groupSelectsChildren=groupSelectsChildren,
                               groupSelectsFiltered=groupSelectsFiltered)
    if ((selection_mode == 'multiple') & (not use_checkbox)):
        gb.configure_selection(selection_mode, use_checkbox=False, rowMultiSelectWithClick=rowMultiSelectWithClick,
                               suppressRowDeselection=suppressRowDeselection)

if enable_pagination:
    if paginationAutoSize:
        gb.configure_pagination(paginationAutoPageSize=True)
    else:
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=paginationPageSize)

gb.configure_grid_options(domLayout='normal')
gridOptions = gb.build()

# Display the grid
st.header("Plan Recommendatiosn")
st.markdown("""
    Below are your plan recommendations for 11/20/2023 through 12/10/2023!
""")

grid_response = AgGrid(
    df,
    gridOptions=gridOptions,
    height=grid_height,
    width='100%',
    data_return_mode=return_mode_value,
    update_mode=update_mode_value,
    fit_columns_on_grid_load=fit_columns_on_grid_load,
    allow_unsafe_jscode=True,  # Set it to True to allow jsfunction to be injected
    enable_enterprise_modules=enable_enterprise_modules
)

df = grid_response['data']
selected = grid_response['selected_rows']
selected_df = pd.DataFrame(selected).apply(pd.to_numeric, errors='coerce')

# with st.spinner("Displaying results..."):
#     #displays the chart
#     chart_data = df.loc[:,['apple','banana','chocolate']].assign(source='total')
#
#     if not selected_df.empty :
#         selected_data = selected_df.loc[:,['apple','banana','chocolate']].assign(source='selection')
#         chart_data = pd.concat([chart_data, selected_data])
#
#     chart_data = pd.melt(chart_data, id_vars=['source'], var_name="item", value_name="quantity")
#     #st.dataframe(chart_data)
#     chart = alt.Chart(data=chart_data).mark_bar().encode(
#         x=alt.X("item:O"),
#         y=alt.Y("sum(quantity):Q", stack=False),
#         color=alt.Color('source:N', scale=alt.Scale(domain=['total','selection'])),
#     )
#
#     st.header("Component Outputs - Example chart")
#     st.markdown("""
#     This chart is built with data returned from the grid. rows that are selected are also identified.
#     Experiment selecting rows, group and filtering and check how the chart updates to match.
#     """)
#
#     st.altair_chart(chart, use_container_width=True)
#
#     st.subheader("Returned grid data:")
#     #returning as HTML table bc streamlit has issues when rendering dataframes with timedeltas:
#     # https://github.com/streamlit/streamlit/issues/3781
#     st.markdown(grid_response['data'].to_html(), unsafe_allow_html=True)
#
#     st.subheader("grid selection:")
#     st.write(grid_response['selected_rows'])
#
#     st.header("Generated gridOptions")
#     st.markdown("""
#         All grid configuration is done thorugh a dictionary passed as ```gridOptions``` parameter to AgGrid call.
#         You can build it yourself, or use ```gridOptionBuilder``` helper class.
#         Ag-Grid documentation can be read [here](https://www.ag-grid.com/documentation)
#     """)
#     st.write(gridOptions)
