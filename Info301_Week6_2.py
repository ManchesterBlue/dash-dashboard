!pip install dash
!pip install dash --upgrade
!pip install umap
!pip install umap-learn

import pandas as pd
import requests
import json
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np

# === Data Loading ===

# Load the main dataset containing engagement data by country
main_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/Main.csv"
df_main = pd.read_csv(main_url)

# Load the Climate-FEVER dataset with climate-related misinformation claims and evidence
climate_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/climate-fever-dataset-r1.jsonl"
climate_data = []
response = requests.get(climate_url, stream=True)
for line in response.iter_lines():
    if line:
        climate_data.append(json.loads(line))
df_climate = pd.DataFrame(climate_data)

# Load Politifact fake and real news datasets
fake_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/politifact_fake.csv"
real_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/politifact_real.csv"
df_fake = pd.read_csv(fake_url)
df_real = pd.read_csv(real_url)

# === Simulated SHAP Interaction Data (Replace with real SHAP interaction values if available) ===
# Assume SHAP interaction values are computed
features = ['Trust', 'Political_Alignment', 'Social_Media_Engagement', 'Risk_Taking']
interaction_matrix = np.random.rand(len(features), len(features))  # Simulated interaction values
interaction_df = pd.DataFrame(interaction_matrix, columns=features, index=features)

# === Dash App Setup ===

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("SHAP Feature Interaction Heatmap", style={"textAlign": "center"}),

    # Interactive heatmap
    dcc.Graph(
        id="interaction-heatmap",
        config={"displayModeBar": True}
    ),

    # Description
    html.Div(
        children="This heatmap visualizes the interaction strength between different features. Higher values indicate stronger interactions.",
        style={"textAlign": "center", "marginTop": "10px"}
    )
])

@app.callback(
    Output("interaction-heatmap", "figure"),
    Input("interaction-heatmap", "id")
)
def update_heatmap(_):
    # Generate the interactive heatmap using Plotly
    fig = px.imshow(
        interaction_df,
        text_auto=True,
        color_continuous_scale="Viridis",
        labels={"color": "Interaction Strength"}
    )
    fig.update_layout(
        title="Feature Interaction Heatmap",
        xaxis_title="Feature A",
        yaxis_title="Feature B",
        coloraxis_colorbar=dict(title="Interaction Value"),
    )
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

import pandas as pd
import requests
import json
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np

# === Data Loading ===

# Load Main.csv dataset containing engagement data by country
main_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/Main.csv"
df_main = pd.read_csv(main_url)

# Load Climate-FEVER dataset with climate misinformation claims and evidence
climate_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/climate-fever-dataset-r1.jsonl"
climate_data = []
response = requests.get(climate_url, stream=True)
for line in response.iter_lines():
    if line:
        climate_data.append(json.loads(line))
df_climate = pd.DataFrame(climate_data)

# Load Politifact fake and real news datasets
fake_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/politifact_fake.csv"
real_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/politifact_real.csv"
df_fake = pd.read_csv(fake_url)
df_real = pd.read_csv(real_url)

# === Simulated SHAP Time-Series Data ===
# Simulate time-series SHAP values for different features
dates = pd.date_range(start="2023-01-01", periods=12, freq="M")  # Monthly data
features = ['Trust', 'Political_Alignment', 'Social_Media_Engagement', 'Risk_Taking']
data = {
    "Date": np.repeat(dates, len(features)),
    "Feature": features * len(dates),
    "SHAP Value": np.random.rand(len(dates) * len(features))  # Random SHAP values
}
df_shap = pd.DataFrame(data)

# === Dash App Setup ===

app = dash.Dash(__name__)
app.title = "Time-Series SHAP Explanations"

# Layout of the Dash App
app.layout = html.Div([
    html.H1("Time-Series SHAP Explanations", style={"textAlign": "center"}),

    # Dropdown for selecting a feature
    html.Div([
        html.Label("Select Feature:", style={"marginTop": "20px"}),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{"label": feature, "value": feature} for feature in features],
            value=features[0],
            clearable=False
        )
    ], style={"width": "50%", "margin": "auto"}),

    # Interactive time-series visualization
    dcc.Graph(
        id="shap-time-series",
        config={"displayModeBar": True},
        style={"marginTop": "20px"}
    ),

    # Description
    html.Div(
        children="This visualization shows how SHAP values for the selected feature evolve over time. Use the dropdown to switch between features.",
        style={"textAlign": "center", "marginTop": "10px", "fontSize": "16px"}
    )
])

# Callback to update the time-series visualization based on the selected feature
@app.callback(
    Output("shap-time-series", "figure"),
    [Input("feature-dropdown", "value")]
)
def update_time_series(selected_feature):
    # Filter data for the selected feature
    filtered_df = df_shap[df_shap["Feature"] == selected_feature]

    # Generate a line plot using Plotly
    fig = px.line(
        filtered_df,
        x="Date",
        y="SHAP Value",
        title=f"Time-Series SHAP Values for {selected_feature}",
        markers=True
    )

    # Add additional elements for advanced visualization
    fig.update_traces(line=dict(width=3, dash="solid"), marker=dict(size=8))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="SHAP Value",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)

import pandas as pd
import requests
import json
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np

# === Data Loading ===

# Load Main.csv dataset containing engagement data by country
main_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/Main.csv"
df_main = pd.read_csv(main_url)

# Load Climate-FEVER dataset with climate misinformation claims and evidence
climate_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/climate-fever-dataset-r1.jsonl"
climate_data = []
response = requests.get(climate_url, stream=True)
for line in response.iter_lines():
    if line:
        climate_data.append(json.loads(line))
df_climate = pd.DataFrame(climate_data)

# Load Politifact fake and real news datasets
fake_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/politifact_fake.csv"
real_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/politifact_real.csv"
df_fake = pd.read_csv(fake_url)
df_real = pd.read_csv(real_url)

# === Simulated SHAP Time-Series Data ===
# Simulate daily SHAP values for multiple features over two years
dates = pd.date_range(start="2023-01-01", periods=730, freq="D")  # Daily data for 2 years
features = ['Trust', 'Political_Alignment', 'Social_Media_Engagement', 'Risk_Taking', 'Education', 'Age']
data = {
    "Date": np.repeat(dates, len(features)),
    "Feature": features * len(dates),
    "SHAP Value": np.random.rand(len(dates) * len(features))  # Random SHAP values
}
df_shap = pd.DataFrame(data)

# === Dash App Setup ===
app = dash.Dash(__name__)
app.title = "Enhanced Time-Series SHAP Explanations"

# Layout of the Dash App
app.layout = html.Div([
    html.H1("Enhanced Time-Series SHAP Explanations", style={"textAlign": "center"}),

    # Feature multi-select
    html.Div([
        html.Label("Select Features:", style={"marginTop": "20px"}),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{"label": feature, "value": feature} for feature in features],
            value=features[:2],  # Default to the first two features
            multi=True,  # Allow multiple selections
            clearable=True
        )
    ], style={"width": "60%", "margin": "auto"}),

    # Date range picker
    html.Div([
        html.Label("Select Date Range:", style={"marginTop": "20px"}),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=str(dates.min()),
            end_date=str(dates.max()),
            display_format="YYYY-MM-DD",
            style={"margin": "auto"}
        )
    ], style={"width": "60%", "margin": "auto", "textAlign": "center"}),

    # Interactive time-series visualization
    dcc.Graph(
        id="shap-time-series",
        config={"displayModeBar": True},
        style={"marginTop": "20px"}
    ),

    # Description
    html.Div(
        children="This enhanced visualization allows you to select multiple features and a custom date range to analyze SHAP values over time.",
        style={"textAlign": "center", "marginTop": "10px", "fontSize": "16px"}
    )
])

# Callback to update the time-series visualization based on the selected features and date range
@app.callback(
    Output("shap-time-series", "figure"),
    [Input("feature-dropdown", "value"),
     Input("date-picker", "start_date"),
     Input("date-picker", "end_date")]
)
def update_time_series(selected_features, start_date, end_date):
    # Filter data for the selected features and date range
    filtered_df = df_shap[
        (df_shap["Feature"].isin(selected_features)) &
        (df_shap["Date"] >= start_date) &
        (df_shap["Date"] <= end_date)
    ]

    # Generate a line plot using Plotly
    fig = px.line(
        filtered_df,
        x="Date",
        y="SHAP Value",
        color="Feature",
        title="Time-Series SHAP Values",
        markers=True
    )

    # Add additional elements for advanced visualization
    fig.update_traces(line=dict(width=2), marker=dict(size=6))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="SHAP Value",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)

import pandas as pd
import requests
import json
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np

# === Simulated SHAP Time-Series Data ===
# Simulate daily SHAP values for multiple features over two years
dates = pd.date_range(start="2023-01-01", periods=730, freq="D")  # Daily data for 2 years
features = ['Trust', 'Political_Alignment', 'Social_Media_Engagement', 'Risk_Taking', 'Education', 'Age']
data = {
    "Date": np.repeat(dates, len(features)),
    "Feature": features * len(dates),
    "SHAP Value": np.random.rand(len(dates) * len(features))  # Random SHAP values
}
df_shap = pd.DataFrame(data)

# === Dash App Setup ===
app = dash.Dash(__name__)
app.title = "Advanced Explainable AI Time-Series"

# Layout of the Dash App
app.layout = html.Div([
    html.H1("Explainable AI - Advanced Time-Series SHAP Visualizations", style={"textAlign": "center"}),

    # Feature multi-select
    html.Div([
        html.Label("Select Features:", style={"marginTop": "20px"}),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{"label": feature, "value": feature} for feature in features],
            value=features[:3],  # Default to the first three features
            multi=True,  # Allow multiple selections
            clearable=True
        )
    ], style={"width": "60%", "margin": "auto"}),

    # Date range picker
    html.Div([
        html.Label("Select Date Range:", style={"marginTop": "20px"}),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=str(dates.min()),
            end_date=str(dates.max()),
            display_format="YYYY-MM-DD",
            style={"margin": "auto"}
        )
    ], style={"width": "60%", "margin": "auto", "textAlign": "center"}),

    # SHAP Time-Series Line Plot
    dcc.Graph(
        id="shap-time-series",
        config={"displayModeBar": True},
        style={"marginTop": "20px"}
    ),

    # SHAP Feature Importance Bar Chart
    dcc.Graph(
        id="shap-feature-importance",
        config={"displayModeBar": False},
        style={"marginTop": "20px"}
    ),

    # Feature statistics table
    html.Div([
        html.H3("Feature Statistics Summary", style={"textAlign": "center", "marginTop": "20px"}),
        html.Table(id="feature-stats-table", style={"margin": "auto", "width": "80%", "border": "1px solid black"})
    ]),

    # Description
    html.Div(
        children="Explore SHAP time-series explanations with dynamic feature selection, date filtering, and feature statistics.",
        style={"textAlign": "center", "marginTop": "10px", "fontSize": "16px"}
    )
])

# Callback to update the time-series line plot, feature importance bar chart, and statistics table
@app.callback(
    [Output("shap-time-series", "figure"),
     Output("shap-feature-importance", "figure"),
     Output("feature-stats-table", "children")],
    [Input("feature-dropdown", "value"),
     Input("date-picker", "start_date"),
     Input("date-picker", "end_date")]
)
def update_visualizations(selected_features, start_date, end_date):
    # Filter data for the selected features and date range
    filtered_df = df_shap[
        (df_shap["Feature"].isin(selected_features)) &
        (df_shap["Date"] >= start_date) &
        (df_shap["Date"] <= end_date)
    ]

    # === SHAP Time-Series Line Plot ===
    line_fig = px.line(
        filtered_df,
        x="Date",
        y="SHAP Value",
        color="Feature",
        title="Time-Series SHAP Values",
        markers=True
    )
    line_fig.update_traces(line=dict(width=2), marker=dict(size=6))
    line_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="SHAP Value",
        template="plotly_white",
        hovermode="x unified"
    )

    # === SHAP Feature Importance Bar Chart ===
    # Aggregate SHAP values by feature for the selected date range
    feature_importance = filtered_df.groupby("Feature")["SHAP Value"].mean().reset_index()
    bar_fig = px.bar(
        feature_importance,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        title="Feature Importance (Average SHAP Values)",
        color="Feature",
        text="SHAP Value"
    )
    bar_fig.update_layout(
        xaxis_title="Average SHAP Value",
        yaxis_title="Feature",
        template="plotly_white"
    )
    bar_fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')

    # === Feature Statistics Table ===
    stats_table = []
    if not filtered_df.empty:
        stats = filtered_df.groupby("Feature")["SHAP Value"].agg(["min", "max", "mean", "std"]).reset_index()
        # Table header
        stats_table.append(html.Tr([html.Th(col) for col in stats.columns]))
        # Table rows
        for i in range(len(stats)):
            stats_table.append(html.Tr([html.Td(stats.iloc[i, col]) for col in range(len(stats.columns))]))
    else:
        stats_table.append(html.Tr([html.Td("No data available for selected range or features.")]))

    return line_fig, bar_fig, stats_table

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# === Extended Causal Data ===
nodes = [
    {"id": "User Interests", "layer": "User Features", "description": "Interest categories of the user."},
    {"id": "Social Network", "layer": "User Features", "description": "User's social connections."},
    {"id": "Age", "layer": "User Features", "description": "User's age group."},
    {"id": "Content Engagement", "layer": "Interaction", "description": "User's engagement with content."},
    {"id": "Recommendation Algorithm", "layer": "System", "description": "System's recommendation algorithm."},
    {"id": "Recommended Content", "layer": "Output", "description": "Content recommended to the user."},
    {"id": "User Feedback", "layer": "Output", "description": "Feedback provided by the user."},
    {"id": "CTR (Click-Through Rate)", "layer": "System", "description": "Click-through rate of recommendations."},
]

edges = [
    {"source": "User Interests", "target": "Content Engagement", "weight": 0.8},
    {"source": "Social Network", "target": "Content Engagement", "weight": 0.6},
    {"source": "Age", "target": "Content Engagement", "weight": 0.4},
    {"source": "Content Engagement", "target": "Recommendation Algorithm", "weight": 0.9},
    {"source": "Recommendation Algorithm", "target": "Recommended Content", "weight": 0.95},
    {"source": "Recommended Content", "target": "User Feedback", "weight": 0.7},
    {"source": "User Feedback", "target": "CTR (Click-Through Rate)", "weight": 0.6},
    {"source": "CTR (Click-Through Rate)", "target": "Recommendation Algorithm", "weight": 0.5},
]

# === Dash App Setup ===
app = dash.Dash(__name__)
app.title = "Advanced Interactive Recommendation Graph"

# App layout
app.layout = html.Div([
    html.H1("Enhanced Recommendation System Graph", style={"textAlign": "center"}),

    # Layer selection
    html.Div([
        html.Label("Select Layers to Display:", style={"marginTop": "20px"}),
        dcc.Checklist(
            id="layer-checklist",
            options=[
                {"label": "User Features", "value": "User Features"},
                {"label": "Interaction", "value": "Interaction"},
                {"label": "System", "value": "System"},
                {"label": "Output", "value": "Output"},
            ],
            value=["User Features", "Interaction", "System", "Output"],
            inline=True
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Slider to filter edge weights
    html.Div([
        html.Label("Adjust Causal Effect Threshold (Edge Weight):", style={"marginTop": "20px"}),
        dcc.Slider(
            id="weight-slider",
            min=0,
            max=1,
            step=0.1,
            value=0.5,
            marks={i / 10: f"{i / 10:.1f}" for i in range(0, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={"width": "60%", "margin": "auto"}),

    # Graph visualization
    dcc.Graph(id="causal-graph", config={"displayModeBar": False}, style={"marginTop": "30px"}),

    # Explanation Section
    html.Div(id="explanation-section", style={"textAlign": "center", "marginTop": "20px", "fontSize": "16px"})
])

# Callback to update the graph and explanation
@app.callback(
    [Output("causal-graph", "figure"),
     Output("explanation-section", "children")],
    [Input("weight-slider", "value"),
     Input("layer-checklist", "value")]
)
def update_graph(weight_threshold, visible_layers):
    try:
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes
        for node in nodes:
            if node["layer"] in visible_layers:
                G.add_node(node["id"], layer=node["layer"], description=node["description"])

        # Add edges with weight filtering
        for edge in edges:
            if edge["weight"] >= weight_threshold and G.has_node(edge["source"]) and G.has_node(edge["target"]):
                G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

        # Handle empty graph
        if not G.edges:
            fig = go.Figure()
            fig.update_layout(
                title="No Edges Meet the Current Threshold",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_white"
            )
            explanation = "No causal relationships meet the selected threshold."
            return fig, explanation

        # Generate positions for the graph
        pos = nx.multipartite_layout(G, subset_key="layer")

        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=edge[2]["weight"] * 5, color="blue"),
                hoverinfo="text",
                mode="lines",
                text=f"{edge[0]} → {edge[1]}<br>Weight: {edge[2]['weight']:.2f}",
                name=f"{edge[0]} → {edge[1]}"  # Add meaningful legend names
            )
            edge_traces.append(edge_trace)

        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = {"User Features": "blue", "Interaction": "green", "System": "orange", "Output": "red"}
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node[0]}<br>{G.nodes[node[0]]['description']}")
            node_sizes.append(len(list(G.neighbors(node[0]))) * 5 + 20)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                size=node_sizes,
                color=[node_colors[node[1]["layer"]] for node in G.nodes(data=True)],
                line=dict(width=2, color="darkgray")
            )
        )

        # Combine traces
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title="Enhanced Recommendation Graph with Detailed Nodes and Edges",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )

        explanation = f"Displaying causal relationships with weights ≥ {weight_threshold:.1f} in selected layers: {', '.join(visible_layers)}."
        return fig, explanation

    except Exception as e:
        fig = go.Figure()
        explanation = f"An error occurred: {e}. Please try again."
        return fig, explanation

if __name__ == "__main__":
    app.run_server(debug=True)

import pandas as pd
import requests
import json
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap

# === Data Loading ===

# Load Climate-FEVER dataset with climate misinformation claims and evidence
climate_url = "https://raw.githubusercontent.com/AidaCPL/INFOSCI301_Final_Project/main/Data/climate-fever-dataset-r1.jsonl"
climate_data = []
response = requests.get(climate_url, stream=True)
for line in response.iter_lines():
    if line:
        climate_data.append(json.loads(line))
df_climate = pd.DataFrame(climate_data)

# Extract claims and their labels for clustering
if "claim" in df_climate and "claim_label" in df_climate:
    claims = df_climate["claim"]
    claim_labels = df_climate["claim_label"]
else:
    claims = ["Example claim " + str(i) for i in range(100)]  # Fallback data
    claim_labels = ["true", "false", "mixture"] * 33 + ["true"]

# === Text Embedding and Clustering ===

# Generate TF-IDF embeddings
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(claims)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Apply UMAP for dimensionality reduction
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
umap_embeddings = umap_model.fit_transform(X.toarray())

# Create a DataFrame for visualization
df_embeddings = pd.DataFrame(umap_embeddings, columns=["x", "y"])
df_embeddings["cluster"] = clusters
df_embeddings["label"] = claim_labels
df_embeddings["claim"] = claims
df_embeddings["importance"] = [len(claim) % 10 + 1 for claim in claims]  # Simulated feature for importance

# === Dash App Setup ===
app = dash.Dash(__name__)
app.title = "Enhanced Dynamic Topic Tracking"

# App layout
app.layout = html.Div([
    html.H1("Enhanced Dynamic Topic Tracking with Word Embedding Clustering", style={"textAlign": "center"}),

    # Dropdown for selecting cluster or label filter
    html.Div([
        html.Label("Filter by Cluster or Label:", style={"marginTop": "20px"}),
        dcc.Dropdown(
            id="filter-dropdown",
            options=[
                {"label": "All Clusters", "value": "all"},
                {"label": "Cluster 0", "value": "cluster_0"},
                {"label": "Cluster 1", "value": "cluster_1"},
                {"label": "Cluster 2", "value": "cluster_2"},
                {"label": "Cluster 3", "value": "cluster_3"},
                {"label": "Cluster 4", "value": "cluster_4"},
                {"label": "Label: True", "value": "true"},
                {"label": "Label: False", "value": "false"},
                {"label": "Label: Mixture", "value": "mixture"},
            ],
            value="all",
            clearable=False
        )
    ], style={"width": "60%", "margin": "auto"}),

    # Slider to adjust point size
    html.Div([
        html.Label("Adjust Point Size:", style={"marginTop": "20px"}),
        dcc.Slider(
            id="size-slider",
            min=5,
            max=50,
            step=1,
            value=15,
            marks={i: str(i) for i in range(5, 51, 5)},
        )
    ], style={"width": "60%", "margin": "auto"}),

    # Visualization
    dcc.Graph(id="embedding-visualization", style={"marginTop": "30px"}),

    # Explanation Section
    html.Div(id="explanation-section", style={"textAlign": "center", "marginTop": "20px", "fontSize": "16px"})
])

# Callback to update the embedding visualization and explanation dynamically
@app.callback(
    [Output("embedding-visualization", "figure"),
     Output("explanation-section", "children")],
    [Input("filter-dropdown", "value"),
     Input("size-slider", "value")]
)
def update_visualization(filter_value, point_size):
    # Filter data based on selection
    if filter_value == "all":
        filtered_data = df_embeddings
        title = "All Clusters and Labels"
    elif filter_value.startswith("cluster"):
        cluster_number = int(filter_value.split("_")[1])
        filtered_data = df_embeddings[df_embeddings["cluster"] == cluster_number]
        title = f"Cluster {cluster_number}"
    else:
        filtered_data = df_embeddings[df_embeddings["label"] == filter_value]
        title = f"Label: {filter_value.capitalize()}"

    # Create scatter plot with dynamic size and color gradient
    fig = px.scatter(
        filtered_data,
        x="x",
        y="y",
        size="importance",
        color="cluster",
        hover_data={"claim": True, "label": True, "x": False, "y": False},
        title=title,
        labels={"cluster": "Cluster"},
        size_max=point_size
    )

    # Explanation
    explanation = (
        f"Displaying {title} with point size adjusted to {point_size}. "
        "Hover over points to see individual claims, labels, and cluster information."
    )

    return fig, explanation


if __name__ == "__main__":
    app.run_server(debug=True)
