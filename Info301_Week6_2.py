# !pip install dash
# !pip install dash --upgrade
# !pip install umap
# !pip install umap-learn

import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
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

# === Simulated SHAP Interaction Data ===
# Generate a random interaction matrix for features
features = ['Trust', 'Political_Alignment', 'Social_Media_Engagement', 'Risk_Taking']
interaction_matrix = np.random.rand(len(features), len(features))
interaction_df = pd.DataFrame(interaction_matrix, columns=features, index=features)

# === Simulated SHAP Time-Series Data ===
# Simulate SHAP values over a time series for features
dates = pd.date_range(start="2023-01-01", periods=12, freq="M")  # Monthly data
time_series_data = {
    "Date": np.repeat(dates, len(features)),
    "Feature": features * len(dates),
    "SHAP Value": np.random.rand(len(dates) * len(features))
}
df_shap = pd.DataFrame(time_series_data)

# === Streamlit Application ===

# Page title
st.title("Explainable AI: SHAP Visualizations")

# Feature Interaction Heatmap
st.header("SHAP Feature Interaction Heatmap")
st.markdown("This heatmap visualizes the interaction strength between different features. Higher values indicate stronger interactions.")
fig_heatmap = px.imshow(
    interaction_df,
    text_auto=True,
    color_continuous_scale="Viridis",
    labels={"color": "Interaction Strength"}
)
fig_heatmap.update_layout(
    title="Feature Interaction Heatmap",
    xaxis_title="Feature A",
    yaxis_title="Feature B",
    coloraxis_colorbar=dict(title="Interaction Value"),
)
st.plotly_chart(fig_heatmap)

# SHAP Time-Series Visualization
st.header("SHAP Time-Series Explanations")
selected_features = st.multiselect(
    "Select Features to Display:",
    options=features,
    default=features[:2]
)
selected_date_range = st.date_input(
    "Select Date Range:",
    value=(dates.min(), dates.max()),
    min_value=dates.min(),
    max_value=dates.max()
)

# Filter the time-series data based on user selection
if selected_features:
    filtered_df = df_shap[
        (df_shap["Feature"].isin(selected_features)) &
        (df_shap["Date"] >= selected_date_range[0]) &
        (df_shap["Date"] <= selected_date_range[1])
    ]

    fig_time_series = px.line(
        filtered_df,
        x="Date",
        y="SHAP Value",
        color="Feature",
        title="SHAP Time-Series Values",
        markers=True
    )
    fig_time_series.update_traces(line=dict(width=3), marker=dict(size=8))
    fig_time_series.update_layout(
        xaxis_title="Date",
        yaxis_title="SHAP Value",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig_time_series)
else:
    st.warning("Please select at least one feature to display the time-series chart.")

import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
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

# Ensure "Date" column is in datetime format
df_shap["Date"] = pd.to_datetime(df_shap["Date"])

# === Streamlit App Setup ===

# Page Title
st.title("Time-Series SHAP Explanations")

# Sidebar for selecting features
st.sidebar.header("Filter Options")

# Dropdown for selecting a feature
selected_feature = st.sidebar.selectbox(
    "Select a Feature:",
    options=features,
    index=0
)

# Date range picker
selected_date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=[df_shap["Date"].min(), df_shap["Date"].max()],
    min_value=df_shap["Date"].min(),
    max_value=df_shap["Date"].max()
)

# Filter the data for the selected feature and date range
filtered_df = df_shap[
    (df_shap["Feature"] == selected_feature) &
    (df_shap["Date"] >= pd.to_datetime(selected_date_range[0])) &
    (df_shap["Date"] <= pd.to_datetime(selected_date_range[1]))
]

# Time-Series Visualization
st.subheader(f"SHAP Time-Series Values for {selected_feature}")
fig = px.line(
    filtered_df,
    x="Date",
    y="SHAP Value",
    title=f"SHAP Time-Series Values for {selected_feature}",
    markers=True
)
fig.update_traces(line=dict(width=3, dash="solid"), marker=dict(size=8))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="SHAP Value",
    template="plotly_white",
    hovermode="x unified"
)
st.plotly_chart(fig)

# Description
st.markdown(
    """
    This visualization shows how SHAP values for the selected feature evolve over time.
    Use the dropdown in the sidebar to switch between features and adjust the date range filter.
    """
)

import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
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

# === Streamlit App ===

# Page title
st.title("Enhanced Time-Series SHAP Explanations")

# Sidebar for selecting features and date range
st.sidebar.header("Filter Options")

# Feature multi-select
selected_features = st.sidebar.multiselect(
    "Select Features:",
    options=features,
    default=features[:2]  # Default to the first two features
)

# Date range picker
start_date, end_date = st.sidebar.date_input(
    "Select Date Range:",
    value=[dates.min(), dates.max()],
    min_value=dates.min(),
    max_value=dates.max()
)

# Filter the data based on selected features and date range
filtered_df = df_shap[
    (df_shap["Feature"].isin(selected_features)) &
    (df_shap["Date"] >= str(start_date)) &
    (df_shap["Date"] <= str(end_date))
]

# Interactive time-series visualization
if not filtered_df.empty:
    st.subheader("Time-Series SHAP Values")
    fig = px.line(
        filtered_df,
        x="Date",
        y="SHAP Value",
        color="Feature",
        title="Time-Series SHAP Values",
        markers=True
    )
    fig.update_traces(line=dict(width=2), marker=dict(size=6))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="SHAP Value",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig)
else:
    st.warning("No data available for the selected features and date range.")

# Description
st.markdown(
    """
    This enhanced visualization allows you to select multiple features and a custom date range to analyze SHAP values over time.
    Use the sidebar to filter by features and adjust the time range.
    """
)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

# === Streamlit App Setup ===

# Page title
st.title("Advanced Explainable AI Time-Series")

# Sidebar inputs for filtering
st.sidebar.header("Filter Options")

# Feature multi-select
selected_features = st.sidebar.multiselect(
    "Select Features:",
    options=features,
    default=features[:3]  # Default to the first three features
)

# Date range picker
start_date, end_date = st.sidebar.date_input(
    "Select Date Range:",
    value=(dates.min(), dates.max()),
    min_value=dates.min(),
    max_value=dates.max()
)

# Filter the data based on the selected features and date range
filtered_df = df_shap[
    (df_shap["Feature"].isin(selected_features)) &
    (df_shap["Date"] >= str(start_date)) &
    (df_shap["Date"] <= str(end_date))
]

# === SHAP Time-Series Line Plot ===
st.subheader("Time-Series SHAP Values")
if not filtered_df.empty:
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
    st.plotly_chart(line_fig)
else:
    st.warning("No data available for the selected features and date range.")

# === SHAP Feature Importance Bar Chart ===
st.subheader("Feature Importance (Average SHAP Values)")
if not filtered_df.empty:
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
    st.plotly_chart(bar_fig)
else:
    st.warning("No data available for feature importance.")

# === Feature Statistics Table ===
st.subheader("Feature Statistics Summary")
if not filtered_df.empty:
    stats = filtered_df.groupby("Feature")["SHAP Value"].agg(["min", "max", "mean", "std"]).reset_index()
    st.dataframe(stats)
else:
    st.warning("No data available for feature statistics.")

# Page footer description
st.markdown(
    """
    This enhanced visualization allows you to explore SHAP values over time with dynamic feature selection, date range filtering,
    and additional insights through feature importance and statistics.
    """
)

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

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

# === Streamlit App Setup ===

st.title("Enhanced Recommendation System Graph")

# Sidebar inputs for layer selection and edge weight threshold
st.sidebar.header("Filter Options")
visible_layers = st.sidebar.multiselect(
    "Select Layers to Display:",
    options=["User Features", "Interaction", "System", "Output"],
    default=["User Features", "Interaction", "System", "Output"]
)

weight_threshold = st.sidebar.slider(
    "Adjust Causal Effect Threshold (Edge Weight):",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

# Graph generation logic
def generate_graph(visible_layers, weight_threshold):
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


# Generate and display the graph
fig, explanation = generate_graph(visible_layers, weight_threshold)
st.plotly_chart(fig)

# Display explanation
st.markdown(f"### Explanation\n{explanation}")

import pandas as pd
import requests
import json
import plotly.express as px
import streamlit as st
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

# === Streamlit App Setup ===

st.title("Enhanced Dynamic Topic Tracking with Word Embedding Clustering")

# Sidebar inputs for cluster/label filter and point size
filter_value = st.sidebar.selectbox(
    "Filter by Cluster or Label:",
    options=[
        "All Clusters",
        "Cluster 0",
        "Cluster 1",
        "Cluster 2",
        "Cluster 3",
        "Cluster 4",
        "Label: True",
        "Label: False",
        "Label: Mixture",
    ],
    index=0,
)

point_size = st.sidebar.slider(
    "Adjust Point Size:",
    min_value=5,
    max_value=50,
    value=15,
    step=1,
)

# Filter data based on selection
if filter_value == "All Clusters":
    filtered_data = df_embeddings
    title = "All Clusters and Labels"
elif filter_value.startswith("Cluster"):
    cluster_number = int(filter_value.split(" ")[1])
    filtered_data = df_embeddings[df_embeddings["cluster"] == cluster_number]
    title = f"Cluster {cluster_number}"
else:
    label = filter_value.split(": ")[1].lower()
    filtered_data = df_embeddings[df_embeddings["label"] == label]
    title = f"Label: {label.capitalize()}"

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
    size_max=point_size,
)

# Display visualization and explanation
st.plotly_chart(fig)

st.markdown(
    f"""
    ### Explanation
    Displaying **{title}** with point size adjusted to **{point_size}**.  
    Hover over points to see individual claims, labels, and cluster information.
    """
)