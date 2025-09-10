

from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pydeck as pdk
import json
import sys
import streamlit as st
from scipy.ndimage import gaussian_filter
from streamlit_folium import st_folium
import folium
from shapely.geometry import Point, shape
from streamlit_extras.stylable_container import stylable_container


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
CLS_PATH = Path(__file__).with_name("classifier_predictions.csv")
LSTM_PATH = Path(__file__).with_name("lstm_predictions.csv")
LOSS_PATH = Path(__file__).with_name("loss_data.csv")
LOW_THR = 0.46
MED_THR = 0.65
HIGH_THR = 0.80

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Alberta Deforestation Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    /* Light mode */
    @media (prefers-color-scheme: light) {
        .main, .stMarkdown, .stText, .css-1d391kg {
            color: black !important;
        }
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        .main, .stMarkdown, .stText, .css-1d391kg {
            color: white !important;
        }
    }
    </style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def plot_loss_trend(filtered_df: pd.DataFrame):
    if filtered_df.empty:
        st.warning("No forest loss points found inside the selected area.")
        return

    # Aggregate forest loss by year
    trend_df = (
        filtered_df.groupby("label")["count"]
        .sum()
        .reset_index()
        .rename(columns={"label": "lossyear", "count": "loss_pixels"})
    )
    trend_df["year"] = trend_df["lossyear"] + 2000
    trend_df = trend_df.sort_values("year")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(trend_df["year"], trend_df["loss_pixels"], marker="o", color="#2bd47d")
    ax.set_title("Annual Forest Loss in Selected Area", fontsize=12, color="white")
    ax.set_xlabel("Year", color="white")
    ax.set_ylabel("Forest Loss (approx. 30m pixels)", color="white")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_facecolor("#0e1117")
    fig.set_facecolor("#0e1117")  # Compatible across versions
    ax.tick_params(colors="white")

    st.pyplot(fig)

def filter_points_in_polygon(df: pd.DataFrame, geojson_polygon: dict) -> pd.DataFrame:
    # Convert GeoJSON to shapely polygon
    polygon = shape(geojson_polygon["geometry"])

    # Build Point objects from coordinates
    df["point"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)

    # Filter only points inside polygon
    filtered_df = df[df["point"].apply(polygon.contains)].copy()

    return filtered_df

def draw_polygon_map():
    # Alberta ROI bounds (southwest and northeast corners)
    bounds_sw = [48.9, -122.8]
    bounds_ne = [54.5, -113.5]
    bounds_box = [bounds_sw, bounds_ne]

    m = folium.Map(
        location=[51.7, -118.0],
        zoom_start=6,
        tiles="OpenStreetMap",
        max_bounds=True  # ← enables map restriction
    )

    # Apply bounding box manually to maxBounds
    m.fit_bounds(bounds_box)
    m.options["maxBounds"] = bounds_box  # ← hard constraint

    # Show ROI as dashed polygon
    folium.Polygon(
        locations=[
            bounds_sw,
            [bounds_sw[0], bounds_ne[1]],
            bounds_ne,
            [bounds_ne[0], bounds_sw[1]],
            bounds_sw
        ],
        color="#ff0000",
        fill=False,
        weight=2,
        dash_array="5, 5",
    ).add_to(m)

    folium.plugins.Draw(
        draw_options={
            "polyline": False,
            "circle": False,
            "rectangle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": {
                "shapeOptions": {
                    "color": "#1f5c1a",
                    "fillOpacity": 0.4,
                }
            }
        },
        edit_options={"edit": False}
    ).add_to(m)

    result = st_folium(m, height=550, width=750, returned_objects=["all_drawings"])

    # Handle drawing
    if result:
        drawings = result.get("all_drawings", [])
        
        if drawings:
            # User drew a new polygon
            st.session_state["polygon"] = drawings[0]
            st.session_state["polygon_updated"] = True  # Optional flag
        elif "polygon" in st.session_state:
            # User deleted the polygon
            del st.session_state["polygon"]
            st.session_state["polygon_updated"] = False


@st.cache_data
def loadHistoricalData(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"CSV file not found at **{path.name}**. Place it next to this script and restart.")
        st.stop()
    df = pd.read_csv(path)
    
    def parse_geo(geo_str):
        geo_dict = json.loads(geo_str)
        lon, lat = geo_dict["coordinates"]
        return pd.Series([lat, lon])

    df[["lat", "lon"]] = df[".geo"].apply(parse_geo)

    # Optional: drop unnecessary columns
    df = df[["lat", "lon", "label", "count"]]
    
    return df

loss_df = loadHistoricalData(LOSS_PATH)

@st.cache_data
def loadLSTMPredictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"CSV file not found at **{path.name}**. Place it next to this script and restart.")
        st.stop()
    df = pd.read_csv(path)
    df = df.dropna()
    df["year"] = df["year"].astype(int)
    df["predicted_loss"] = df["predicted_loss"].astype(int)
    return df

lstm_df = loadLSTMPredictions(LSTM_PATH)
years = sorted(lstm_df["year"].unique())


def loadClassifierPredictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"CSV file not found at **{path.name}**. Place it next to this script and restart.")
        st.stop()
    df = pd.read_csv(path)
    df = df.dropna()
    return apply_smoothing(df)



def apply_smoothing(df: pd.DataFrame, resolution: int = 300) -> pd.DataFrame:
    df = df.dropna(subset=["lat", "lon", "prob"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "prob"])

    lats = df["lat"].values
    lons = df["lon"].values
    probs = df["prob"].values

    lat_bins = np.linspace(lats.min(), lats.max(), resolution)
    lon_bins = np.linspace(lons.min(), lons.max(), resolution)

    # Weighted average of probs per bin
    heatmap_grid, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins], weights=probs)
    counts, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins])
    avg_probs = np.divide(heatmap_grid, counts, out=np.zeros_like(heatmap_grid), where=counts != 0)

    smoothed_probs = gaussian_filter(avg_probs, sigma=1.5)

    # Build output: one row per cell (lat, lon, smoothed_prob)
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

    smoothed_points = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            prob = smoothed_probs[i, j]
            if prob > 0:
                smoothed_points.append({
                    "lat": lat_centers[i],
                    "lon": lon_centers[j],
                    "prob": prob
                })

    return pd.DataFrame(smoothed_points)

classifier_df = loadClassifierPredictions(CLS_PATH)

# ---------- CIRCULAR GAUGES ----------
def plot_circular_metric(label: str, value: float):
    # Good = green, bad = red
    if value >= 75:
        color = "#2bd47d"  # Green
    elif value >= 50:
        color = "#f9c74f"  # Yellow
    else:
        color = "#e63946"  # Red

    fig, ax = plt.subplots(figsize=(1.8, 1.8), facecolor="#0e1117")

    ax.pie(
        [value, 100 - value],
        startangle=90,
        colors=[color, "#333333"],
        radius=1.0,
        wedgeprops={"width": 0.3, "edgecolor": "#0e1117"},
    )

    ax.text(0, 0.15, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(0, -0.25, f"{value:.1f}%", ha="center", va="center", fontsize=10, color="white")

    ax.set(aspect="equal")
    ax.axis("off")

    return fig

# ----------------------------------------------------------------------------

# Maintain the current section in session_state
SECTIONS = (
    "Introduction",
    "Historical Trends",
    "Models",
    "Insights",
    "About",
)

if "section" not in st.session_state:
    st.session_state["section"] = "Introduction"

with st.sidebar:
    st.title("📚 Navigation")
    for sec in SECTIONS:
        btn = st.sidebar.button(sec, key=f"nav_{sec}", disabled=(sec == st.session_state["section"]), type="tertiary")
        if btn:
            st.session_state["section"] = sec
            st.rerun()

    # Divider before any other sidebar widgets that individual sections might add
    st.markdown("---")

section = st.session_state["section"]

# ----------------------------------------------------------------------------
# Introduction
# ----------------------------------------------------------------------------
if section == "Introduction":
    # Main title
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 3.5em; color: #dddddd;'>
            Predicting <span style='color: #1f5c1a;'>Deforestation</span> using Deep Learning
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # Sub‑title
    st.markdown(
        """
        <h4 style='text-align: center; color: #bbbbbb;'>
            A data‑driven approach to understanding environmental risk in Alberta 🇨🇦
        </h4>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Intro sections
    st.markdown(
        """
        <div style='padding: 2rem 12%; font-size: 1.1rem; color: #e0e0e0; line-height: 1.6;'>
            <h4 style='color: #ffffff;'>🔍 What It Is</h4>
            <p>
            An interactive dashboard that visualises high‑risk deforestation zones in Alberta using
            deep learning models. The models were trained using satellite imagery obtained through Google Earth Engine (GEE). The app provides insights into historical trends,
            model predictions, and key insights into deforestation risk.
            </p>
            <h4 style='margin-top: 2rem; color: #ffffff;'>🌿 Why It Matters</h4>
            <p>
            Forests are disappearing faster than ever. By identifying high‑risk zones early, we can inform
            conservation efforts, improve policy, and protect ecosystems before they're lost. This dashboard
            aims to provide a comprehensive overview of deforestation risk in Alberta, helping stakeholders
            make informed decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


    with stylable_container(
        "green",
        css_styles="""
            button {
                background-color: #1f5c1a;
                color: white;
                font-size: 1.5rem;
                padding: 0.75rem 2.5rem;
                border: none;
                border-radius: 10px;
                transition: background-color 0.2s ease;
            }
            button:hover {
                background-color: #1c1e26;
                color: black;
                cursor: pointer;
            }
            """,
    ):
        col1, col2, col3 , col4, col5 = st.columns(5)

        with col1:
            pass
        with col2:
            pass
        with col4:
            pass
        with col5:
            pass
        with col3 :
            center_button = st.button('Get Started')
    
    if center_button:
        st.session_state["section"] = "Historical Trends"
        st.rerun()
# ----------------------------------------------------------------------------
# Historical Trends (placeholder)
# ----------------------------------------------------------------------------
elif section == "Historical Trends":
    st.header("Historical Trends")
    st.subheader("🗺️ Select an Area")

    st.markdown(
        """
        <div style='font-size: 1.05rem; color: #dddddd; line-height: 1.6; padding-bottom: 1rem;'>
            Use the map below to draw a custom region of interest anywhere in Alberta. 
            Once a polygon is drawn, the dashboard will display annual deforestation trends within that area 
            from <strong>2001 to 2023</strong>, based on the <em>Global Forest Change v1.10</em> dataset by Hansen et al.
            <br><br>
            Each data point represents an area of approximately <strong>30×30 meters</strong> of forest cover loss 
            detected via satellite imagery. This interactive tool helps identify deforestation hotspots 
            and temporal patterns over two decades.
        </div>
        """, unsafe_allow_html=True
    )

    draw_polygon_map()
    if "polygon" in st.session_state:
        with st.spinner("Processing selected area..."):
            filtered_df = filter_points_in_polygon(loss_df, st.session_state["polygon"])
            st.markdown(f"📌 **{len(filtered_df):,}** deforestation points found in selected area.")
            plot_loss_trend(filtered_df)
            st.caption("Each pixel represents ~30×30m of tree cover loss as mapped by satellite imagery.")
    else:
        st.info("Draw a polygon above to begin.")

    st.divider()
    
    st.markdown(
        """
        <h4 style='text-align: center; color: #bbbbbb;'>
            More visualisations coming soon...
        </h4>
        """
    , unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Models – with sub‑tabs
# ----------------------------------------------------------------------------
elif section == "Models":
    st.header("Model Outputs")
    tabs = st.tabs(["Classifier", "LSTM"])

    # --------------------------- Classifier tab -----------------------------
    with tabs[0]:
        st.subheader("Classifier")

        st.markdown("""
        <div style='font-size: 1.05rem; color: #dddddd; line-height: 1.6; padding-bottom: 1rem;'>
            This model assigns a <strong>deforestation risk score</strong> to locations in Alberta based on satellite imagery. 
            Each point is evaluated using spatial features, and categorized as <em>low</em>, <em>medium</em>, or <em>high</em> risk.
            <br><br>
            Use the toggle below to filter and visualize only the risk levels you're most interested in. 
            The colors correspond to the severity of risk: <strong style='color: darkred;'>dark red</strong> (high), 
            <strong style='color: orangered;'>orange-red</strong> (medium), and <strong style='color: orange;'>orange</strong> (low).
        </div>
        """, unsafe_allow_html=True)


        LOW, MED, HIGH = 0.04, 0.19, 0.29

        with st.spinner("Loading Map..."):
            risk_level = st.radio(
                "Select Risk Level:",
                options=["All", "Low", "Medium", "High"],
                horizontal=True
            )

            filtered_df = classifier_df.copy()

            # Set threshold
            if risk_level == "High":
                filtered_df = filtered_df[filtered_df["prob"] >= HIGH].copy()
            elif risk_level == "Medium":
                filtered_df = filtered_df[(filtered_df["prob"] >= MED) & (filtered_df["prob"] < HIGH)].copy()
            elif risk_level == "Low":
                filtered_df = filtered_df[(filtered_df["prob"] >= LOW) & (filtered_df["prob"] < MED)].copy()
            else:
                filtered_df = filtered_df[filtered_df["prob"] >= LOW].copy()

            # Color function
            def get_color(prob):
                if prob >= HIGH:
                    return [139, 0, 0]       # dark red
                elif prob >= MED:
                    return [255, 69, 0]      # orangered
                else:
                    return [255, 165, 0]     # orange

        

            filtered_df["color"] = filtered_df["prob"].apply(get_color)

            # PyDeck layer
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered_df,
                get_position='[lon, lat]',
                get_color='color',
                get_radius=1000,
                pickable=True,
                opacity=0.5,
            )

            view_state = pdk.ViewState(
                latitude=filtered_df["lat"].mean(),
                longitude=filtered_df["lon"].mean(),
                zoom=5,
                pitch=0,
            )

            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/dark-v11"
            ))

            #------------- Metrics -------------
            st.divider()
            st.markdown(
                "<h4 style='color: #ffffff;'>Model Performance</h4>",
                unsafe_allow_html=True
            )

            accuracy = 74.7
            precision = 85.3
            recall = 75.8
            f1_score = 80.3

            col1, col2, col3, col4 = st.columns(4)
            col1.pyplot(plot_circular_metric("Accuracy", accuracy))
            col2.pyplot(plot_circular_metric("Precision", precision))
            col3.pyplot(plot_circular_metric("Recall", recall))
            col4.pyplot(plot_circular_metric("F1 Score", f1_score))

            st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

            cm = np.array([
                [6878, 2635],  # Actual: No Loss
                [4893, 15302]  # Actual: Loss
            ])

            labels = ["No Loss", "Loss"]

            fig = plt.figure(figsize=(4, 4), dpi=100)
            fig.set_facecolor("#1c1e26")  # Set figure background to lighter dark
            ax = fig.add_subplot(111)     # Create axes explicitly
            ax.set_facecolor("#1c1e26")   # Set axes background to match

            # Plot confusion matrix with seaborn heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',                  # Display integers
                cmap='Blues',             # Color scheme for cells
                xticklabels=labels,
                yticklabels=labels,
                cbar=True,               # No colorbar for compactness
                linewidths=1,             # Cell separation lines
                linecolor="#2a2a2a",      # Slightly lighter dark for lines
                annot_kws={"color": "black", "fontsize": 11},  # White annotations
                ax=ax
            )

            # Customize labels and ticks
            ax.set_xlabel("Predicted", fontsize=10, color="white", labelpad=10)
            ax.set_ylabel("Actual", fontsize=10, color="white", labelpad=10)
            ax.set_title("Confusion Matrix", fontsize=11, color="white", pad=12)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            # Adjust layout to minimize margins
            fig.tight_layout(pad=0)

            # Display in Streamlit (adjust if using a different method)
            st.pyplot(fig)



    # --------------------------- LSTM tab ----------------------------------
    with tabs[1]:
        st.subheader("LSTM")
        st.markdown("""
        <div style='font-size: 1.05rem; color: #dddddd; line-height: 1.6; padding-bottom: 1rem;'>
            This model uses a <span style='color: #1f5c1a;'>Long Short-Term Memory (LSTM)</span> neural network to forecast 
            <em>future deforestation risk</em> across Alberta. It learns from historical satellite data and 
            predicts likely forest loss zones for each year between <strong>2024 and 2030</strong>.
            <br><br>
            Select a year below to view predicted high-risk areas as a heatmap. These forecasts are probabilistic 
            and should be interpreted as early-warning indicators, not guarantees.
        </div>
        """, unsafe_allow_html=True)

        st.warning("Note: Since we lack actual satellite data for future years, the model was trained using \
                simulated or projected input features. As a result, predictions beyond 2023 are indicative \
                and may not reflect future outcomes precisely.")

        with st.spinner("Processing selected area..."):
            selected_year = st.radio("Select year:", years, horizontal=True)
            df_year = lstm_df[(lstm_df["year"] == selected_year) & (lstm_df["predicted_loss"] == 1)]

            if not df_year.empty:
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=df_year,
                    get_position="[lon, lat]",
                    radiusPixels=60,
                    intensity=1,
                    threshold=0.2,
                    get_weight=1,
                    opacity=0.8,
                )

                view_state = pdk.ViewState(latitude=53.5, longitude=-115.5, zoom=5, pitch=0)

                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        map_style="mapbox://styles/mapbox/dark-v11",
                    )
                )
            else:
                st.warning("No predicted deforestation points for the selected year.")

            st.caption("""
            Predictions represent areas at high risk of deforestation based on temporal modeling of satellite imagery. Results
            are indicative, not definitive.
            """)

            st.divider()
            st.markdown(
                "<h4 style='color: #ffffff;'>Model Performance</h4>",
                unsafe_allow_html=True
            )

            accuracy = 67.7
            precision = 73.7
            recall = 76.1
            f1_score = 74.9
            auc = 73.8

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.pyplot(plot_circular_metric("Accuracy", accuracy))
            col2.pyplot(plot_circular_metric("Precision", precision))
            col3.pyplot(plot_circular_metric("Recall", recall))
            col4.pyplot(plot_circular_metric("F1 Score", f1_score))
            col5.pyplot(plot_circular_metric("AUC", auc))

            st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

            cm = np.array([
                [7624, 6770],  # Actual: No Loss
                [5966, 19029]  # Actual: Loss
            ])

            labels = ["No Loss", "Loss"]

            fig = plt.figure(figsize=(4, 4), dpi=100)
            fig.set_facecolor("#1c1e26")  # Set figure background to lighter dark
            ax = fig.add_subplot(111)     # Create axes explicitly
            ax.set_facecolor("#1c1e26")   # Set axes background to match

            # Plot confusion matrix with seaborn heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',                  # Display integers
                cmap='Blues',             # Color scheme for cells
                xticklabels=labels,
                yticklabels=labels,
                cbar=True,               # No colorbar for compactness
                linewidths=1,             # Cell separation lines
                linecolor="#2a2a2a",      # Slightly lighter dark for lines
                annot_kws={"color": "black", "fontsize": 11},  # White annotations
                ax=ax
            )

            # Customize labels and ticks
            ax.set_xlabel("Predicted", fontsize=10, color="white", labelpad=10)
            ax.set_ylabel("Actual", fontsize=10, color="white", labelpad=10)
            ax.set_title("Confusion Matrix", fontsize=11, color="white", pad=12)
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            # Adjust layout to minimize margins
            fig.tight_layout(pad=0)

            # Display in Streamlit (adjust if using a different method)
            st.pyplot(fig)

# ----------------------------------------------------------------------------
# Insights (placeholder)
# ----------------------------------------------------------------------------
elif section == "Insights":
    st.header("Key Insights")
    st.subheader("🧠 Understanding What Drives Deforestation")

    st.markdown("""
    <div style='font-size: 1.05rem; color: #dddddd; line-height: 1.6; padding-bottom: 1rem;'>
        Below is a breakdown of the top predictive features used by our <strong>XGBoost classifier</strong> model.
        These features were extracted from satellite-derived land cover and geospatial layers.
        Understanding their importance provides insight into the factors most strongly associated with
        deforestation risk across Alberta.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)


    st.image("feature-imp.png", caption="Figure: Summary of regional deforestation patterns (2024–2030).", use_container_width=True)

    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    ### 🔍 Feature Glossary (Top Predictors Explained)

    **🌾 `dw_shrub_and_scrub`**  
    Percentage of land covered by shrubs and scrub. High values often signal transitional ecosystems prone to degradation.

    **🌳 `dw_trees`**  
    Proportion of tree cover in an area. Dense forest zones are closely monitored for logging and clearing risk.

    **📈 `gain`**  
    Areas where vegetation has regrown or increased. Gain can be misleading in zones with recurring disturbance.

    **🌿 `dw_grass`**  
    Extent of grasslands. These may expand when forests are cleared, especially for pasture or fire-prone areas.

    **🏙️ `dw_built`**  
    Urban and built-up land. Construction and development near forests often drive tree loss.

    **🌾 `dw_crops`**  
    Agricultural coverage. A major factor behind deforestation due to farmland expansion.

    **🛰️ `B8`**  
    Sentinel-2 Band 8 (near-infrared). Used to detect vegetation density and structural changes in canopy.

    **⛰️ `elevation`**  
    Height above sea level. Lower elevations are more prone to development and deforestation.

    **📐 `slope`**  
    Terrain steepness. Flatter areas are more accessible and often more deforested.

    **💧 `dw_water`**  
    Presence of water bodies. Lakes and rivers can attract agriculture and human settlement nearby.

    **🌄 `slope_stddev_5x5`**  
    Ruggedness over a 5x5 pixel window. Captures terrain variability that affects land use feasibility.

    **💦 `NDWI`**  
    Normalized Difference Water Index. Detects water content in vegetation or nearby bodies.

    **🌱 `NDVI`**  
    Normalized Difference Vegetation Index. A common vegetation health indicator — low NDVI may signal deforestation.

    **❄️ `dw_snow_and_ice`**  
    Snow and ice coverage. Typically associated with protected, less-disturbed regions.

    **🧱 `slope_stddev_3x3`**  
    Local slope variability at a finer scale. Helps identify minor terrain shifts that may influence land use.
    """)

    st.divider()
    
    st.markdown(
        """
        <h4 style='text-align: center; color: #bbbbbb;'>
            More visualisations coming soon...
        </h4>
        """
    , unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# About
# ----------------------------------------------------------------------------
else:  # About
    st.header("📘 About the Deforestation Dashboard")

    st.markdown("""
    <div style='color: #dddddd; font-size: 1.05rem; line-height: 1.7;'>
    This dashboard is part of the <strong style='color: #2bd47d;'>DATA 501 capstone project</strong> at the <strong style='color: #2bd47d;'>University of Calgary</strong>, aimed at analyzing and visualizing forest loss
    across <strong>Alberta, Canada</strong> using deep learning and satellite imagery. The app provides users with tools to:
    <br><br>
    <ul style='padding-left: 1.5rem;'>
        <li>🌲 Explore annual deforestation trends using the <strong>Global Forest Change</strong> (Hansen) dataset</li>
        <li>🧠 Visualize high‑risk areas predicted by <strong>machine learning models</strong> (including an LSTM)</li>
        <li>📍 Interactively select custom regions to analyze <strong>historical forest loss</strong></li>
        <li>📊 Compare predictions across multiple model types (classifier, LSTM)</li>
    </ul>
    <h4 style='color: #2bd47d;'>🛠 Technologies Used</h4>
    <ul style='padding-left: 1.5rem;'>
        <li>📦 <strong>Streamlit</strong> & PyDeck for interactive UI and mapping</li>
        <li>🧪 <strong>Scikit-learn</strong> and <strong>TensorFlow</strong> for ML modeling</li>
        <li>🛰️ <strong>Google Earth Engine</strong> for geospatial data and satellite imagery</li>
        <li>📊 <strong>Pandas, NumPy, Shapely, Folium</strong> for data manipulation</li>
    </ul>
    <h4 style='color: #2bd47d;'>📂 Data Sources</h4>
    <ul style='padding-left: 1.5rem;'>
        <li>🗺️ Hansen et al. <em>Global Forest Change</em> v1.10 (2000–2023)</li>
        <li>🛰️ Satellite imagery processed via Earth Engine pipelines</li>
    </ul>
    <h4 style='color: #2bd47d;'>👨‍💻 Authors</h4>
    <ul style='padding-left: 1.5rem;'>
        <li><strong>Arij Ashar</strong> | B.S. Computer Science (Data Science minor) | <a href="https://www.linkedin.com/in/arij-ashar/" target="_blank" style='color: #2bd47d; text-decoration: none;'>Connect on LinkedIn</a></li>
        <li><strong>Yurii Bezborodov</strong> | B.S. Data Science | <a href="https://www.linkedin.com/in/yurii-bezborodov/" target="_blank" style='color: #2bd47d; text-decoration: none;'>Connect on LinkedIn</a></li>
    </ul>
    <p style='margin-top: 1.5rem;'>
        For feedback, contributions, or to learn more, feel free to reach out via LinkedIn.
    </p>
    </div>
    """, unsafe_allow_html=True)
