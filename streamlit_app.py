import streamlit as st
import networkx as nx
import json
import folium
from streamlit_folium import folium_static
import numpy as np
import os 

# --- 0. Configuration and Coordinate Mapping ---

# Fixed center point for default map view (Central India)
INDIA_CENTER = [23.0, 78.0] 

# 1. REAL, MANUALLY DEFINED COORDINATES FOR HABITAT NODES (Lat, Lon)
# These coordinates ensure the points appear in the correct geographic area of India.
REAL_COORDINATES = {
    # Node ID (float) : [Latitude (Y), Longitude (X)]
    0.0: [25.939, 76.438],  # Ranthambore National Park (Rajasthan)
    1.0: [21.080, 70.835],  # Gir Forest (Gujarat)
    2.0: [21.900, 89.000],  # Sundarbans Delta (West Bengal)
    3.0: [26.700, 93.300],  # Kaziranga Grasslands (Assam)
    4.0: [13.200, 75.000],  # Western Ghats Corridor (General region)
    5.0: [22.450, 78.430],  # Satpura Tiger Reserve (MP)
    6.0: [28.600, 77.200],  # Delhi-NCR Urban Fringe (Symbolic high resistance near capital)
    7.0: [22.330, 80.620],  # Kanha Tiger Reserve (MP)
    8.0: [21.660, 79.330],  # Pench Forest (MP/Maharashtra)
    9.0: [11.830, 76.500]   # Bandipur-Nagarhole Reserve (Karnataka)
}

# Descriptive Node Labels (India Focused)
NODE_LABELS = {
    0.0: "Ranthambore National Park (Rajasthan)",
    1.0: "Gir Forest (Gujarat)",
    2.0: "Sundarbans Delta (West Bengal)",
    3.0: "Kaziranga Grasslands (Assam)",
    4.0: "Western Ghats Corridor",
    5.0: "Satpura Tiger Reserve (MP)",
    6.0: "Delhi-NCR Urban Fringe (High Resistance)",
    7.0: "Kanha Tiger Reserve (MP)",
    8.0: "Pench Forest (MP/Maharashtra)",
    9.0: "Bandipur-Nagarhole Reserve (Karnataka)",
}
REVERSE_MAP = {v: k for k, v in NODE_LABELS.items()}


# --- 1. Data Loading and Caching ---

@st.cache_data
def load_corridor_graph(file_path="corridor_graph.json"):
    """Loads the NetworkX graph and uses hardcoded REAL_COORDINATES."""
    full_path = os.path.abspath(file_path)
    
    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
        
        # CORE FIX: Inject REAL_COORDINATES, overriding synthetic x/y
        for node_id in G.nodes():
            if node_id in REAL_COORDINATES:
                # The node must be float for consistent mapping
                float_id = float(node_id)
                lat, lon = REAL_COORDINATES[float_id]
                
                # Store the real coordinates in the graph attributes for Folium
                G.nodes[float_id]['lat'] = lat
                G.nodes[float_id]['lon'] = lon
            else:
                # Fallback if a node ID exists in graph but not in our list
                G.nodes[node_id]['lat'] = INDIA_CENTER[0]
                G.nodes[node_id]['lon'] = INDIA_CENTER[1]
        
        # Get the sorted list of descriptive labels for the dropdowns
        float_ids = sorted([float(n) for n in G.nodes()])
        descriptive_labels = [NODE_LABELS.get(i, f"Node {i}") for i in float_ids]
        
        return G, descriptive_labels
    except FileNotFoundError:
        st.error(f"Error: Graph file not found. Please ensure 'corridor_graph.json' is in the same directory as this script and run the pipeline first.")
        return None, []
    except Exception as e:
        st.error(f"Error loading graph data: {e}")
        return None, []

# --- 2. Pathfinding Logic (Unchanged) ---

def find_path(G, source_float, target_float):
    """Runs Dijkstra's algorithm for the least-cost path."""
    if G is None: return []
    try:
        path = nx.dijkstra_path(G, source=source_float, target=target_float, weight="weight")
        return path
    except nx.NetworkXNoPath:
        st.warning("No valid path found.")
        return []
    except Exception as e:
        st.error(f"Pathfinding error: {e}")
        return []

# --- 3. Streamlit App Interface ---

st.set_page_config(layout="wide")
st.title("🗺️ AI-Enhanced Wildlife Corridor Finder (India Focus)")
st.markdown("Select a **Start** and **End Habitat** to calculate the least-cost corridor for wildlife movement. The habitat points are mapped to approximate locations within India.")


GRAPH, DESCRIPTIVE_LABELS = load_corridor_graph()

if GRAPH:
    # Force rerun to clear cache if the file loads successfully
    if 'load_success' not in st.session_state:
        st.session_state['load_success'] = True
        st.rerun()

    col1, col2 = st.columns([1, 2]) # Split layout

    with col1:
        st.subheader("Select Path Endpoints")
        
        # Set default index for target to be the last node in the list
        default_target_index = len(DESCRIPTIVE_LABELS) - 1
        
        source_label = st.selectbox("Start Habitat (Source)", DESCRIPTIVE_LABELS, key='src')
        target_label = st.selectbox("End Habitat (Target)", DESCRIPTIVE_LABELS, index=default_target_index, key='tgt')
        
        if st.button("Find Least-Cost Corridor", type="primary"):
            st.session_state['run_analysis'] = True
            
        if st.button("Clear Map"):
            st.session_state['run_analysis'] = False
            st.cache_data.clear() 
            st.rerun()
            
        st.markdown("""
        ***
        **Note:** Coordinates for the 10 habitats are manually defined for accuracy in India.
        """)

    with col2:
        st.subheader("Corridor Map & Analysis")
        
        if 'run_analysis' not in st.session_state:
             st.session_state['run_analysis'] = False
        
        if st.session_state['run_analysis']:
            
            source_float = REVERSE_MAP[source_label]
            target_float = REVERSE_MAP[target_label]

            path_nodes_float = find_path(GRAPH, source_float, target_float)
            
            if path_nodes_float:
                
                # --- Map and Cost Calculation ---
                path_coords = []
                for node_id in path_nodes_float:
                    # FETCH REAL LAT/LON FROM THE UPDATED GRAPH
                    lat = GRAPH.nodes[node_id]['lat']
                    lon = GRAPH.nodes[node_id]['lon']
                    path_coords.append((lat, lon))
                
                cost = nx.shortest_path_length(GRAPH, source_float, target_float, weight='weight')
                path_labels = [NODE_LABELS[i] for i in path_nodes_float]
                
                # --- Map Center Fix: Center map based on the path coordinates ---
                path_lats = [c[0] for c in path_coords]
                path_lons = [c[1] for c in path_coords]
                map_center_path = [np.mean(path_lats), np.mean(path_lons)]
                
                # --- Display Results ---
                st.success(f"Path Found! Total Ecological Cost: **{cost:.4f}** (Lower is better)")
                st.info(f"Corridor Sequence: **{' → '.join(path_labels)}**")

                # --- Map Visualization (Folium) ---
                m = folium.Map(location=map_center_path, zoom_start=4) 

                # Add the corridor line
                folium.PolyLine(path_coords, color="red", weight=4, opacity=0.8, tooltip="Least-Cost Path").add_to(m)

                # Add markers for all nodes
                for node_id in GRAPH.nodes():
                    n_lat = GRAPH.nodes[node_id]['lat']
                    n_lon = GRAPH.nodes[node_id]['lon']
                    
                    marker_color = 'green' if node_id == source_float else ('red' if node_id == target_float else 'blue')
                    
                    folium.Marker(
                        [n_lat, n_lon], 
                        popup=NODE_LABELS.get(node_id), 
                        icon=folium.Icon(color=marker_color, icon='info-sign')
                    ).add_to(m)
                
                folium_static(m, width=800, height=600)
            else:
                st.warning("Could not find a path between the selected habitats.")
        else:
             # Default map view (India) when no analysis is running
             m = folium.Map(location=INDIA_CENTER, zoom_start=4)
             folium_static(m, width=800, height=600)
             st.info("Select habitat endpoints and click 'Find Least-Cost Corridor' to begin the analysis.")