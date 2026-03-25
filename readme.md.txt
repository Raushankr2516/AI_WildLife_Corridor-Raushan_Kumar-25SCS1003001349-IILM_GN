# AI-Enhanced Wildlife Corridor Pipeline

This project is a comprehensive 70-task pipeline designed to model, analyze, and visualize wildlife corridors in India. It includes a backend data processing engine and an interactive Streamlit dashboard.

## 🚀 Features
- **Data Pipeline:** 70 automated tasks covering data cleaning, ML modeling, and Graph Theory (Least-Cost Path).
- **Interactive Map:** A Streamlit-based dashboard using Folium to visualize habitat connectivity.
- **Predictive Analytics:** Uses XGBoost and TensorFlow/CNNs to predict habitat suitability.

## 📂 File Structure
- `wildlife_corridor_pipeline.py`: The main 70-task data processing script.
- `streamlit_app.py`: The interactive web dashboard.
- `corridor_graph.json`: Pre-computed graph data for the map.
- `connectivity.csv`: Matrix data for habitat relationships.

## 🛠️ Setup and Execution

### 1. Online VS Code (GitHub Codespaces)
1. Upload this folder to a GitHub repository.
2. Click **Code** > **Codespaces** > **Create codespace**.
3. Once the terminal is ready, install dependencies:
   ```bash
   pip install -r requirements.txt