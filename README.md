# 🌲 Predicting High-Risk Deforestation Zones in Alberta

This project is a Streamlit-based dashboard that visualizes and predicts deforestation risk across a region in Alberta, Canada. Using satellite-derived features like NDVI, Dynamic World land cover probabilities, and topographic data, it allows users to explore historical trends and forecast deforestation events using machine learning models (XGBoost and LSTM).

## ⚙️ How to Run

### ✅ Requirements
- Python 3.10 or higher
- pip 21.0 or higher

### 📦 Installation and Running
```bash
git clone https://github.com/yourusername/deforestation-dashboard.git
cd deforestation-dashboard
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
