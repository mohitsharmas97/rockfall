# AI-Powered Rockfall Prediction System

[](https://www.python.org/downloads/)
[](https://flask.palletsprojects.com/)
[](https://pytorch.org/)
[](LICENSE.md)

An intelligent system developed for the Smart India Hackathon (SIH) to proactively predict rockfall events in open-pit mines. By leveraging time-series analysis with a deep learning model, we shift from reactive monitoring to a predictive safety framework, safeguarding personnel and equipment.

-----

## Core Idea

Traditional safety protocols often rely on manual inspection or post-event analysis. Our system transforms this paradigm by analyzing temporal data patterns leading up to a potential rockfall. We process historical weather and topographical data into **10-day sequences**, feeding them into a **Bidirectional LSTM with an Attention mechanism**. This model learns to identify subtle, critical precursors to instability, providing a probability-based forecast of rockfall risk.

-----

## System Architecture

The project follows a modular architecture, from raw data ingestion to a user-facing web application.

1.  **Data Acquisition**: Collects historical landslide data from the NASA Global Landslide Catalog and corresponding weather data.
2.  **Preprocessing & Feature Engineering**: Cleans, integrates, and transforms the data into time-series sequences (`X_lstm_ready.npy`, `y_lstm_ready.npy`) suitable for the LSTM model.
3.  **Deep Learning Model**: A trained Bi-LSTM with Attention model (`rockfall_bilstm_attention.pth`) serves as the predictive core, analyzing sequences to forecast risk.
4.  **Backend API**: A **Flask** server (`app.py`) wraps the model, exposing prediction endpoints.
5.  **Frontend Dashboard**: An intuitive web interface for users to input data, view predictions, and visualize risk zones.

   

-----

## Project Structure

The repository is organized to separate data, notebooks, and application source code for clarity and maintainability.

```
.
├── 📄 .gitignore
├── 📄 README.md
├── 📁 data/
│   ├── 📄 enhanced_landslide_dataset.csv
│   ├── 📄 featured_weather_data.csv
│   ├── 📄 X_lstm_ready.npy      # Feature data (sequences) for the model
│   └── 📄 y_lstm_ready.npy      # Target data for the model
├── 📁 notebook/
│   └── 📄 rockfall_lstm.ipynb   # Jupyter Notebook for experimentation & model dev
├── 📁 src/
│   └── 📁 backend/
│       ├── 📁 static/         # CSS, JS files
│       ├── 📁 templates/      # HTML templates for Flask
│       ├── 🐍 app.py          # Main Flask application
│       ├── 🐍 combine.py      # Utility scripts
│       └── ...
├── 🧠 rockfall_bilstm_attention.pth  # The trained PyTorch model
├── 🔧 scaler.gz                     # The fitted data scaler
└── ...
```

-----

## Tech Stack

  - **Backend**: **Flask**
  - **Deep Learning**: **PyTorch**
  - **Data Manipulation**: **Pandas, NumPy**
  - **Geospatial**: **Rasterio, earthaccess**
  - **Machine Learning**: **Scikit-learn**
  - **Development**: **Jupyter Notebook**

-----

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

  - Python 3.8 or higher
  - A virtual environment tool like `venv` or `conda`

### Installation & Execution

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/mohitsharmas97/rockfall.git
    cd rockfall
    ```

2.  **Create and activate a virtual environment:**

    ```sh
    # For venv
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # For conda
    conda create --name rockfall_env python=3.8
    conda activate rockfall_env
    ```

3.  **Install the required dependencies:**
    

    ```sh
    pip install flask torch pandas numpy scikit-learn
    ```

4.  **Navigate to the backend directory and run the Flask application:**

    ```sh
    cd src/backend
    python app.py
    ```

5.  **Open your browser** and go to `http://127.0.0.1:5000` to see the application in action\!

-----

## Project Workflow

Our project was developed in four key phases, resulting in a fully integrated system.

### Phase 1: Data Acquisition & Preprocessing (✅ Completed)

  - **Input Data**: Utilized the NASA Global Landslide Catalog.
  - **Geographic Filtering**: Isolated events in high-risk zones to create a relevant dataset.
  - **Data Enrichment**: Fetched and integrated elevation and historical weather data for each event.
  - **Output**: A clean, feature-rich dataset ready for sequence creation.

### Phase 2: Time-Series Feature Engineering (✅ Completed)

  - **Sequence Generation**: Transformed the static data into **10-day sequences**, capturing the temporal dynamics leading up to each landslide event.
  - **Data Scaling**: Normalized the features using a standard scaler to ensure model stability.
  - **Output**: The final `X_lstm_ready.npy` and `y_lstm_ready.npy` files used for training.

### Phase 3: Model Development & Training (✅ Completed)

  - **Model Selection**: Chose a **Bidirectional LSTM with an Attention mechanism** to effectively capture long-range dependencies and focus on the most influential time steps.
  - **Training**: Trained the model on the prepared time-series data until convergence.
  - **Evaluation**: Assessed the model's performance using metrics like Accuracy, Precision, Recall, and F1-Score to confirm its predictive power.

### Phase 4: Deployment & Visualization (✅ Completed)

  - **API Development**: Encapsulated the trained model and scaler into a lightweight **Flask API** (`app.py`).
  - **Dashboard Creation**: Developed a user-friendly web dashboard using HTML/CSS and Flask's template engine to visualize predictions.
  - **Alert System**: Implemented a basic alert mechanism within the frontend to notify users of high-risk predictions.

-----

## Future Scope

  - **Real-time Data Integration**: Integrate live weather data and geotechnical sensor feeds (displacement, strain) for on-the-fly predictions.
  - **Model Enhancement**: Experiment with more advanced architectures like Transformers for time-series forecasting.
  - **Cloud Deployment**: Deploy the application on a cloud platform (AWS, GCP, Azure) for scalability and accessibility.
  - **Advanced Visualization**: Create more detailed risk maps with temporal heatmaps and interactive charts.

-----

## Contributors
  - [NotAceNinja](https://github.com/pushkar-hue)
  - [mohitsharmas97](https://github.com/mohitsharmas97)

-----

##  Acknowledgments

  - **Smart India Hackathon** for providing the platform and opportunity.
  - **NASA** for making the Global Landslide Catalog and NASADEM datasets publicly available.
