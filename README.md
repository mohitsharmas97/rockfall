# ⛰️ AI-Powered Rockfall Prediction System

##  About The Project

This project is an AI-based system designed to predict potential rockfall incidents in open-pit mines, developed for the Smart India Hackathon (SIH). Rockfalls pose a significant threat to personnel and equipment, and our goal is to move from reactive monitoring to proactive, predictive analysis for enhanced safety and operational efficiency.

The system processes multi-source data inputs, starting with topographical and environmental data, to train a machine learning model that can identify patterns preceding rockfall events.

###  Core Idea

Our approach is to build a robust dataset by combining historical landslide data with relevant geographical and environmental features. This enriched dataset will be the foundation for training a predictive model capable of assessing rockfall risk in real-time.

The initial phase, demonstrated in our data processing notebook, focuses on:
* Processing the **NASA Global Landslide Catalog**, which contains over 11,000 recorded events.
* Filtering these events to a high-risk geological hotspot **(South Asia / Himalayas)**, creating a focused dataset of ~2,500 events.
* Programmatically downloading corresponding **Digital Elevation Model (DEM)** tiles from NASA's NASADEM dataset.
* Enriching the landslide data by extracting the precise **elevation** for each event's coordinates.

This creates a foundational dataset ready for further feature engineering and model training.

## 🛠️ Tech Stack

* **Data Processing & Analysis:** Python, Pandas, Rasterio
* **Geospatial Data Acquisition:** `earthaccess` (for NASA Earthdata API)
* **Development Environment:** Jupyter Notebook, Google Colab

##  Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

1.  **Python 3.8+**
2.  **NASA Earthdata Account:** You need a free account to download the elevation data. You can register at [URS Earthdata](https://urs.earthdata.nasa.gov/users/new). The `earthaccess` library will prompt you to log in interactively the first time you run the script.
3.  **NASA Global Landslide Catalog:** Download the dataset and place it in the project's root directory.
    * File needed: `Global_Landslide_Catalog_Export_rows.csv`

### Installation

1.  **Clone the repo**
    ```sh
    git clone [https://github.com/your_username/your_repository.git](https://github.com/your_username/your_repository.git)
    cd your_repository
    ```

2.  **Create a virtual environment (recommended)**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies from `requirements.txt`**
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should contain the following libraries)*
    ```
    pandas
    earthaccess
    rasterio
    tqdm
    jupyter
    ```

### Execution

1.  Ensure the `Global_Landslide_Catalog_Export_rows.csv` file is in the same directory as the notebook.
2.  Launch the Jupyter Notebook:
    ```sh
    jupyter notebook
    ```
3.  Open the `.ipynb` file and run the cells sequentially. The script will handle authentication, data filtering, downloading, and processing automatically.

## 📈 Project Workflow

Our project is structured in four key phases, moving from data collection to a fully deployed predictive system.

**Phase 1: Data Acquisition & Preprocessing ( Completed)**
* **Input Data:** Utilize the NASA Global Landslide Catalog as the source of historical events.
* **Geographic Filtering:** Isolate landslides in our defined hotspot boundary (`lon: 60 to 98`, `lat: -2 to 39`) to create a high-quality, relevant dataset.
* **Topographical Data Acquisition:** Use the `earthaccess` library to automatically download all necessary NASADEM HGT elevation tiles for the hotspot region.
* **Data Enrichment:** Extract the precise elevation for each landslide's latitude and longitude using the downloaded DEM tiles.
* **Output:** A clean CSV file (`landslides_hotspot_with_elevation.csv`) ready for the next phase.

**Phase 2: Feature Engineering & Data Integration (⏳ In Progress)**
* **Weather Data Integration:** Use a weather API (e.g., Open-Meteo) to fetch historical data (rainfall, temperature) for the date and location of each landslide.
* **Topographical Feature Creation:** Calculate advanced features from the DEM tiles, such as **slope**, **aspect** (the direction a slope faces), and **terrain roughness**, which are critical indicators for rockfall risk.
* **Geotechnical Data (Future Scope):** For a real-world mine, integrate sensor data like displacement, strain, and pore pressure.

**Phase 3: Model Development & Training**
* **Model Selection:** Experiment with robust classification models like **XGBoost, LightGBM, or Random Forest** to predict rockfall risk (e.g., High, Medium, Low).
* **Training:** Train the model on the comprehensive dataset created in the previous phases.
* **Evaluation:** Rigorously evaluate the model's performance using metrics like Accuracy, Precision, Recall, and F1-Score to ensure its reliability.

**Phase 4: Deployment & Visualization**
* **API Development:** Wrap the trained model in a lightweight API using **Flask or FastAPI**. This API will accept new data (e.g., current conditions at a specific mine location) and return a risk prediction.
* **Dashboard Creation:** Develop a user-friendly web dashboard (using **Streamlit or Dash**) to visualize vulnerable zones on a map, display probability-based forecasts, and allow mine planners to analyze risks.
* **Alert System:** Implement an automated alert mechanism via **SMS (Twilio) or email (SendGrid)** to notify stakeholders when the predicted risk level exceeds a critical threshold, along with suggested action plans.

---
