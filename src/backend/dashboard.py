# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# import time

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Landslide Alert System",
#     page_icon="⛰️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- Custom CSS ---
# st.markdown("""
# <style>
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin: 0.5rem 0;
#     }
#     .stAlert > div {
#         padding: 1rem;
#     }
#     .risk-very-high {
#         background-color: #ffebee;
#         border-left: 5px solid #d32f2f;
#         padding: 10px;
#         margin: 10px 0;
#     }
#     .risk-high {
#         background-color: #fff3e0;
#         border-left: 5px solid #f57c00;
#         padding: 10px;
#         margin: 10px 0;
#     }
#     .risk-moderate {
#         background-color: #fffde7;
#         border-left: 5px solid #fbc02d;
#         padding: 10px;
#         margin: 10px 0;
#     }
#     .risk-low {
#         background-color: #e8f5e9;
#         border-left: 5px solid #388e3c;
#         padding: 10px;
#         margin: 10px 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- App Title and Description ---
# st.title("⛰️ Landslide Early Warning System")
# st.markdown("""
# **Advanced AI-powered landslide risk assessment** using BiLSTM neural networks trained on historical weather patterns,
# topographical data, and geological factors. Get real-time risk predictions for any location worldwide.
# """)

# # --- API Configuration ---
# API_BASE_URL = "http://127.0.0.1:5000"

# # --- Helper Functions ---
# @st.cache_data(ttl=300)  # Cache for 5 minutes
# def check_api_health():
#     """Check if the API is running."""
#     try:
#         response = requests.get(f"{API_BASE_URL}/health", timeout=5)
#         if response.status_code == 200:
#             return True, response.json()
#         return False, None
#     except requests.exceptions.ConnectionError:
#         return False, None
#     except Exception as e:
#         return False, str(e)

# def get_prediction(lat, lon):
#     """Get landslide prediction from API without caching."""
#     try:
#         # Validate inputs
#         lat = float(lat)
#         lon = float(lon)
        
#         if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
#             return False, "Invalid coordinates. Latitude must be between -90 and 90, longitude between -180 and 180."
        
#         payload = {'lat': lat, 'lon': lon}
        
#         # Make request with longer timeout for weather data fetching
#         response = requests.post(
#             f"{API_BASE_URL}/predict", 
#             json=payload, 
#             timeout=60  # Increased timeout
#         )
        
#         if response.status_code == 200:
#             return True, response.json()
#         else:
#             try:
#                 error_data = response.json()
#                 error_msg = error_data.get('error', 'Unknown error')
#             except:
#                 error_msg = response.text
#             return False, f"API Error ({response.status_code}): {error_msg}"
            
#     except requests.exceptions.Timeout:
#         return False, "Request timed out. The weather data API might be slow. Please try again."
#     except requests.exceptions.ConnectionError:
#         return False, "Cannot connect to the backend server. Please ensure it's running."
#     except requests.exceptions.RequestException as e:
#         return False, f"Request failed: {str(e)}"
#     except ValueError as e:
#         return False, f"Invalid input: {str(e)}"
#     except Exception as e:
#         return False, f"Unexpected error: {str(e)}"

# def get_risk_color(risk_level):
#     """Get color based on risk level."""
#     colors = {
#         "Very High Risk": "#d32f2f",
#         "High Risk": "#f57c00",
#         "Moderate Risk": "#fbc02d",
#         "Low Risk": "#388e3c"
#     }
#     return colors.get(risk_level, "#757575")

# def create_risk_gauge(probability, risk_level):
#     """Create a risk gauge chart."""
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number+delta",
#         value=probability * 100,
#         domain={'x': [0, 1], 'y': [0, 1]},
#         title={'text': "Risk Probability", 'font': {'size': 24}},
#         delta={'reference': 44, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
#         gauge={
#             'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
#             'bar': {'color': get_risk_color(risk_level)},
#             'steps': [
#                 {'range': [0, 30], 'color': "#e8f5e9"},
#                 {'range': [30, 50], 'color': "#fff3e0"},
#                 {'range': [50, 70], 'color': "#ffecb3"},
#                 {'range': [70, 100], 'color': "#ffebee"}
#             ],
#             'threshold': {
#                 'line': {'color': "red", 'width': 4},
#                 'thickness': 0.75,
#                 'value': 44
#             }
#         }
#     ))
#     fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
#     return fig

# def format_precipitation(value):
#     """Format precipitation value safely."""
#     if value is None or value == "N/A":
#         return "N/A"
#     if isinstance(value, str):
#         # If it already has 'mm' suffix, return as is
#         if 'mm' in value:
#             return value
#         # Try to parse and format
#         try:
#             num_value = float(value)
#             return f"{num_value:.1f}mm"
#         except:
#             return value
#     try:
#         return f"{float(value):.1f}mm"
#     except:
#         return "N/A"

# # --- Session State Initialization ---
# if 'prediction_result' not in st.session_state:
#     st.session_state.prediction_result = None
# if 'last_coords' not in st.session_state:
#     st.session_state.last_coords = (None, None)
# if 'prediction_history' not in st.session_state:
#     st.session_state.prediction_history = []

# # --- Sidebar ---
# with st.sidebar:
#     st.header("🔧 System Status")
    
#     # Check API health
#     api_healthy, health_data = check_api_health()
    
#     if api_healthy and health_data:
#         st.success("✅ API Connected")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Model", "✅ Ready" if health_data.get('model_loaded') else "❌ Error")
#         with col2:
#             st.metric("Scaler", "✅ Ready" if health_data.get('scaler_loaded') else "❌ Error")
        
#         if 'device' in health_data:
#             st.info(f"🖥️ Device: {health_data['device']}")
#     else:
#         st.error("❌ API Disconnected")
#         st.warning("Please ensure the backend server is running:")
#         st.code("python app.py", language="bash")
    
#     st.divider()
    
#     st.header("📍 Quick Locations")
#     quick_locations = {
#         "Kukas, Rajasthan": (27.0511, 75.8851),
#         "Shimla, Himachal Pradesh": (31.1048, 77.1734),
#         "Darjeeling, West Bengal": (27.0360, 88.2627),
#         "Ooty, Tamil Nadu": (11.4064, 76.6932),
#         "Mussoorie, Uttarakhand": (30.4598, 78.0664),
#         "Manali, Himachal Pradesh": (32.2396, 77.1887),
#         "Gangtok, Sikkim": (27.3389, 88.6065),
#         "Cherrapunji, Meghalaya": (25.2703, 91.7334),
#         "Custom Location": None
#     }
#     selected_location = st.selectbox("Choose a location:", list(quick_locations.keys()))
    
#     st.divider()
    
#     st.header("ℹ️ About")
#     with st.expander("Model Information"):
#         st.markdown("""
#         **Model Architecture:**
#         - Bidirectional LSTM Network
#         - 14 input features
#         - 10-day temporal sequence
#         - Trained on historical landslide data
        
#         **Features Used:**
#         - Weather: Precipitation, temperature, humidity
#         - Soil: Moisture at multiple depths
#         - Terrain: Elevation, slope, aspect
#         """)
    
#     st.header("⚠️ Disclaimer")
#     st.warning(
#         "This is a predictive model for informational purposes only. "
#         "Always consult official geological surveys and "
#         "local authorities for emergency decisions."
#     )

# # --- Main Content ---
# st.header("📍 Location Input")

# # Determine default coordinates
# default_lat, default_lon = (27.0511, 75.8851)  # Default to Kukas
# if selected_location != "Custom Location" and quick_locations[selected_location]:
#     default_lat, default_lon = quick_locations[selected_location]
#     st.info(f"📍 Selected: **{selected_location}** ({default_lat:.4f}, {default_lon:.4f})")

# # Input fields
# col1, col2, col3 = st.columns([2, 2, 1])
# with col1:
#     lat = st.number_input(
#         "Latitude", 
#         min_value=-90.0, 
#         max_value=90.0, 
#         value=float(default_lat), 
#         step=0.0001, 
#         format="%.4f",
#         help="Enter latitude between -90 and 90"
#     )
# with col2:
#     lon = st.number_input(
#         "Longitude", 
#         min_value=-180.0, 
#         max_value=180.0, 
#         value=float(default_lon), 
#         step=0.0001, 
#         format="%.4f",
#         help="Enter longitude between -180 and 180"
#     )
# with col3:
#     st.write("")  # Spacer
#     st.write("")  # Align button
#     predict_button = st.button("🔍 Assess Risk", type="primary", use_container_width=True)

# # --- Prediction Logic ---
# if predict_button:
#     if not api_healthy:
#         st.error("❌ Cannot make prediction: API server is not running.")
#         st.info("Please start the backend server first: `python app.py`")
#     else:
#         # Show progress
#         progress_text = st.empty()
#         progress_bar = st.progress(0)
        
#         progress_text.text("🌐 Connecting to server...")
#         progress_bar.progress(20)
#         time.sleep(0.5)
        
#         progress_text.text("🏔️ Fetching elevation data...")
#         progress_bar.progress(40)
        
#         progress_text.text("🌤️ Retrieving weather data (this may take a moment)...")
#         progress_bar.progress(60)
        
#         # Make prediction
#         success, result = get_prediction(lat, lon)
        
#         progress_text.text("🧠 Running AI model...")
#         progress_bar.progress(80)
#         time.sleep(0.5)
        
#         if success:
#             progress_text.text("✅ Analysis complete!")
#             progress_bar.progress(100)
#             time.sleep(0.5)
            
#             # Clear progress indicators
#             progress_text.empty()
#             progress_bar.empty()
            
#             # Store result
#             st.session_state.prediction_result = result
#             st.session_state.last_coords = (lat, lon)
            
#             # Add to history
#             result_with_coords = result.copy()
#             result_with_coords['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             result_with_coords['lat'] = lat
#             result_with_coords['lon'] = lon
#             st.session_state.prediction_history.append(result_with_coords)
            
#         else:
#             progress_text.empty()
#             progress_bar.empty()
#             st.error(f"❌ Prediction failed: {result}")
#             if "weather data" in str(result).lower():
#                 st.info("💡 The weather API might be experiencing issues. Please try again in a few moments.")
#             st.session_state.prediction_result = None

# # --- Results Display ---
# if st.session_state.prediction_result:
#     result = st.session_state.prediction_result
    
#     st.header("📊 Risk Assessment Results")
    
#     # Extract data
#     risk_level = result.get('risk_level', 'Unknown')
#     probability = result.get('probability', 0)
#     probability_percent = result.get('probability_percent', '0%')
#     threshold = result.get('threshold', 0.44)
    
#     # Risk Alert Box
#     if "Very High" in risk_level:
#         st.markdown(f"""
#         <div class="risk-very-high">
#             <h2>🚨 CRITICAL ALERT: {risk_level}</h2>
#             <p>Immediate action recommended. High probability of landslide.</p>
#         </div>
#         """, unsafe_allow_html=True)
#     elif "High" in risk_level:
#         st.markdown(f"""
#         <div class="risk-high">
#             <h2>⚠️ WARNING: {risk_level}</h2>
#             <p>Elevated risk detected. Monitor conditions closely.</p>
#         </div>
#         """, unsafe_allow_html=True)
#     elif "Moderate" in risk_level:
#         st.markdown(f"""
#         <div class="risk-moderate">
#             <h2>⚡ CAUTION: {risk_level}</h2>
#             <p>Moderate risk present. Stay informed of weather changes.</p>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#         <div class="risk-low">
#             <h2>✅ STATUS: {risk_level}</h2>
#             <p>Conditions are relatively stable.</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Metrics
#     location_data = result.get('location', {})
#     weather_summary = result.get('weather_summary', {})
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric(
#             "🎯 Risk Probability", 
#             probability_percent,
#             delta=f"{(probability - threshold)*100:.1f}% vs threshold",
#             delta_color="inverse" if probability < threshold else "normal"
#         )
    
#     with col2:
#         st.metric("🏔️ Elevation", location_data.get('elevation', 'N/A'))
    
#     with col3:
#         st.metric("📐 Slope", location_data.get('slope', 'N/A'))
    
#     with col4:
#         precip_value = weather_summary.get('precipitation_sum_last_10_days', 'N/A')
#         st.metric("🌧️ 10-Day Rainfall", format_precipitation(precip_value))
    
#     st.markdown("---")
    
#     # Detailed Analysis
#     col1, col2 = st.columns([3, 2])
    
#     with col1:
#         st.subheader("📈 Risk Probability Gauge")
#         gauge_fig = create_risk_gauge(probability, risk_level)
#         st.plotly_chart(gauge_fig, use_container_width=True)
        
#         # Weather Summary
#         st.subheader("🌤️ Weather Analysis")
#         weather_col1, weather_col2, weather_col3 = st.columns(3)
        
#         with weather_col1:
#             st.metric(
#                 "3-Day Rainfall",
#                 format_precipitation(weather_summary.get('precipitation_sum_last_3_days', 'N/A'))
#             )
        
#         with weather_col2:
#             st.metric(
#                 "7-Day Rainfall",
#                 format_precipitation(weather_summary.get('precipitation_sum_last_7_days', 'N/A'))
#             )
        
#         with weather_col3:
#             st.metric(
#                 "Total Rain",
#                 format_precipitation(weather_summary.get('rain_sum_total', 'N/A'))
#             )
    
#     with col2:
#         st.subheader("🌍 Location Details")
        
#         # Location info box
#         st.info(f"""
#         **Coordinates:**  
#         📍 Latitude: {lat:.4f}  
#         📍 Longitude: {lon:.4f}
        
#         **Topography:**  
#         🏔️ Elevation: {location_data.get('elevation', 'N/A')}  
#         📐 Slope: {location_data.get('slope', 'N/A')}  
#         🧭 Aspect: {location_data.get('aspect', 'N/A')}
#         """)
        
#         # Risk interpretation
#         st.subheader("📋 Risk Interpretation")
#         if probability >= 0.7:
#             st.error("**Immediate Action Required**")
#             recommendations = [
#                 "Monitor official warnings",
#                 "Prepare emergency kit",
#                 "Plan evacuation routes",
#                 "Stay alert to changes"
#             ]
#         elif probability >= 0.5:
#             st.warning("**High Alert Status**")
#             recommendations = [
#                 "Increase vigilance",
#                 "Monitor weather updates",
#                 "Check emergency supplies",
#                 "Review safety plans"
#             ]
#         elif probability >= 0.3:
#             st.warning("**Moderate Caution**")
#             recommendations = [
#                 "Stay informed",
#                 "Monitor conditions",
#                 "Be prepared",
#                 "Know evacuation routes"
#             ]
#         else:
#             st.success("**Normal Conditions**")
#             recommendations = [
#                 "Maintain awareness",
#                 "Regular monitoring",
#                 "Keep emergency kit ready",
#                 "Stay informed"
#             ]
        
#         for rec in recommendations:
#             st.write(f"• {rec}")
    
#     # Map
#     st.subheader("🗺️ Location Map")
#     map_data = pd.DataFrame({
#         'lat': [lat],
#         'lon': [lon],
#         'risk': [risk_level],
#         'probability': [probability]
#     })
    
#     st.map(map_data, zoom=10, use_container_width=True)
    
#     # Technical Details
#     with st.expander("🔬 View Technical Details"):
#         st.json(result)

# # --- Batch Prediction Section ---
# st.markdown("---")
# st.header("📊 Batch Analysis")

# tab1, tab2 = st.tabs(["📤 Upload CSV", "📜 Prediction History"])

# with tab1:
#     st.markdown("""
#     Upload a CSV file with locations for batch risk assessment.
#     The file must contain `lat` and `lon` columns.
#     """)
    
#     # Sample CSV download
#     sample_df = pd.DataFrame({
#         'lat': [31.1048, 27.0360, 11.4064],
#         'lon': [77.1734, 88.2627, 76.6932],
#         'location_name': ['Shimla', 'Darjeeling', 'Ooty']
#     })
    
#     csv_sample = sample_df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         "📥 Download Sample CSV",
#         csv_sample,
#         "sample_locations.csv",
#         "text/csv",
#         help="Download a sample CSV file with the correct format"
#     )
    
#     uploaded_file = st.file_uploader(
#         "Choose a CSV file (max 50 locations)",
#         type="csv",
#         help="Upload a CSV with lat and lon columns"
#     )
    
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
            
#             # Validate columns
#             if 'lat' not in df.columns or 'lon' not in df.columns:
#                 st.error("❌ CSV must contain 'lat' and 'lon' columns")
#             else:
#                 # Limit to 50 locations
#                 if len(df) > 50:
#                     st.warning(f"⚠️ File has {len(df)} locations. Processing first 50 only.")
#                     df = df.head(50)
                
#                 st.success(f"✅ Loaded {len(df)} locations")
#                 st.dataframe(df.head(), use_container_width=True)
                
#                 if st.button("🚀 Run Batch Analysis", type="primary"):
#                     if not api_healthy:
#                         st.error("❌ API server not available")
#                     else:
#                         # Prepare locations
#                         locations = df[['lat', 'lon']].to_dict('records')
                        
#                         # Progress tracking
#                         progress_bar = st.progress(0)
#                         status_text = st.empty()
                        
#                         status_text.text(f"Processing {len(locations)} locations...")
                        
#                         try:
#                             # Make batch request
#                             response = requests.post(
#                                 f"{API_BASE_URL}/batch_predict",
#                                 json={"locations": locations},
#                                 timeout=180
#                             )
                            
#                             if response.status_code == 200:
#                                 batch_results = response.json()['results']
                                
#                                 # Process results
#                                 results_df = pd.DataFrame(batch_results)
                                
#                                 # Add risk levels
#                                 if 'probability' in results_df.columns:
#                                     results_df['risk_level'] = results_df['probability'].apply(
#                                         lambda p: (
#                                             "Very High Risk" if p >= 0.7 else
#                                             "High Risk" if p >= 0.5 else
#                                             "Moderate Risk" if p >= 0.3 else
#                                             "Low Risk"
#                                         )
#                                     )
                                
#                                 progress_bar.progress(100)
#                                 status_text.text("✅ Batch analysis complete!")
                                
#                                 # Display results
#                                 st.subheader("📊 Batch Results")
                                
#                                 # Summary metrics
#                                 if 'risk_level' in results_df.columns:
#                                     col1, col2, col3, col4 = st.columns(4)
                                    
#                                     risk_counts = results_df['risk_level'].value_counts()
                                    
#                                     with col1:
#                                         st.metric("Very High Risk", risk_counts.get("Very High Risk", 0))
#                                     with col2:
#                                         st.metric("High Risk", risk_counts.get("High Risk", 0))
#                                     with col3:
#                                         st.metric("Moderate Risk", risk_counts.get("Moderate Risk", 0))
#                                     with col4:
#                                         st.metric("Low Risk", risk_counts.get("Low Risk", 0))
                                    
#                                     # Visualization
#                                     fig = px.pie(
#                                         values=risk_counts.values,
#                                         names=risk_counts.index,
#                                         title="Risk Distribution",
#                                         color=risk_counts.index,
#                                         color_discrete_map={
#                                             'Very High Risk': '#d32f2f',
#                                             'High Risk': '#f57c00',
#                                             'Moderate Risk': '#fbc02d',
#                                             'Low Risk': '#388e3c'
#                                         }
#                                     )
#                                     st.plotly_chart(fig, use_container_width=True)
                                
#                                 # Results table
#                                 st.subheader("📋 Detailed Results")
#                                 st.dataframe(results_df, use_container_width=True)
                                
#                                 # Download results
#                                 csv_results = results_df.to_csv(index=False).encode('utf-8')
#                                 st.download_button(
#                                     "📥 Download Results CSV",
#                                     csv_results,
#                                     f"landslide_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                     "text/csv"
#                                 )
                                
#                             else:
#                                 st.error(f"Batch prediction failed: {response.text}")
                                
#                         except requests.exceptions.Timeout:
#                             st.error("Request timed out. Try with fewer locations.")
#                         except Exception as e:
#                             st.error(f"Error: {str(e)}")
#                         finally:
#                             progress_bar.empty()
#                             status_text.empty()
                            
#         except Exception as e:
#             st.error(f"Error reading CSV: {str(e)}")

# with tab2:
#     st.subheader("📜 Recent Predictions")
    
#     if st.session_state.prediction_history:
#         history_df = pd.DataFrame(st.session_state.prediction_history)
        
#         # Format display
#         display_df = history_df[['timestamp', 'lat', 'lon', 'risk_level', 'probability']].copy()
#         display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.2%}")
        
#         st.dataframe(display_df, use_container_width=True)
        
#         # Clear history button
#         if st.button("🗑️ Clear History"):
#             st.session_state.prediction_history = []
#             st.rerun()
#     else:
#         st.info("No predictions yet. Make a prediction to see history.")

# # --- Footer ---
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center; color: #666;'>
#     <p>⛰️ Landslide Early Warning System v1.0 | Powered by BiLSTM Neural Networks</p>
#     <p>Data sources: Open-Meteo Archive API, Open Topo Data API</p>
# </div>
# """, unsafe_allow_html=True)


