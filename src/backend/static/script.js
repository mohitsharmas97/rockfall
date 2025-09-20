document.addEventListener('DOMContentLoaded', () => {
    // --- Global Variables & Constants ---
    const API_BASE_URL = "https://rockfall-production-bd29.up.railway.app/";
    let riskGaugeChart = null;
    let map = null;
    let marker = null;

    // --- DOM Element References ---
    const DOMElements = {
        apiStatus: document.getElementById('api-status'),
        statusIndicator: document.querySelector('#api-status .status-indicator'),
        statusText: document.querySelector('#api-status .status-text'),
        modelStatus: document.getElementById('model-status'),
        scalerStatus: document.getElementById('scaler-status'),
        deviceInfo: document.getElementById('device-info'),
        apiErrorInfo: document.getElementById('api-error-info'),
        quickLocations: document.getElementById('quick-locations'),
        latInput: document.getElementById('latitude'),
        lonInput: document.getElementById('longitude'),
        selectedLocationInfo: document.getElementById('selected-location-info'),
        predictionForm: document.getElementById('prediction-form'),
        predictButton: document.getElementById('predict-button'),
        progressContainer: document.getElementById('progress-container'),
        progressText: document.getElementById('progress-text'),
        progressBarInner: document.getElementById('progress-bar-inner'),
        errorMessage: document.getElementById('error-message'),
        resultsAndActions: document.getElementById('results-and-actions'),
        resultsContainer: document.getElementById('results-container'),
        riskAlertBox: document.getElementById('risk-alert-box'),
        resProbability: document.getElementById('res-probability'),
        resProbabilityDelta: document.getElementById('res-probability-delta'),
        resElevation: document.getElementById('res-elevation'),
        resSlope: document.getElementById('res-slope'),
        resRainfall: document.getElementById('res-rainfall'),
        resRain3d: document.getElementById('res-rain-3d'),
        resRain7d: document.getElementById('res-rain-7d'),
        resRainTotal: document.getElementById('res-rain-total'),
        locationDetailsBox: document.getElementById('location-details-box'),
        riskInterpretationBox: document.getElementById('risk-interpretation-box'),
        jsonDetails: document.getElementById('json-details'),
        tabs: document.querySelectorAll('.tab-button'),
        tabContents: document.querySelectorAll('.tab-content'),
        uploadCsvButton: document.getElementById('upload-csv-button'),
        csvFileInput: document.getElementById('csv-file-input'),
        downloadSampleCsv: document.getElementById('download-sample-csv'),
        batchStatus: document.getElementById('batch-status'),
        batchResultsContainer: document.getElementById('batch-results-container'),
        historyTableContainer: document.getElementById('history-table-container'),
        clearHistoryButton: document.getElementById('clear-history-button'),
        saveAlertsButton: document.getElementById('save-alerts-button'),
        smsInput: document.getElementById('sms-number'),
        emailInput: document.getElementById('email-address'),

        // --- SHAP INTEGRATION: Add new element reference ---
        explanationBox: document.getElementById('explanation-box'),
    };

    // --- Alert Configuration Logic (Unchanged) ---
    DOMElements.saveAlertsButton.addEventListener('click', async () => {
        const smsNumber = DOMElements.smsInput.value;
        const emailAddress = DOMElements.emailInput.value;
        if (!smsNumber && !emailAddress) {
            alert("Please enter a phone number or an email address.");
            return;
        }
        const originalButtonText = DOMElements.saveAlertsButton.innerHTML;
        DOMElements.saveAlertsButton.disabled = true;
        DOMElements.saveAlertsButton.innerHTML = '💾 Saving...';
        try {
            const response = await fetch(`${API_BASE_URL}/save_alerts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sms: smsNumber, email: emailAddress }),
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || 'Failed to save settings.');
            }
            DOMElements.saveAlertsButton.innerHTML = '✅ Saved!';
        } catch (error) {
            console.error("Failed to save alert settings:", error);
            alert(`Error: ${error.message}`);
            DOMElements.saveAlertsButton.innerHTML = '❌ Error';
        } finally {
            setTimeout(() => {
                DOMElements.saveAlertsButton.disabled = false;
                DOMElements.saveAlertsButton.innerHTML = originalButtonText;
            }, 3000);
        }
    });

    // --- Helper Functions (Unchanged)---
    const formatPrecipitation = (value) => {
        if (value === null || value === undefined || value === "N/A") return "N/A";
        return `${parseFloat(value).toFixed(1)} mm`;
    };
    const getRiskStyles = (level) => {
        const styles = {
            "Very High Risk": { class: 'risk-very-high', color: '#c62828' },
            "High Risk": { class: 'risk-high', color: '#e65100' },
            "Moderate Risk": { class: 'risk-moderate', color: '#fbc02d' },
            "Low Risk": { class: 'risk-low', color: '#2e7d32' },
        };
        return styles[level] || { class: '', color: '#757575' };
    };

    // --- API Communication (Unchanged)---
    const checkApiHealth = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
            if (!response.ok) throw new Error('API response not OK');
            const data = await response.json();
            DOMElements.statusIndicator.className = 'status-indicator success';
            DOMElements.statusText.textContent = 'API Connected';
            DOMElements.modelStatus.textContent = data.model_loaded ? '✅ Ready' : '❌ Error';
            DOMElements.scalerStatus.textContent = data.scaler_loaded ? '✅ Ready' : '❌ Error';
            DOMElements.deviceInfo.textContent = `🖥️ Device: ${data.device}`;
            DOMElements.deviceInfo.style.display = 'block';
            DOMElements.apiErrorInfo.style.display = 'none';
            return true;
        } catch (error) {
            DOMElements.statusIndicator.className = 'status-indicator error';
            DOMElements.statusText.textContent = 'API Disconnected';
            DOMElements.modelStatus.textContent = '...';
            DOMElements.scalerStatus.textContent = '...';
            DOMElements.deviceInfo.style.display = 'none';
            DOMElements.apiErrorInfo.style.display = 'block';
            console.error("API Health Check Failed:", error);
            return false;
        }
    };
    const getPrediction = async (lat, lon) => {
        const payload = { lat: parseFloat(lat), lon: parseFloat(lon) };
        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: AbortSignal.timeout(60000)
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || `API Error (${response.status})`);
            }
            return { success: true, data: result };
        } catch (error) {
            let errorMessage = "An unexpected error occurred.";
            if (error.name === 'TimeoutError') {
                errorMessage = "Request timed out. The weather API might be slow. Please try again.";
            } else if (error instanceof TypeError) {
                 errorMessage = "Cannot connect to the backend server. Please ensure it's running.";
            } else {
                 errorMessage = error.message;
            }
            return { success: false, error: errorMessage };
        }
    };
    
    // --- UI Update Functions ---
    const updateProgress = (percentage, text) => {
        DOMElements.progressBarInner.style.width = `${percentage}%`;
        DOMElements.progressText.textContent = text;
    };
    const displayError = (message) => {
        DOMElements.errorMessage.textContent = `❌ Prediction failed: ${message}`;
        DOMElements.errorMessage.style.display = 'block';
        DOMElements.resultsAndActions.style.display = 'none';
    };

    // --- SHAP INTEGRATION: New function to display explanations ---
    const displayExplanation = (explanation) => {
        if (!explanation || !explanation.impacts) {
            DOMElements.explanationBox.innerHTML = '<h4>Model Reasoning</h4><p>Explanation data not available.</p>';
            return;
        }

        let html = `<h4>🧠 Model Reasoning</h4><p><small>${explanation.message}</small></p><ul>`;

        // Loop through the feature impacts received from the API
        for (const [feature, impact] of Object.entries(explanation.impacts)) {
            const impactClass = impact > 0 ? 'impact-positive' : 'impact-negative';
            const icon = impact > 0 ? '🔺' : '🔻';
            html += `<li>
                <span class="feature-name">${feature}</span>
                <span class="feature-impact ${impactClass}">${icon} ${impact.toFixed(4)}</span>
            </li>`;
        }

        html += '</ul>';
        DOMElements.explanationBox.innerHTML = html;
    };
    
    const displayResults = (result) => {
        DOMElements.errorMessage.style.display = 'none';
        
        // --- SHAP INTEGRATION: Destructure the new explanation key ---
        const { risk_level, probability, probability_percent, threshold, location, weather_summary, explanation } = result;
        const lat = DOMElements.latInput.value;
        const lon = DOMElements.lonInput.value;

        // The rest of this function is mostly the same...
        const riskStyles = getRiskStyles(risk_level);
        const alertContent = {
            "Very High Risk": `<h2>🚨 CRITICAL ALERT: ${risk_level}</h2><p>Immediate action recommended. High probability of rockfall event.</p>`,
            "High Risk": `<h2>⚠️ WARNING: ${risk_level}</h2><p>Elevated risk detected. Monitor conditions closely and prepare mitigation.</p>`,
            "Moderate Risk": `<h2>⚡ CAUTION: ${risk_level}</h2><p>Moderate risk present. Increase monitoring frequency.</p>`,
            "Low Risk": `<h2>✅ STATUS: ${risk_level}</h2><p>Conditions appear stable. Continue standard monitoring.</p>`
        };
        DOMElements.riskAlertBox.innerHTML = alertContent[risk_level] || `<p>${risk_level}</p>`;
        DOMElements.riskAlertBox.className = `risk-alert ${riskStyles.class}`;

        const delta = ((probability - threshold) * 100).toFixed(1);
        DOMElements.resProbability.textContent = probability_percent;
        DOMElements.resProbabilityDelta.textContent = `${delta > 0 ? '+' : ''}${delta}% vs threshold`;
        DOMElements.resProbabilityDelta.style.color = probability < threshold ? 'green' : 'red';
        DOMElements.resElevation.textContent = `${location.elevation || 'N/A'} m`;
        DOMElements.resSlope.textContent = `${location.slope || 'N/A'}°`;
        DOMElements.resRainfall.textContent = formatPrecipitation(weather_summary.precipitation_sum_last_10_days);
        DOMElements.resRain3d.textContent = formatPrecipitation(weather_summary.precipitation_sum_last_3_days);
        DOMElements.resRain7d.textContent = formatPrecipitation(weather_summary.precipitation_sum_last_7_days);
        DOMElements.resRainTotal.textContent = formatPrecipitation(weather_summary.rain_sum_total);

        DOMElements.locationDetailsBox.innerHTML = `
            <strong>Coordinates:</strong> 📍 ${parseFloat(lat).toFixed(4)}, ${parseFloat(lon).toFixed(4)}<br>
            <strong>Topography:</strong><br>
            🏔️ Elevation: ${location.elevation || 'N/A'} m<br>
            📐 Slope: ${location.slope || 'N/A'}°<br>
            🧭 Aspect: ${location.aspect || 'N/A'}°
        `;

        let recs, title;
        if (probability >= 0.7) {
            title = '<h4><span style="color: red;">Immediate Action Plan</span></h4>';
            recs = ["Initiate Level 3 monitoring", "Restrict access to high-risk zones", "Verify blast-zone clearance", "Prepare evacuation equipment"];
        } else if (probability >= 0.5) {
            title = '<h4><span style="color: orange;">High Alert Action Plan</span></h4>';
            recs = ["Deploy rapid-response survey team", "Increase sensor polling frequency", "Review and communicate evacuation routes", "Alert on-site managers"];
        } else if (probability >= 0.3) {
            title = '<h4><span style="color: #fbc02d;">Moderate Caution Plan</span></h4>';
            recs = ["Schedule follow-up geological inspection", "Cross-reference with seismic data", "Ensure monitoring hardware is fully operational", "Note for daily briefing"];
        } else {
            title = '<h4><span style="color: green;">Standard Operating Procedures</span></h4>';
            recs = ["Continue routine monitoring", "Log prediction data", "No immediate action required"];
        }
        DOMElements.riskInterpretationBox.innerHTML = title + `<ul>${recs.map(r => `<li>${r}</li>`).join('')}</ul>`;

        // --- SHAP INTEGRATION: Call the new display function ---
        displayExplanation(explanation);

        renderGaugeChart(probability, riskStyles.color);
        updateMapMarker(lat, lon, 12);
        DOMElements.jsonDetails.textContent = JSON.stringify(result, null, 2);
        DOMElements.resultsAndActions.style.display = 'block';
    };

    const renderGaugeChart = (probability, color) => {
        const ctx = document.getElementById('riskGauge').getContext('2d');
        const value = probability * 100;
        if (riskGaugeChart) {
            riskGaugeChart.destroy();
        }
        riskGaugeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [value, 100 - value],
                    backgroundColor: [color, '#e9ecef'],
                    borderColor: ['#fff'],
                    borderWidth: 2,
                    circumference: 180,
                    rotation: 270,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                cutout: '70%',
            }
        });
    };
    
    // --- Map Logic (Unchanged)---
    const initMap = () => {
        const initialLat = DOMElements.latInput.value;
        const initialLon = DOMElements.lonInput.value;
        map = L.map('map').setView([initialLat, initialLon], 10);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
        marker = L.marker([initialLat, initialLon]).addTo(map);
        map.on('click', (e) => {
            const { lat, lng } = e.latlng;
            DOMElements.latInput.value = lat.toFixed(4);
            DOMElements.lonInput.value = lng.toFixed(4);
            updateMapMarker(lat, lng);
            DOMElements.selectedLocationInfo.textContent = `📍 Selected via map: ${lat.toFixed(4)}, ${lng.toFixed(4)}`;
            DOMElements.selectedLocationInfo.style.display = 'block';
            DOMElements.quickLocations.value = 'custom';
        });
    };
    const updateMapMarker = (lat, lon, zoomLevel = null) => {
        if (!map) {
            initMap();
        }
        const newLatLng = L.latLng(lat, lon);
        if (zoomLevel) {
            map.setView(newLatLng, zoomLevel);
        } else {
            map.panTo(newLatLng);
        }
        if (marker) {
            marker.setLatLng(newLatLng);
        } else {
            marker = L.marker(newLatLng).addTo(map);
        }
    };

    // --- Event Handlers (Unchanged)---
    DOMElements.predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const lat = DOMElements.latInput.value;
        const lon = DOMElements.lonInput.value;
        
        DOMElements.predictButton.disabled = true;
        DOMElements.predictButton.textContent = 'Assessing...';
        DOMElements.resultsAndActions.style.display = 'none';
        DOMElements.errorMessage.style.display = 'none';
        DOMElements.progressContainer.style.display = 'block';

        updateProgress(20, '🌐 Connecting to server...');
        if (!await checkApiHealth()) {
            displayError("Cannot connect to the backend server.");
            DOMElements.predictButton.disabled = false;
            DOMElements.predictButton.textContent = '🔍 Assess Risk';
            DOMElements.progressContainer.style.display = 'none';
            return;
        }

        updateProgress(40, '🌤️ Retrieving geo-spatial & weather data...');
        const result = await getPrediction(lat, lon);
        
        updateProgress(80, '🧠 Running AI prediction model...');
        
        if (result.success) {
            updateProgress(100, '✅ Analysis complete!');
            setTimeout(() => {
                displayResults(result.data);
                saveToHistory({ ...result.data, lat, lon });
                DOMElements.progressContainer.style.display = 'none';
            }, 500);
        } else {
            displayError(result.error);
            DOMElements.progressContainer.style.display = 'none';
        }

        DOMElements.predictButton.disabled = false;
        DOMElements.predictButton.textContent = '🔍 Assess Risk';
    });
    DOMElements.quickLocations.addEventListener('change', (e) => {
        const value = e.target.value;
        if (value === 'custom') {
            DOMElements.selectedLocationInfo.style.display = 'none';
            return;
        };
        const [lat, lon] = value.split(',');
        DOMElements.latInput.value = lat;
        DOMElements.lonInput.value = lon;
        updateMapMarker(lat, lon);
        const selectedText = e.target.options[e.target.selectedIndex].text;
        DOMElements.selectedLocationInfo.textContent = `📍 Selected: ${selectedText} (${parseFloat(lat).toFixed(4)}, ${parseFloat(lon).toFixed(4)})`;
        DOMElements.selectedLocationInfo.style.display = 'block';
    });

    // --- Tabs, Batch, and History Logic (Unchanged) ---
    DOMElements.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            DOMElements.tabs.forEach(t => t.classList.remove('active'));
            DOMElements.tabContents.forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.tab).classList.add('active');
            if(tab.dataset.tab === 'history') {
                renderHistoryTable();
            }
        });
    });
    DOMElements.downloadSampleCsv.addEventListener('click', (e) => {
        e.preventDefault();
        const csvContent = "lat,lon,location_name\n-33.93,115.17,Margaret River\n44.46,-110.82,Bingham Canyon\n-22.45,15.03,Rossing Mine";
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", "sample_mine_sites.csv");
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
    DOMElements.uploadCsvButton.addEventListener('click', () => DOMElements.csvFileInput.click());
    DOMElements.csvFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            DOMElements.batchStatus.textContent = `Processing file: ${file.name}...`;
            Papa.parse(file, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => processBatch(results.data),
                error: (error) => {
                    DOMElements.batchStatus.textContent = `Error parsing CSV: ${error.message}`;
                    DOMElements.batchStatus.style.color = 'red';
                }
            });
        }
    });
    const processBatch = async (locations) => {
        if (!locations || !locations[0] || !('lat' in locations[0]) || !('lon' in locations[0])) {
            DOMElements.batchStatus.textContent = "Error: CSV must contain 'lat' and 'lon' columns.";
            DOMElements.batchStatus.style.color = 'red';
            return;
        }
        if (locations.length > 50) {
            locations = locations.slice(0, 50);
            DOMElements.batchStatus.textContent = `Processing first 50 of ${locations.length} locations...`;
        } else {
            DOMElements.batchStatus.textContent = `Processing ${locations.length} locations...`;
        }
        try {
            const response = await fetch(`${API_BASE_URL}/batch_predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ locations }),
                signal: AbortSignal.timeout(180000)
            });
            if (!response.ok) throw new Error(`API Error (${response.status})`);
            const data = await response.json();
            displayBatchResults(data.results);
            DOMElements.batchStatus.textContent = '✅ Batch analysis complete!';
            DOMElements.batchStatus.style.color = 'green';
        } catch (error) {
            DOMElements.batchStatus.textContent = `Batch prediction failed: ${error.message}`;
            DOMElements.batchStatus.style.color = 'red';
        }
    };
    const displayBatchResults = (results) => {
        const table = document.createElement('table');
        const headers = ['Latitude', 'Longitude', 'Probability', 'Risk Level', 'Error'];
        table.innerHTML = `<thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>`;
        const tbody = document.createElement('tbody');
        results.forEach(res => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${res.lat || 'N/A'}</td>
                <td>${res.lon || 'N/A'}</td>
                <td>${res.probability !== undefined ? (res.probability * 100).toFixed(2) + '%' : 'N/A'}</td>
                <td>${res.risk_level || 'N/A'}</td>
                <td>${res.error || 'None'}</td>
            `;
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        DOMElements.batchResultsContainer.innerHTML = '<h3>Batch Results</h3>';
        DOMElements.batchResultsContainer.appendChild(table);
        DOMElements.batchResultsContainer.style.display = 'block';
    };
    const getHistory = () => { return JSON.parse(localStorage.getItem('predictionHistory') || '[]'); };
    const saveToHistory = (result) => {
        let history = getHistory();
        const newEntry = { timestamp: new Date().toISOString(), ...result };
        history.unshift(newEntry);
        if (history.length > 50) history = history.slice(0, 50);
        localStorage.setItem('predictionHistory', JSON.stringify(history));
    };
    const renderHistoryTable = () => {
        const history = getHistory();
        DOMElements.historyTableContainer.innerHTML = '';
        if (history.length === 0) {
            DOMElements.historyTableContainer.innerHTML = '<p>No predictions yet. Make a prediction to see history.</p>';
            DOMElements.clearHistoryButton.style.display = 'none';
            return;
        }
        const table = document.createElement('table');
        table.innerHTML = `<thead><tr><th>Timestamp</th><th>Latitude</th><th>Longitude</th><th>Risk Level</th><th>Probability</th></tr></thead>`;
        const tbody = document.createElement('tbody');
        history.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${new Date(item.timestamp).toLocaleString()}</td><td>${parseFloat(item.lat).toFixed(4)}</td><td>${parseFloat(item.lon).toFixed(4)}</td><td>${item.risk_level}</td><td>${item.probability_percent}</td>`;
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        DOMElements.historyTableContainer.appendChild(table);
        DOMElements.clearHistoryButton.style.display = 'block';
    };
    DOMElements.clearHistoryButton.addEventListener('click', () => {
        localStorage.removeItem('predictionHistory');
        renderHistoryTable();
    });

    // --- Initializations ---
    checkApiHealth();
    initMap();
    DOMElements.quickLocations.dispatchEvent(new Event('change'));
});