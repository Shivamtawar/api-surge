# 🏥 Patient Surge Prediction API - Setup & Testing Guide

## 📋 Overview

REST API for predicting disease surges and calculating hospital resource requirements based on environmental conditions, temporal patterns, and contextual factors.

---

## 🚀 Quick Start

### **Step 1: Install Dependencies**

```bash
pip install flask flask-cors pandas numpy scikit-learn joblib --break-system-packages
```

### **Step 2: Train the Model**

```bash
# First, train the ML model
python patient_surge_predictor.py
```

This will:
- Load your dataset (`patient_surge_full_dataset.csv`)
- Train the model
- Save it as `surge_prediction_model.pkl`
- Show performance metrics

### **Step 3: Start the API Server**

```bash
python surge_prediction_api.py
```

API will start at: **http://localhost:5000**

---

## 📍 API Endpoints

### **Health & Information**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation and available endpoints |
| `/health` | GET | Health check (model status) |
| `/api/diseases` | GET | List all predictable diseases |
| `/api/model/info` | GET | Model details and configuration |
| `/api/example` | GET | Example request payload |

### **Predictions**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Single scenario prediction |
| `/api/predict/batch` | POST | Batch predictions (max 10 scenarios) |

---

## 📦 Testing with Postman

### **Import Collection**

1. Open Postman
2. Click **Import** button
3. Select `Patient_Surge_Prediction_API.postman_collection.json`
4. Collection will appear in your sidebar

### **Available Test Requests**

The collection includes:

**📊 Health & Info (5 requests)**
- API Home / Documentation
- Health Check
- Get Diseases List
- Get Model Info
- Get Example Payload

**🎯 Predictions (6 requests)**
- Post-Diwali Delhi (High Pollution)
- Weekend Monsoon Mumbai (Heavy Rain)
- Winter Fog Delhi NCR
- Summer Bangalore (Good Air Quality)
- Holi Festival (Moderate Conditions)
- Specific Diseases Only

**📋 Batch Predictions (2 requests)**
- Multi-City Batch Prediction
- Weekly Forecast (7 Days)

**❌ Error Cases (3 requests)**
- Missing Required Fields
- Invalid Data Types
- Empty JSON Body

---

## 🧪 Example API Calls

### **1. Health Check**

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-11-28T10:30:00Z"
}
```

---

### **2. Single Prediction - Post-Diwali Delhi**

**Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Delhi",
    "aqi": 420,
    "pm25": 320,
    "pm10": 450,
    "temperature": 22,
    "humidity": 40,
    "rainfall": 0.0,
    "season": "Autumn",
    "festival": "Diwali",
    "day_type": "Holiday",
    "city_population": 2000000
  }'
```

**Response Structure:**
```json
{
  "success": true,
  "timestamp": "2024-11-28T10:30:00Z",
  "input_parameters": {
    "city": "Delhi",
    "aqi": 420,
    "temperature": 22,
    "season": "Autumn",
    "day_type": "Holiday"
  },
  "predictions": {
    "diseases": [
      {
        "disease": "Traffic_Accident",
        "predicted_cases": 125.0,
        "baseline_median": 75.0,
        "surge_threshold": 97.5,
        "is_surge": true,
        "surge_status": "🚨 SURGE",
        "resources": {
          "beds": 100,
          "oxygen_units": 38,
          "ventilators": 19,
          "ors_kits": 0,
          "nebulizers": 0,
          "masks": 125,
          "ppe_kits": 125,
          "staff": 25
        },
        "disease_specific_resources": {
          "trauma_kits": 113,
          "blood_units": 50,
          "xray": 106,
          "ct_scan": 38,
          "surgeons": 6
        }
      },
      {
        "disease": "Asthma",
        "predicted_cases": 89.3,
        "baseline_median": 45.2,
        "surge_threshold": 58.8,
        "is_surge": true,
        "surge_status": "🚨 SURGE",
        "resources": {
          "beds": 18,
          "oxygen_units": 36,
          "ventilators": 4,
          "ors_kits": 0,
          "nebulizers": 71,
          "masks": 268,
          "ppe_kits": 4,
          "staff": 6
        },
        "disease_specific_resources": {
          "inhalers": 89,
          "bronchodilators": 80
        }
      }
    ],
    "summary": {
      "total_surges_detected": 3,
      "risk_level": "HIGH",
      "resources_required": {
        "total_beds": 248,
        "total_oxygen_units": 112,
        "total_ventilators": 35,
        "total_ors_kits": 15,
        "total_nebulizers": 89,
        "total_masks": 567,
        "total_ppe_kits": 178,
        "total_staff": 58
      },
      "advisories": [
        "🚦 Deploy additional traffic police at accident-prone intersections",
        "⚠️ Issue fog/rain advisory for reduced speed and high-beam usage",
        "🚑 Ensure ambulances are on standby at major highways",
        "😷 Distribute N95 masks to high-risk patients during pollution spikes",
        "💨 Ensure hospitals stock adequate inhalers and nebulizers",
        "🏠 Advise patients to stay indoors during peak AQI hours"
      ]
    }
  }
}
```

---

### **3. Batch Prediction - Multi-City**

**Request:**
```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": [
      {
        "city": "Delhi",
        "aqi": 420,
        "pm25": 320,
        "pm10": 450,
        "temperature": 22,
        "humidity": 40,
        "rainfall": 0.0,
        "season": "Autumn",
        "festival": "Diwali",
        "day_type": "Holiday",
        "city_population": 2000000
      },
      {
        "city": "Mumbai",
        "aqi": 85,
        "pm25": 55,
        "pm10": 90,
        "temperature": 28,
        "humidity": 88,
        "rainfall": 65,
        "season": "Monsoon",
        "festival": "None",
        "day_type": "Saturday",
        "city_population": 2500000
      }
    ]
  }'
```

**Response Structure:**
```json
{
  "success": true,
  "timestamp": "2024-11-28T10:30:00Z",
  "total_scenarios": 2,
  "results": [
    {
      "scenario_index": 0,
      "success": true,
      "predictions": { /* Delhi results */ }
    },
    {
      "scenario_index": 1,
      "success": true,
      "predictions": { /* Mumbai results */ }
    }
  ]
}
```

---

### **4. Specific Diseases Only**

**Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Mumbai",
    "aqi": 120,
    "pm25": 75,
    "pm10": 130,
    "temperature": 30,
    "humidity": 75,
    "rainfall": 15,
    "season": "Monsoon",
    "festival": "None",
    "day_type": "Weekday",
    "city_population": 2500000,
    "diseases": ["Dengue", "Malaria", "Diarrhea"]
  }'
```

This will predict **only** the specified diseases (plus Traffic_Accident).

---

### **5. Custom Surge Threshold**

**Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Delhi",
    "aqi": 200,
    "pm25": 150,
    "pm10": 220,
    "temperature": 25,
    "humidity": 50,
    "rainfall": 0,
    "season": "Spring",
    "festival": "Holi",
    "day_type": "Holiday",
    "city_population": 2000000,
    "surge_multiplier": 1.5
  }'
```

Uses **1.5x** baseline instead of default 1.3x for stricter surge detection.

---

## 📋 Request Parameters

### **Required Parameters**

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `city` | string | City name | "Delhi" |
| `aqi` | number | Air Quality Index (0-500) | 420 |
| `pm25` | number | PM2.5 concentration (μg/m³) | 320 |
| `pm10` | number | PM10 concentration (μg/m³) | 450 |
| `temperature` | number | Temperature (°C) | 22 |
| `humidity` | number | Humidity percentage (0-100) | 40 |
| `rainfall` | number | Rainfall (mm) | 0.0 |
| `season` | string | Season name | "Autumn" |
| `festival` | string | Festival name | "Diwali" |
| `day_type` | string | Day type | "Holiday" |

### **Optional Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `city_population` | number | 1000000 | City population |
| `diseases` | array | All diseases | Specific diseases to predict |
| `surge_multiplier` | number | 1.3 | Surge threshold multiplier |

### **Valid Values**

**Season:**
- `Summer`, `Monsoon`, `Autumn`, `Winter`, `Spring`

**Festival:**
- `None`, `Diwali`, `Holi`, `Dussehra`, `Eid`, `Christmas`, `New_Year`

**Day Type:**
- `Weekday`, `Saturday`, `Sunday`, `Holiday`

---

## 🎯 Use Case Examples

### **Use Case 1: Daily Morning Forecast**

Run prediction every morning at 6 AM with weather API data:

```python
import requests
from datetime import datetime

# Get weather data from API
weather_data = get_weather_api()  # Your weather API

# Make prediction
response = requests.post('http://localhost:5000/api/predict', json={
    "city": "Delhi",
    "aqi": weather_data['aqi'],
    "pm25": weather_data['pm25'],
    "pm10": weather_data['pm10'],
    "temperature": weather_data['temp'],
    "humidity": weather_data['humidity'],
    "rainfall": weather_data['rainfall'],
    "season": get_current_season(),
    "festival": get_current_festival(),
    "day_type": get_day_type(),
    "city_population": 2000000
})

results = response.json()

# Send alerts if HIGH risk
if results['predictions']['summary']['risk_level'] == 'HIGH':
    send_alert_to_hospitals(results)
```

---

### **Use Case 2: Festival Preparedness**

Predict resource needs for upcoming Diwali:

```python
# Predict for next 3 days of Diwali
scenarios = []
for day in range(3):
    scenarios.append({
        "city": "Delhi",
        "aqi": 400 + (day * 20),  # Increasing pollution
        "pm25": 300 + (day * 15),
        "pm10": 430 + (day * 20),
        "temperature": 22,
        "humidity": 40,
        "rainfall": 0,
        "season": "Autumn",
        "festival": "Diwali",
        "day_type": "Holiday",
        "city_population": 2000000
    })

response = requests.post('http://localhost:5000/api/predict/batch', 
                        json={"scenarios": scenarios})

# Aggregate resources for 3 days
total_resources = calculate_3day_needs(response.json())
```

---

### **Use Case 3: Real-time Dashboard**

Fetch predictions every hour and update dashboard:

```javascript
// Frontend JavaScript
async function updateDashboard() {
    const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            city: 'Delhi',
            aqi: getCurrentAQI(),
            // ... other parameters
        })
    });
    
    const data = await response.json();
    
    // Update charts
    updateResourceChart(data.predictions.summary.resources_required);
    updateSurgeAlerts(data.predictions.diseases);
    displayAdvisories(data.predictions.summary.advisories);
}

// Update every hour
setInterval(updateDashboard, 3600000);
```

---

## 🔧 Configuration

Edit `surge_prediction_api.py` to customize:

```python
# Change port
app.run(host='0.0.0.0', port=8000, debug=False)

# Enable HTTPS (in production)
app.run(ssl_context='adhoc')

# Change model path
Config.MODEL_SAVE_PATH = "/path/to/your/model.pkl"
```

---

## 🐛 Troubleshooting

### **Problem: "Model not loaded" error**

**Solution:**
```bash
# Train the model first
python patient_surge_predictor.py

# Then start API
python surge_prediction_api.py
```

---

### **Problem: CORS errors in browser**

**Solution:**
Flask-CORS is already enabled. If issues persist:

```python
# In surge_prediction_api.py
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

---

### **Problem: Port 5000 already in use**

**Solution:**
```bash
# Kill existing process
lsof -ti:5000 | xargs kill -9

# Or change port in code
app.run(port=8000)
```

---

### **Problem: Predictions seem incorrect**

**Solution:**
1. Check input data ranges (AQI: 0-500, Humidity: 0-100)
2. Verify season/festival spellings match valid values
3. Check model was trained on similar data
4. View model info: `GET /api/model/info`

---

## 📊 Response Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid input data or missing fields |
| 404 | Not Found | Endpoint doesn't exist |
| 405 | Method Not Allowed | Wrong HTTP method |
| 500 | Internal Error | Server error during prediction |
| 503 | Service Unavailable | Model not loaded |

---

## 🚀 Production Deployment

### **Using Gunicorn (Recommended)**

```bash
# Install Gunicorn
pip install gunicorn --break-system-packages

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 surge_prediction_api:app
```

### **Using Docker**

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install flask flask-cors pandas numpy scikit-learn joblib gunicorn

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "surge_prediction_api:app"]
```

Build and run:
```bash
docker build -t surge-prediction-api .
docker run -p 5000:5000 surge-prediction-api
```

---

## 📈 Performance Tips

1. **Model Loading:** Model loads once at startup (not per request)
2. **Batch Predictions:** Use `/api/predict/batch` for multiple scenarios
3. **Caching:** Consider Redis for frequently requested predictions
4. **Rate Limiting:** Add Flask-Limiter for production
5. **Async:** Use async workers for high traffic

---

## 📞 Support & Documentation

- **API Docs:** GET http://localhost:5000/
- **Example Payload:** GET http://localhost:5000/api/example
- **Model Info:** GET http://localhost:5000/api/model/info

---

## ✅ Quick Test Checklist

Before going live:

- [ ] Model trained successfully (`patient_surge_predictor.py`)
- [ ] API starts without errors (`surge_prediction_api.py`)
- [ ] Health check returns "healthy" (`GET /health`)
- [ ] Test prediction with Postman (any scenario)
- [ ] Batch prediction works (2-3 scenarios)
- [ ] Error handling works (invalid data test)
- [ ] Advisories appear for surge conditions
- [ ] Resource totals are reasonable

---

**🎉 You're ready to predict patient surges and optimize healthcare resources!**

Need integration with a frontend dashboard or mobile app? Let me know!#   a p i - s u r g e  
 