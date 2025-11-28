# patient_surge_predictor.py
"""
Enhanced Patient Surge Prediction System
Predicts disease surges and resource requirements based on environmental, temporal, and contextual factors
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')


# ==========================
# 1. CONFIGURATION
# ==========================

class Config:
    DATA_PATH = "patient_surge_full_dataset.csv"
    MODEL_SAVE_PATH = "surge_prediction_model.pkl"
    PREPROCESSOR_SAVE_PATH = "surge_preprocessor.pkl"
    
    # Model hyperparameters
    N_ESTIMATORS = 300
    LEARNING_RATE = 0.1
    MAX_DEPTH = 7
    RANDOM_STATE = 42
    
    # Surge detection
    SURGE_MULTIPLIER = 1.3  # 30% above baseline = surge
    
    # Traffic accident factors (cases per 100K population)
    TRAFFIC_BASE_RATE = 5  # Base accidents on weekday
    WEEKEND_MULTIPLIER = 1.8  # 80% more on weekends
    HOLIDAY_MULTIPLIER = 2.5  # 150% more on holidays
    RAIN_MULTIPLIER = 1.4  # 40% more during rain
    FOG_MULTIPLIER = 2.0  # 100% more during fog


# ==========================
# 2. ENHANCED FEATURES
# ==========================

def add_engineered_features(df):
    """Add calculated features for better predictions"""
    
    # Air Quality Index Categories
    df['AQI_Category'] = pd.cut(
        df['AQI'],
        bins=[0, 50, 100, 150, 200, 300, 500],
        labels=['Good', 'Moderate', 'Unhealthy_Sensitive', 'Unhealthy', 'Very_Unhealthy', 'Hazardous']
    )
    
    # Temperature Categories
    df['Temp_Category'] = pd.cut(
        df['Temperature'],
        bins=[-10, 10, 20, 30, 40, 50],
        labels=['Cold', 'Cool', 'Moderate', 'Hot', 'Very_Hot']
    )
    
    # Humidity Categories
    df['Humidity_Category'] = pd.cut(
        df['Humidity'],
        bins=[0, 30, 60, 80, 100],
        labels=['Low', 'Moderate', 'High', 'Very_High']
    )
    
    # Rainfall indicator
    df['Is_Rainy'] = (df['Rainfall'] > 0).astype(int)
    df['Heavy_Rain'] = (df['Rainfall'] > 50).astype(int)
    
    # Fog conditions (high humidity + cold = fog)
    df['Is_Foggy'] = ((df['Humidity'] > 85) & (df['Temperature'] < 15)).astype(int)
    
    # Weekend flag (if not already present)
    if 'Day_Type' in df.columns:
        df['Is_Weekend'] = df['Day_Type'].isin(['Saturday', 'Sunday']).astype(int)
        df['Is_Holiday'] = (df['Day_Type'] == 'Holiday').astype(int)
    
    # Air pollution severity score (combined PM index)
    df['Pollution_Score'] = (df['PM2.5'] * 0.6 + df['PM10'] * 0.4) / 100
    
    # Seasonal risk factors
    season_risk = {
        'Summer': 1.2,    # Higher vector-borne diseases
        'Monsoon': 1.5,   # Highest disease risk
        'Autumn': 1.1,    # Moderate
        'Winter': 0.9,    # Lower overall
        'Spring': 1.0     # Baseline
    }
    df['Season_Risk'] = df['Season'].map(season_risk).fillna(1.0)
    
    # Festival impact (crowd-based disease spread)
    festival_risk = {
        'None': 1.0,
        'Diwali': 1.8,    # High pollution + crowds
        'Holi': 1.5,      # Water + crowds
        'Dussehra': 1.3,  # Moderate crowds
        'Eid': 1.4,       # Gathering events
        'Christmas': 1.2,
        'New_Year': 1.3
    }
    df['Festival_Risk'] = df['Festival'].map(festival_risk).fillna(1.0)
    
    return df


def calculate_traffic_accidents(aqi, temperature, humidity, rainfall, 
                                is_weekend, is_holiday, is_foggy, city_population=1000000):
    """
    Calculate expected traffic accident cases based on conditions
    """
    base_accidents = Config.TRAFFIC_BASE_RATE * (city_population / 100000)
    
    # Apply multipliers
    if is_holiday:
        base_accidents *= Config.HOLIDAY_MULTIPLIER
    elif is_weekend:
        base_accidents *= Config.WEEKEND_MULTIPLIER
    
    if rainfall > 10:
        base_accidents *= Config.RAIN_MULTIPLIER
    
    if is_foggy:
        base_accidents *= Config.FOG_MULTIPLIER
    
    # High pollution reduces visibility slightly
    if aqi > 300:
        base_accidents *= 1.2
    
    return int(round(base_accidents))


# ==========================
# 3. DATA LOADING & PREP
# ==========================

def load_and_prepare_data(data_path):
    """Load data and engineer features"""
    print("📊 Loading dataset...")
    df = pd.read_csv(data_path)
    
    print(f"   Original shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Add engineered features
    print("🔧 Engineering features...")
    df = add_engineered_features(df)
    
    return df


# ==========================
# 4. MODEL TRAINING
# ==========================

class SurgePredictionModel:
    def __init__(self):
        self.pipeline = None
        self.median_baselines = None
        self.feature_names = None
        
        # Define feature sets
        self.NUM_FEATURES = [
            "AQI", "PM2.5", "PM10", "Temperature", "Humidity", "Rainfall",
            "Pollution_Score", "Season_Risk", "Festival_Risk",
            "Is_Rainy", "Heavy_Rain", "Is_Foggy", "Is_Weekend", "Is_Holiday"
        ]
        
        self.CAT_FEATURES = [
            "City", "Disease", "Season", "Festival", "Day_Type",
            "AQI_Category", "Temp_Category", "Humidity_Category"
        ]
        
        self.TARGET = "Case_Count"
    
    def build_pipeline(self):
        """Create preprocessing and model pipeline"""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUM_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.CAT_FEATURES),
            ],
            remainder='drop'
        )
        
        # Using Gradient Boosting for better performance
        model = GradientBoostingRegressor(
            n_estimators=Config.N_ESTIMATORS,
            learning_rate=Config.LEARNING_RATE,
            max_depth=Config.MAX_DEPTH,
            random_state=Config.RANDOM_STATE,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=4
        )
        
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        return self.pipeline
    
    def train(self, df):
        """Train the model"""
        print("\n🎯 Preparing training data...")
        
        # Ensure all required features exist
        self.FEATURES = self.NUM_FEATURES + self.CAT_FEATURES
        available_features = [f for f in self.FEATURES if f in df.columns]
        
        X = df[available_features]
        y = df[self.TARGET]
        
        print(f"   Features: {len(available_features)}")
        print(f"   Samples: {len(df)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=Config.RANDOM_STATE, stratify=df['Disease']
        )
        
        # Build and train
        print("\n🚀 Training model...")
        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        print("\n📈 Model Performance:")
        y_pred_train = self.pipeline.predict(X_train)
        y_pred_test = self.pipeline.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"   Training   - MAE: {train_mae:.2f}, R²: {train_r2:.3f}")
        print(f"   Validation - MAE: {test_mae:.2f}, R²: {test_r2:.3f}, RMSE: {test_rmse:.2f}")
        
        # Calculate median baselines per disease
        print("\n📊 Calculating disease baselines...")
        self.median_baselines = df.groupby("Disease")[self.TARGET].median().to_dict()
        
        for disease, median in self.median_baselines.items():
            print(f"   {disease}: {median:.1f} cases (median)")
        
        self.feature_names = available_features
        
        return {
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse
        }
    
    def save_model(self, model_path=Config.MODEL_SAVE_PATH):
        """Save trained model"""
        joblib.dump({
            'pipeline': self.pipeline,
            'median_baselines': self.median_baselines,
            'feature_names': self.feature_names
        }, model_path)
        print(f"\n💾 Model saved to: {model_path}")
    
    def load_model(self, model_path=Config.MODEL_SAVE_PATH):
        """Load trained model"""
        saved_data = joblib.load(model_path)
        self.pipeline = saved_data['pipeline']
        self.median_baselines = saved_data['median_baselines']
        self.feature_names = saved_data['feature_names']
        print(f"✅ Model loaded from: {model_path}")


# ==========================
# 5. RESOURCE PLANNING
# ==========================

RESOURCE_FACTORS = {
    "Influenza": {
        "bed_ratio": 0.15,
        "oxygen_per_case": 0.05,
        "ventilators_per_case": 0.01,
        "ors_kits_per_case": 0.0,
        "neb_kits_per_case": 0.05,
        "masks_per_case": 2.0,
        "ppe_kits_per_case": 0.1,
        "antivirals_per_case": 0.8,
        "staff_per_10_cases": 0.5
    },
    "Dengue": {
        "bed_ratio": 0.40,
        "oxygen_per_case": 0.10,
        "ventilators_per_case": 0.02,
        "ors_kits_per_case": 0.20,
        "neb_kits_per_case": 0.0,
        "masks_per_case": 1.0,
        "ppe_kits_per_case": 0.15,
        "platelets_units_per_case": 0.25,
        "iv_fluids_per_case": 0.60,
        "staff_per_10_cases": 0.8
    },
    "Malaria": {
        "bed_ratio": 0.30,
        "oxygen_per_case": 0.05,
        "ventilators_per_case": 0.01,
        "ors_kits_per_case": 0.10,
        "neb_kits_per_case": 0.0,
        "masks_per_case": 0.5,
        "ppe_kits_per_case": 0.1,
        "antimalarials_per_case": 1.0,
        "rdt_kits_per_case": 1.0,
        "staff_per_10_cases": 0.6
    },
    "Asthma": {
        "bed_ratio": 0.20,
        "oxygen_per_case": 0.40,
        "ventilators_per_case": 0.05,
        "ors_kits_per_case": 0.0,
        "neb_kits_per_case": 0.80,
        "masks_per_case": 3.0,
        "ppe_kits_per_case": 0.05,
        "inhalers_per_case": 1.0,
        "bronchodilators_per_case": 0.9,
        "staff_per_10_cases": 0.7
    },
    "Diarrhea": {
        "bed_ratio": 0.10,
        "oxygen_per_case": 0.02,
        "ventilators_per_case": 0.005,
        "ors_kits_per_case": 0.95,
        "neb_kits_per_case": 0.0,
        "masks_per_case": 0.5,
        "ppe_kits_per_case": 0.2,
        "zinc_tablets_per_case": 1.0,
        "iv_fluids_per_case": 0.30,
        "staff_per_10_cases": 0.4
    },
    "Typhoid": {
        "bed_ratio": 0.35,
        "oxygen_per_case": 0.08,
        "ventilators_per_case": 0.01,
        "ors_kits_per_case": 0.30,
        "neb_kits_per_case": 0.0,
        "masks_per_case": 1.0,
        "ppe_kits_per_case": 0.15,
        "antibiotics_per_case": 1.0,
        "blood_culture_kits_per_case": 0.8,
        "staff_per_10_cases": 0.7
    },
    "Traffic_Accident": {
        "bed_ratio": 0.80,
        "oxygen_per_case": 0.30,
        "ventilators_per_case": 0.15,
        "ors_kits_per_case": 0.0,
        "neb_kits_per_case": 0.0,
        "masks_per_case": 1.0,
        "ppe_kits_per_case": 1.0,
        "trauma_kits_per_case": 0.90,
        "blood_units_per_case": 0.40,
        "xray_per_case": 0.85,
        "ct_scan_per_case": 0.30,
        "staff_per_10_cases": 2.0,
        "surgeons_per_10_cases": 0.5
    }
}

DISEASE_ADVISORIES = {
    "Influenza": [
        "🩹 Promote flu vaccination for high-risk groups (elderly, children, immunocompromised)",
        "😷 Encourage mask usage in crowded indoor places and public transport",
        "🧼 Launch hand hygiene campaigns and cough etiquette education",
        "🏥 Ensure adequate stock of antivirals (Oseltamivir) in hospitals",
        "📢 Issue public advisories on flu symptoms and when to seek medical care"
    ],
    "Dengue": [
        "🦟 Conduct intensive anti-larval operations and fogging in hotspot areas",
        "💧 Advise public to eliminate stagnant water sources around homes",
        "🔬 Increase diagnostic testing availability (NS1, IgM) in clinics",
        "🩸 Ensure blood banks are prepared for platelet requirements",
        "📱 Activate community reporting for mosquito breeding sites"
    ],
    "Malaria": [
        "🛏️ Distribute insecticide-treated mosquito nets in high-risk zones",
        "🦟 Deploy indoor residual spraying programs in affected areas",
        "💊 Ensure adequate stock of antimalarials and rapid diagnostic tests",
        "🏥 Train healthcare workers on malaria case management protocols",
        "📢 Educate communities about early symptoms and prompt testing"
    ],
    "Asthma": [
        "⚠️ Issue air quality alerts to asthma patients via SMS/app",
        "😷 Distribute N95 masks to high-risk patients during pollution spikes",
        "🏠 Advise patients to stay indoors during peak AQI hours",
        "💨 Ensure hospitals stock adequate inhalers, nebulizers, and bronchodilators",
        "🚑 Prepare emergency departments for increased respiratory distress cases"
    ],
    "Diarrhea": [
        "💧 Issue boil water advisory in affected areas",
        "🔬 Test municipal water quality and chlorination levels",
        "💊 Ensure adequate ORS packets and zinc tablets at health centers",
        "🍽️ Inspect food vendors and restaurants for hygiene compliance",
        "📢 Run public awareness on handwashing before meals"
    ],
    "Typhoid": [
        "🍽️ Intensify food safety inspections at eateries and street vendors",
        "💧 Promote safe drinking water practices and water purification",
        "💉 Consider targeted vaccination in outbreak areas",
        "🔬 Ensure lab capacity for blood cultures and Widal tests",
        "🏥 Stock adequate antibiotics (fluoroquinolones, cephalosporins)"
    ],
    "Traffic_Accident": [
        "🚦 Deploy additional traffic police at accident-prone intersections",
        "⚠️ Issue fog/rain advisory for reduced speed and high-beam usage",
        "🚑 Ensure ambulances are on standby at major highways",
        "🏥 Alert trauma centers and emergency departments for surge",
        "📱 Promote use of navigation apps to avoid congested routes",
        "🚗 Run public service messages on safe driving during adverse weather"
    ]
}


# ==========================
# 6. PREDICTION ENGINE
# ==========================

class SurgePredictionEngine:
    def __init__(self, model: SurgePredictionModel):
        self.model = model
    
    def predict_surge_and_resources(
        self,
        city: str,
        aqi: float,
        pm25: float,
        pm10: float,
        temperature: float,
        humidity: float,
        rainfall: float,
        season: str,
        festival: str = "None",
        day_type: str = "Weekday",
        city_population: int = 1000000,
        diseases: list = None,
        surge_multiplier: float = Config.SURGE_MULTIPLIER
    ):
        """
        Predict disease surges and resource requirements
        """
        
        # Create base input features
        base_features = {
            'AQI': aqi,
            'PM2.5': pm25,
            'PM10': pm10,
            'Temperature': temperature,
            'Humidity': humidity,
            'Rainfall': rainfall,
            'Season': season,
            'Festival': festival,
            'Day_Type': day_type,
        }
        
        # Add engineered features
        pollution_score = (pm25 * 0.6 + pm10 * 0.4) / 100
        is_rainy = int(rainfall > 0)
        heavy_rain = int(rainfall > 50)
        is_foggy = int((humidity > 85) and (temperature < 15))
        is_weekend = int(day_type in ['Saturday', 'Sunday'])
        is_holiday = int(day_type == 'Holiday')
        
        # Map categorical risk factors
        season_risk_map = {'Summer': 1.2, 'Monsoon': 1.5, 'Autumn': 1.1, 'Winter': 0.9, 'Spring': 1.0}
        festival_risk_map = {
            'None': 1.0, 'Diwali': 1.8, 'Holi': 1.5, 'Dussehra': 1.3,
            'Eid': 1.4, 'Christmas': 1.2, 'New_Year': 1.3
        }
        
        season_risk = season_risk_map.get(season, 1.0)
        festival_risk = festival_risk_map.get(festival, 1.0)
        
        # Categorize AQI
        if aqi <= 50:
            aqi_cat = 'Good'
        elif aqi <= 100:
            aqi_cat = 'Moderate'
        elif aqi <= 150:
            aqi_cat = 'Unhealthy_Sensitive'
        elif aqi <= 200:
            aqi_cat = 'Unhealthy'
        elif aqi <= 300:
            aqi_cat = 'Very_Unhealthy'
        else:
            aqi_cat = 'Hazardous'
        
        # Categorize Temperature
        if temperature <= 10:
            temp_cat = 'Cold'
        elif temperature <= 20:
            temp_cat = 'Cool'
        elif temperature <= 30:
            temp_cat = 'Moderate'
        elif temperature <= 40:
            temp_cat = 'Hot'
        else:
            temp_cat = 'Very_Hot'
        
        # Categorize Humidity
        if humidity <= 30:
            hum_cat = 'Low'
        elif humidity <= 60:
            hum_cat = 'Moderate'
        elif humidity <= 80:
            hum_cat = 'High'
        else:
            hum_cat = 'Very_High'
        
        # Default diseases if not specified
        if diseases is None:
            diseases = list(self.model.median_baselines.keys())
        
        # Add traffic accidents
        diseases_to_predict = list(diseases) + ['Traffic_Accident']
        
        records = []
        advisory_set = set()
        
        # Resource totals
        resource_totals = {
            'Total_Beds': 0,
            'Total_Oxygen_Units': 0,
            'Total_Ventilators': 0,
            'Total_ORS_Kits': 0,
            'Total_Nebulizer_Kits': 0,
            'Total_Masks': 0,
            'Total_PPE_Kits': 0,
            'Total_Staff_Required': 0
        }
        
        for disease in diseases_to_predict:
            if disease == 'Traffic_Accident':
                # Calculate traffic accidents
                predicted_cases = calculate_traffic_accidents(
                    aqi, temperature, humidity, rainfall,
                    is_weekend, is_holiday, is_foggy, city_population
                )
                baseline = predicted_cases * 0.6  # Lower baseline for comparison
                surge_threshold = predicted_cases * 0.8
                is_surge = predicted_cases >= surge_threshold
            else:
                # Create input dataframe
                input_data = {
                    'City': city,
                    'Disease': disease,
                    **base_features,
                    'Pollution_Score': pollution_score,
                    'Season_Risk': season_risk,
                    'Festival_Risk': festival_risk,
                    'Is_Rainy': is_rainy,
                    'Heavy_Rain': heavy_rain,
                    'Is_Foggy': is_foggy,
                    'Is_Weekend': is_weekend,
                    'Is_Holiday': is_holiday,
                    'AQI_Category': aqi_cat,
                    'Temp_Category': temp_cat,
                    'Humidity_Category': hum_cat
                }
                
                row_df = pd.DataFrame([input_data])
                
                # Predict
                predicted_cases = float(self.model.pipeline.predict(row_df)[0])
                predicted_cases = max(0, predicted_cases)  # No negative cases
                
                # Baseline and surge
                baseline = self.model.median_baselines.get(disease, 1.0)
                surge_threshold = surge_multiplier * baseline
                is_surge = predicted_cases >= surge_threshold
            
            # Get resource factors
            factors = RESOURCE_FACTORS.get(disease, {})
            
            # Calculate resources
            beds = predicted_cases * factors.get('bed_ratio', 0.2)
            oxygen = predicted_cases * factors.get('oxygen_per_case', 0.1)
            ventilators = predicted_cases * factors.get('ventilators_per_case', 0.01)
            ors = predicted_cases * factors.get('ors_kits_per_case', 0)
            neb = predicted_cases * factors.get('neb_kits_per_case', 0)
            masks = predicted_cases * factors.get('masks_per_case', 1)
            ppe = predicted_cases * factors.get('ppe_kits_per_case', 0.1)
            staff = (predicted_cases / 10) * factors.get('staff_per_10_cases', 0.5)
            
            # Update totals
            resource_totals['Total_Beds'] += beds
            resource_totals['Total_Oxygen_Units'] += oxygen
            resource_totals['Total_Ventilators'] += ventilators
            resource_totals['Total_ORS_Kits'] += ors
            resource_totals['Total_Nebulizer_Kits'] += neb
            resource_totals['Total_Masks'] += masks
            resource_totals['Total_PPE_Kits'] += ppe
            resource_totals['Total_Staff_Required'] += staff
            
            # Add disease-specific resources
            disease_specific = {}
            for key, value in factors.items():
                if key not in ['bed_ratio', 'oxygen_per_case', 'ventilators_per_case', 
                               'ors_kits_per_case', 'neb_kits_per_case', 'masks_per_case',
                               'ppe_kits_per_case', 'staff_per_10_cases']:
                    disease_specific[key] = int(round(predicted_cases * value))
            
            # Advisories
            if is_surge:
                for adv in DISEASE_ADVISORIES.get(disease, []):
                    advisory_set.add(adv)
            
            # Record
            record = {
                'Disease': disease,
                'Predicted_Cases': round(predicted_cases, 1),
                'Baseline_Median': round(baseline, 1),
                'Surge_Threshold': round(surge_threshold, 1),
                'Is_Surge': '🚨 SURGE' if is_surge else '✅ Normal',
                'Surge_Flag': is_surge,
                'Beds_Needed': int(round(beds)),
                'Oxygen_Units': int(round(oxygen)),
                'Ventilators': int(round(ventilators)),
                'ORS_Kits': int(round(ors)),
                'Nebulizers': int(round(neb)),
                'Masks': int(round(masks)),
                'PPE_Kits': int(round(ppe)),
                'Staff_Required': int(round(staff))
            }
            
            # Add disease-specific resources
            record.update(disease_specific)
            
            records.append(record)
        
        # Create results dataframe
        results_df = pd.DataFrame(records).sort_values(by='Predicted_Cases', ascending=False)
        
        # Round totals
        for key in resource_totals:
            resource_totals[key] = int(round(resource_totals[key]))
        
        # Overall summary
        summary = {
            **resource_totals,
            'Advisories': sorted(list(advisory_set)),
            'Total_Surges_Detected': int(results_df['Surge_Flag'].sum()),
            'Risk_Level': 'HIGH' if results_df['Surge_Flag'].sum() >= 3 else 'MODERATE' if results_df['Surge_Flag'].sum() >= 1 else 'LOW'
        }
        
        return results_df, summary


# ==========================
# 7. MAIN EXECUTION
# ==========================

def main():
    print("=" * 70)
    print("🏥 PATIENT SURGE PREDICTION & RESOURCE PLANNING SYSTEM")
    print("=" * 70)
    
    # Load and prepare data
    df = load_and_prepare_data(Config.DATA_PATH)
    
    # Initialize and train model
    model = SurgePredictionModel()
    metrics = model.train(df)
    
    # Save model
    model.save_model()
    
    # Create prediction engine
    engine = SurgePredictionEngine(model)
    
    # Example prediction scenarios
    print("\n" + "=" * 70)
    print("📋 EXAMPLE PREDICTION SCENARIOS")
    print("=" * 70)
    
    scenarios = [
        {
            "name": "Post-Diwali Delhi (High Pollution)",
            "params": {
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
            }
        },
        {
            "name": "Weekend Monsoon Mumbai (Rain + Traffic)",
            "params": {
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
        },
        {
            "name": "Winter Fog Situation in NCR",
            "params": {
                "city": "Delhi",
                "aqi": 380,
                "pm25": 280,
                "pm10": 420,
                "temperature": 8,
                "humidity": 92,
                "rainfall": 0,
                "season": "Winter",
                "festival": "None",
                "day_type": "Weekday",
                "city_population": 2000000
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"🎯 Scenario: {scenario['name']}")
        print(f"{'='*70}")
        
        params = scenario['params']
        print(f"\n📊 Input Parameters:")
        for key, val in params.items():
            print(f"   {key}: {val}")
        
        results_df, summary = engine.predict_surge_and_resources(**params)
        
        print(f"\n{'='*70}")
        print("📈 DISEASE PREDICTIONS & RESOURCE REQUIREMENTS")
        print(f"{'='*70}")
        print(results_df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print("📦 OVERALL RESOURCE SUMMARY")
        print(f"{'='*70}")
        for key, val in summary.items():
            if key == 'Advisories':
                print(f"\n⚠️  RECOMMENDED ADVISORIES:")
                for i, adv in enumerate(val, 1):
                    print(f"   {i}. {adv}")
            elif key == 'Risk_Level':
                emoji = '🔴' if val == 'HIGH' else '🟡' if val == 'MODERATE' else '🟢'
                print(f"\n{emoji} Overall Risk Level: {val}")
            elif key == 'Total_Surges_Detected':
                print(f"🚨 Total Surges Detected: {val}")
            else:
                print(f"   {key.replace('_', ' ')}: {val}")
    
    print("\n" + "=" * 70)
    print("✅ Training Complete! Model ready for deployment.")
    print("=" * 70)


if __name__ == "__main__":
    main()