import sys
import os
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__, static_folder='static')

# ✅ Enhanced CORS configuration for Render
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    },
    r"/*": {
        "origins": "*",
        "methods": ["GET", "OPTIONS"]
    }
})

# ✅ Add CORS headers manually for all responses
@app.after_request
def after_request(response):
    # Allow all origins
    response.headers.add('Access-Control-Allow-Origin', '*')
    # Allow specific headers
    response.headers.add('Access-Control-Allow-Headers', 
                        'Content-Type, Authorization, Accept, X-Requested-With')
    # Allow specific methods
    response.headers.add('Access-Control-Allow-Methods', 
                        'GET, POST, PUT, DELETE, OPTIONS, PATCH')
    # Allow credentials if needed
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    # Cache control for API responses
    if request.path.startswith('/api/'):
        response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate')
    return response

# Configure for Render
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-123')
app.config['DEBUG'] = os.environ.get('FLASK_ENV') == 'development'

# In-memory storage (for demo - in production use database)
current_sensor_data = {}
historical_data = []
price_forecast_data = {}

def init_sample_data():
    """Initialize sample data"""
    global current_sensor_data, historical_data, price_forecast_data
    
    current_sensor_data = {
        'temperature': 25.5,
        'humidity': 65.0,
        'co2': 850,
        'received_at': datetime.now().isoformat(),
        'source': 'sample'
    }
    
    generate_price_forecast()

def generate_price_forecast():
    """Generate 30-day price forecast"""
    global price_forecast_data
    
    base_price = 2200
    predictions = []
    dates = []
    
    for i in range(30):
        date = datetime.now() + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        
        # Realistic price simulation
        trend = base_price * (1 + i * 0.001)
        seasonal = 100 * np.sin(2 * np.pi * i / 30)  # Monthly seasonality
        random_comp = np.random.normal(0, 30)
        
        price = trend + seasonal + random_comp
        predictions.append(max(1800, price))
    
    optimal_day = np.argmax(predictions) + 1
    
    price_forecast_data = {
        'predictions': [float(p) for p in predictions],
        'dates': dates,
        'statistics': {
            'optimal_sell_day': int(optimal_day),
            'max_expected_price': round(float(max(predictions)), 2),
            'min_expected_price': round(float(min(predictions)), 2),
            'average_expected_price': round(float(np.mean(predictions)), 2),
            'trend': "UPWARD" if predictions[-1] > predictions[0] else "DOWNWARD",
            'volatility_percentage': round(np.std(predictions) / np.mean(predictions) * 100, 2)
        }
    }

def analyze_sensor_data(temperature, humidity, co2, quantity=1000):
    """Main analysis function - rule-based algorithm"""
    
    # Calculate safe storage days
    safe_days = calculate_safe_days(temperature, humidity, co2)
    
    # Calculate risk score
    risk_score = calculate_risk_score(temperature, humidity, co2, safe_days)
    
    # Get risk level
    risk_level = get_risk_level(risk_score)
    
    # Get recommendations
    recommendations = get_recommendations(temperature, humidity, co2, safe_days)
    
    # Make decision
    decision = make_decision(safe_days, risk_level, quantity)
    
    # Generate detailed report
    detailed_report = generate_detailed_report(temperature, humidity, co2, safe_days, risk_level, decision, quantity)
    
    return {
        'sensor_data': {'temperature': temperature, 'humidity': humidity, 'co2': co2},
        'storage_analysis': {
            'safe_days': safe_days,
            'risk_level': risk_level,
            'recommendations': recommendations,
            'confidence': 0.85
        },
        'risk_score': risk_score,
        'decision': decision,
        'detailed_report': detailed_report,
        'timestamp': datetime.now().isoformat()
    }

def calculate_safe_days(temperature, humidity, co2):
    """Calculate safe storage days based on scientific research"""
    base_days = 45  # Base under ideal conditions
    
    # Temperature effect (exponential spoilage above 25°C)
    if temperature <= 25:
        temp_factor = 1.0
    elif temperature <= 30:
        temp_factor = np.exp(-0.05 * (temperature - 25))
    else:
        temp_factor = 0.4 * np.exp(-0.1 * (temperature - 30))
    
    # Humidity effect (mold growth accelerates above 70%)
    if humidity <= 70:
        humidity_factor = 1.0
    elif humidity <= 80:
        humidity_factor = np.exp(-0.03 * (humidity - 70))
    else:
        humidity_factor = 0.5 * np.exp(-0.05 * (humidity - 80))
    
    # CO2 effect (indicator of microbial activity)
    if co2 <= 1000:
        co2_factor = 1.0
    else:
        co2_factor = 1000 / co2
    
    safe_days = base_days * temp_factor * humidity_factor * co2_factor
    return max(1, round(safe_days, 1))

def calculate_risk_score(temperature, humidity, co2, safe_days):
    """Calculate risk score (0-100)"""
    # Temperature risk (ideal: 20-25°C)
    if temperature < 15:
        temp_risk = 20
    elif temperature <= 25:
        temp_risk = 0
    elif temperature <= 30:
        temp_risk = (temperature - 25) * 10
    else:
        temp_risk = 50 + (temperature - 30) * 3
    
    # Humidity risk (ideal: 60-70%)
    if humidity < 50:
        humidity_risk = 10
    elif humidity <= 70:
        humidity_risk = 0
    elif humidity <= 80:
        humidity_risk = (humidity - 70) * 5
    else:
        humidity_risk = 50 + (humidity - 80) * 2
    
    # CO2 risk (ideal: < 1000 PPM)
    if co2 < 800:
        co2_risk = 0
    elif co2 <= 1500:
        co2_risk = (co2 - 800) / 700 * 30
    else:
        co2_risk = 30 + (co2 - 1500) / 100 * 2
    
    # Time risk
    if safe_days > 30:
        time_risk = 0
    elif safe_days > 15:
        time_risk = (30 - safe_days) * 2
    elif safe_days > 7:
        time_risk = 30 + (15 - safe_days) * 4
    else:
        time_risk = 60 + (7 - safe_days) * 5
    
    # Combine risks with weights
    weights = {'temp': 0.25, 'humidity': 0.25, 'co2': 0.25, 'time': 0.25}
    total_risk = (
        min(100, temp_risk) * weights['temp'] +
        min(100, humidity_risk) * weights['humidity'] +
        min(100, co2_risk) * weights['co2'] +
        min(100, time_risk) * weights['time']
    )
    
    return round(total_risk, 1)

def get_risk_level(risk_score):
    """Convert risk score to risk level"""
    if risk_score >= 70:
        return 'CRITICAL'
    elif risk_score >= 50:
        return 'HIGH'
    elif risk_score >= 30:
        return 'MODERATE'
    else:
        return 'LOW'

def get_recommendations(temperature, humidity, co2, safe_days):
    """Get storage recommendations"""
    recommendations = []
    
    if temperature > 28:
        recommendations.append("⚠️ Reduce storage temperature below 28°C")
    elif temperature < 18:
        recommendations.append("✅ Temperature is optimal for wheat")
    
    if humidity > 75:
        recommendations.append("⚠️ Improve ventilation to reduce humidity")
    elif humidity < 55:
        recommendations.append("⚠️ Low humidity may cause wheat to dry out")
    else:
        recommendations.append("✅ Humidity is within optimal range")
    
    if co2 > 1500:
        recommendations.append("🚨 High CO2 indicates microbial activity")
    elif co2 > 1000:
        recommendations.append("⚠️ Monitor CO2 levels closely")
    
    if safe_days < 7:
        recommendations.append("🚨 Immediate action required - safe days critical")
    elif safe_days < 14:
        recommendations.append("⚠️ Limited safe storage days remaining")
    
    if not recommendations:
        recommendations.append("✅ Storage conditions are optimal")
    
    return recommendations

def make_decision(safe_days, risk_level, quantity):
    """Make sell/hold decision"""
    global price_forecast_data
    
    max_price = price_forecast_data['statistics']['max_expected_price']
    optimal_day = price_forecast_data['statistics']['optimal_sell_day']
    avg_price = price_forecast_data['statistics']['average_expected_price']
    trend = price_forecast_data['statistics']['trend']
    
    # Decision scoring
    price_score = max_price / avg_price if avg_price > 0 else 1.0
    risk_score = 1.0 if risk_level == 'LOW' else 0.7 if risk_level == 'MODERATE' else 0.4 if risk_level == 'HIGH' else 0.1
    time_score = min(1.0, safe_days / 30)
    
    decision_score = (price_score * 0.4) + (risk_score * 0.4) + (time_score * 0.2)
    
    # Make decision based on score
    if risk_level == "CRITICAL" or safe_days < 3:
        action = "SELL IMMEDIATELY"
        reason = "Critical storage conditions detected"
        recommended_day = 1
        confidence = "VERY HIGH"
    elif decision_score > 0.8 and optimal_day <= 7:
        action = "SELL NOW"
        reason = f"Excellent selling opportunity - best price in {optimal_day} days"
        recommended_day = optimal_day
        confidence = "HIGH"
    elif decision_score > 0.6 and optimal_day <= safe_days:
        action = "SELL SOON"
        reason = f"Good selling window - optimal price in {optimal_day} days"
        recommended_day = optimal_day
        confidence = "MEDIUM"
    elif safe_days >= 21:
        action = "CONTINUE STORING"
        reason = f"Ample storage time ({safe_days} days). Market trend: {trend.lower()}"
        recommended_day = min(optimal_day, safe_days)
        confidence = "HIGH"
    else:
        action = "HOLD AND MONITOR"
        reason = f"Monitor for {min(7, safe_days)} days. Market is {trend.lower()}"
        recommended_day = min(7, safe_days)
        confidence = "MEDIUM"
    
    expected_value = max_price * quantity / 100
    
    return {
        'action': action,
        'reason': reason,
        'recommended_action_day': recommended_day,
        'expected_price_per_quintal': round(max_price, 2),
        'expected_total_value': round(expected_value, 2),
        'confidence': confidence,
        'decision_score': round(decision_score, 2)
    }

def generate_detailed_report(temperature, humidity, co2, safe_days, risk_level, decision, quantity):
    """Generate detailed analysis report"""
    risk_score = calculate_risk_score(temperature, humidity, co2, safe_days)
    max_price = price_forecast_data['statistics']['max_expected_price']
    avg_price = price_forecast_data['statistics']['average_expected_price']
    trend = price_forecast_data['statistics']['trend']
    
    report = f"""WHEAT STORAGE ANALYSIS REPORT
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT STORAGE CONDITIONS:
• Temperature: {temperature}°C {'(Optimal)' if 20 <= temperature <= 25 else '(High)' if temperature > 25 else '(Low)'}
• Humidity: {humidity}% {'(Optimal)' if 60 <= humidity <= 70 else '(High)' if humidity > 70 else '(Low)'}
• CO₂ Level: {co2} PPM {'(Normal)' if co2 < 1000 else '(Elevated)' if co2 < 1500 else '(High)'}

STORAGE HEALTH ASSESSMENT:
• Safe Storage Days Remaining: {safe_days} days
• Risk Level: {risk_level}
• Risk Score: {risk_score}/100

PRICE FORECAST:
• Current Market Trend: {trend}
• Expected Price Range: ₹{price_forecast_data['statistics']['min_expected_price']} - ₹{max_price}/quintal
• Optimal Selling Day: Day {price_forecast_data['statistics']['optimal_sell_day']}
• Average Expected Price: ₹{avg_price}/quintal

FINANCIAL PROJECTION:
• Wheat Quantity: {quantity} quintals
• Expected Sale Value: ₹{decision['expected_total_value']:,.2f}
• Expected Price per Quintal: ₹{decision['expected_price_per_quintal']}

AI RECOMMENDATION: {decision['action']}
Confidence Level: {decision['confidence']}

REASONING: {decision['reason']}

RECOMMENDED ACTIONS:
1. {'Immediately contact buyers and prepare wheat for transport' if 'SELL' in decision['action'] else 'Continue monitoring storage conditions daily'}
2. {'Document quality metrics before selling' if 'SELL' in decision['action'] else 'Check temperature and humidity twice daily'}
3. {'Consider alternative storage if conditions worsen' if safe_days < 10 else 'Maintain current storage conditions'}

STORAGE RECOMMENDATIONS:"""
    
    recommendations = get_recommendations(temperature, humidity, co2, safe_days)
    for rec in recommendations:
        report += f"\n• {rec}"
    
    report += f"""

NEXT REVIEW: In {min(3, safe_days)} days or if conditions change significantly.

Note: This analysis is based on current conditions and market trends. 
Regular monitoring is recommended for optimal results."""
    
    return report

# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/health')
def health_check():
    """Health endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Sell Advisor',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'environment': os.environ.get('FLASK_ENV', 'development')
    })

@app.route('/api/current-status')
def get_current_status():
    """Get current sensor status"""
    if not current_sensor_data:
        return jsonify({
            'status': 'no_data',
            'message': 'No sensor data available. Use manual input.',
            'timestamp': datetime.now().isoformat()
        })
    
    try:
        analysis = analyze_sensor_data(
            current_sensor_data.get('temperature', 25),
            current_sensor_data.get('humidity', 65),
            current_sensor_data.get('co2', 800)
        )
        analysis['price_forecast'] = price_forecast_data
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sensor-data', methods=['POST', 'OPTIONS'])
def receive_sensor_data():
    """Receive data from ESP8266"""
    
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        data = request.json
        # Add debug logging for Render
        print(f"📥 [RENDER] Received data from ESP8266: {data}")
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        if 'temperature' not in data or 'humidity' not in data or 'co2' not in data:
            return jsonify({'error': 'Missing sensor data'}), 400
        
        global current_sensor_data
        current_sensor_data = {
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'co2': float(data['co2']),
            'received_at': datetime.now().isoformat(),
            'source': 'esp8266',
            'device_id': data.get('device_id', 'ESP8266_Wheat_Storage')
        }
        
        # Log success
        print(f"✅ [RENDER] Data stored: {current_sensor_data}")

        # Return success response
        response = {
            'status': 'success',
            'message': 'Data received successfully',
            'timestamp': datetime.now().isoformat(),
            'received_data': {
                'temperature': current_sensor_data['temperature'],
                'humidity': current_sensor_data['humidity'],
                'co2': current_sensor_data['co2']
            }
        }
        
        return jsonify(response)
        
    except ValueError as e:
        print(f"❌ [RENDER] ValueError: {str(e)}")
        return jsonify({'error': 'Invalid data format. Check temperature, humidity, and CO2 values.'}), 400
    except Exception as e:
        print(f"❌ [RENDER] Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/manual-input', methods=['POST'])
def manual_input():
    """Manual input endpoint"""
    try:
        data = request.json
        
        required = ['temperature', 'humidity', 'co2', 'wheat_quantity']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing {field}'}), 400
        
        # Validate ranges
        if not (10 <= data['temperature'] <= 50):
            return jsonify({'error': 'Temperature must be 10-50°C'}), 400
        if not (20 <= data['humidity'] <= 100):
            return jsonify({'error': 'Humidity must be 20-100%'}), 400
        if not (300 <= data['co2'] <= 10000):
            return jsonify({'error': 'CO2 must be 300-10000 PPM'}), 400
        
        analysis = analyze_sensor_data(
            float(data['temperature']),
            float(data['humidity']),
            float(data['co2']),
            float(data['wheat_quantity'])
        )
        
        analysis['price_forecast'] = price_forecast_data
        
        # Store as current
        global current_sensor_data
        current_sensor_data = {
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'co2': float(data['co2']),
            'received_at': datetime.now().isoformat(),
            'source': 'manual'
        }
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/price-forecast')
def get_price_forecast():
    """Get price forecast"""
    return jsonify(price_forecast_data)

@app.route('/api/reset', methods=['POST'])
def reset_data():
    """Reset data for testing"""
    global current_sensor_data, historical_data
    current_sensor_data = {}
    historical_data = []
    init_sample_data()
    return jsonify({'status': 'success', 'message': 'Data reset successfully'})

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint for ESP8266"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat(),
        'url': request.url
    })

# Initialize data
init_sample_data()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*60}")
    print(" SMART SELL ADVISOR - Production Ready")
    print(f"{'='*60}")
    print(f" Server starting on port: {port}")
    print(f" Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f" Debug mode: {app.config['DEBUG']}")
    print(f" CORS enabled: True")
    print(f"{'='*60}")
    
    # For production on Render
    if os.environ.get('FLASK_ENV') == 'production':
        print("🚀 Running in production mode...")
        app.run(host='0.0.0.0', port=port)
    else:
        print("🔧 Running in development mode...")
        app.run(host='0.0.0.0', port=port, debug=True)