import os
from datetime import datetime
import json
import uuid
import base64
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, make_response
import keras
import numpy as np
from PIL import Image
from database import RiceClassificationDB
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize data
db = RiceClassificationDB('rice_classification.db')

# Configuration
# Get the absolute directory of this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use os.path.join for cross-platform compatibility
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'rice_classifier_20250625_210044.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Rice class names
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Load the trained model
print("Loading trained model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (224x224 for MobileNet)
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_rice_variety(image):
    """Predict the rice variety from the preprocessed image."""
    if model is None:
        return None, None
    
    try:
        # Get prediction
        predictions = model.predict(image)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get all class probabilities
        all_probabilities = {}
        for i, class_name in enumerate(CLASS_NAMES):
            all_probabilities[class_name] = float(predictions[0][i])
        
        return predicted_class, confidence, all_probabilities
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

@app.route('/')
def index():
    """If user not in session, ask for username or offer guest option."""
    if 'user_id' not in session and 'username' not in session:
        return redirect(url_for('set_user'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('set_user'))

@app.route('/set_user', methods=['GET', 'POST'])
def set_user():
    if request.method == 'POST':
        username = request.form.get('username')
        if username:
            user_id = db.create_user(username)
            session['user_id'] = user_id
            session['username'] = username
        else:
            guest_name = f"guest_{uuid.uuid4().hex[:8]}"
            user_id = db.create_user(guest_name)
            session['user_id'] = user_id
            session['username'] = guest_name
        return redirect(url_for('index'))
    return render_template('set_user.html', username=session.get('username'))

@app.route('/switch_user')
def switch_user():
    session.clear()
    return redirect(url_for('set_user'))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'user_id' not in session:
        return jsonify({'error': 'User not set'}), 400
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF'}), 400
    
    try:
        # Read and process the image
        image = Image.open(file.stream)
        original_format = image.format or 'JPEG'

        # Always convert to RGB for prediction (support RGBA, L, P, etc.)
        if image.mode not in ['RGB']:
            image_rgb = image.convert('RGB')
        else:
            image_rgb = image

        # Save a copy of the uploaded image in its original format
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        try:
            image.save(filepath, format=original_format)
        except Exception:
            # As fallback, save as JPEG
            image_rgb.save(filepath, format='JPEG')

        # Convert to base64 as JPEG for database storage (standardizes format)
        import io
        import sys
        img_buffer = io.BytesIO()
        image_rgb.save(img_buffer, format='JPEG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Preprocess image for prediction
        processed_image = preprocess_image(image)
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        # Make prediction
        predicted_class, confidence, all_probabilities = predict_rice_variety(processed_image)
        
        if predicted_class is None:
            return jsonify({'error': 'Error making prediction'}), 500
        
        user_id = session['user_id']
        # Save prediction to database
        prediction_id = db.save_prediction(
            user_id=user_id,
            image_filename=filename,
            predicted_class=predicted_class,
            confidence=confidence,
            all_predictions=all_probabilities,
            processing_time=None,  # Add if measured
            image_size=f"{image.width}x{image.height}",
            model_version='v1.0'
        )
        
        # Prepare response
        response = {
            'success': True,
            'prediction_id': prediction_id,
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'all_probabilities': {k: round(v * 100, 2) for k, v in all_probabilities.items()},
            'image_path': filename
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/history')
def history():
    """Show prediction history page (HTML UI or JSON for AJAX)."""
    # Serve JSON only if a real AJAX XHR/fetch, or if browser prefers JSON over HTML
    if (
        request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        or request.accept_mimetypes['application/json'] > request.accept_mimetypes['text/html']
    ):
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'history': []})
        predictions = db.get_user_predictions(user_id)
        history_data = []
        for pred in predictions:
            history_data.append({
                'id': pred['id'],
                'predicted_class': pred['predicted_class'],
                'confidence': round(float(pred['confidence']) * 100, 2),
                'timestamp': pred['created_at'],
                'all_probabilities': {k: round(v * 100, 2) for k, v in pred['all_predictions'].items()},
                'image_data': '',  # TODO: provide image if you wish for preview
            })
        return jsonify({'history': history_data})
    # Otherwise always return HTML UI for direct browser loads (not for fetches)
    return render_template('history.html')

# API Routes for database operations

@app.route('/clear-history', methods=['POST'])
def clear_history():
    user_id = session.get('user_id')
    if not user_id:
        # Always reply with JSON, never a redirect, for AJAX/fetch requests
        resp = make_response(jsonify({'success': False, 'error': 'Not logged in'}), 401)
        resp.headers['Content-Type'] = 'application/json'
        return resp
    try:
        db.clear_user_predictions(user_id)
        return jsonify({'success': True})
    except Exception as e:
        resp = make_response(jsonify({'success': False, 'error': str(e)}), 500)
        resp.headers['Content-Type'] = 'application/json'
        return resp
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get prediction history for current user."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'predictions': []})
        
        predictions = db.get_user_predictions(user_id)
        
        # Process predictions for JSON response
        response_data = []
        for pred in predictions:
            pred_data = {
                'id': pred[0],
                'predicted_class': pred[3],
                'confidence': round(pred[4] * 100, 2),
                'timestamp': pred[6],
                'image_filename': pred[7] if pred[7] else f'prediction_{pred[0]}.jpg'
            }
            
            # Parse all probabilities if available
            if pred[5]:
                try:
                    all_probs = json.loads(pred[5])
                    pred_data['all_probabilities'] = {k: round(v * 100, 2) for k, v in all_probs.items()}
                except:
                    pred_data['all_probabilities'] = {}
            
            response_data.append(pred_data)
        
        return jsonify({'predictions': response_data})
    
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return jsonify({'error': 'Failed to fetch predictions'}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get statistics for current user and overall system."""
    try:
        user_id = session.get('user_id')
        
        # Get user statistics
        user_stats = {}
        if user_id:
            user_predictions = db.get_user_predictions(user_id)
            user_stats = {
                'total_predictions': len(user_predictions),
                'class_distribution': {},
                'avg_confidence': 0
            }
            
            if user_predictions:
                # Calculate class distribution
                classes = [pred[3] for pred in user_predictions]
                for class_name in CLASS_NAMES:
                    user_stats['class_distribution'][class_name] = classes.count(class_name)
                
                # Calculate average confidence
                confidences = [pred[4] for pred in user_predictions if pred[4] is not None]
                if confidences:
                    user_stats['avg_confidence'] = round(sum(confidences) / len(confidences) * 100, 2)
        
        # Get overall system statistics
        all_predictions = db.get_all_predictions()
        system_stats = {
            'total_predictions': len(all_predictions),
            'total_users': len(db.get_all_users()),
            'class_distribution': {},
            'avg_confidence': 0
        }
        
        if all_predictions:
            # Calculate system class distribution
            classes = [pred[3] for pred in all_predictions]
            for class_name in CLASS_NAMES:
                system_stats['class_distribution'][class_name] = classes.count(class_name)
            
            # Calculate system average confidence
            confidences = [pred[4] for pred in all_predictions if pred[4] is not None]
            if confidences:
                system_stats['avg_confidence'] = round(sum(confidences) / len(confidences) * 100, 2)
        
        return jsonify({
            'user_stats': user_stats,
            'system_stats': system_stats
        })
    
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return jsonify({'error': 'Failed to fetch statistics'}), 500

@app.route('/api/prediction/<int:prediction_id>', methods=['GET'])
def get_prediction_detail(prediction_id):
    """Get detailed information about a specific prediction."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        prediction = db.get_prediction_by_id(prediction_id)
        if not prediction or prediction[1] != user_id:
            return jsonify({'error': 'Prediction not found'}), 404
        
        pred_data = {
            'id': prediction[0],
            'predicted_class': prediction[3],
            'confidence': round(prediction[4] * 100, 2),
            'timestamp': prediction[6],
            'image_filename': prediction[7],
            'image_data': prediction[2]  # Base64 encoded image
        }
        
        # Parse all probabilities if available
        if prediction[5]:
            try:
                all_probs = json.loads(prediction[5])
                pred_data['all_probabilities'] = {k: round(v * 100, 2) for k, v in all_probs.items()}
            except:
                pred_data['all_probabilities'] = {}
        
        return jsonify(pred_data)
    
    except Exception as e:
        print(f"Error getting prediction detail: {e}")
        return jsonify({'error': 'Failed to fetch prediction'}), 500

@app.route('/api/user/session', methods=['GET'])
def get_user_session():
    """Get current user session information."""
    try:
        if 'user_id' not in session:
            return jsonify({'authenticated': False})
        
        user_id = session['user_id']
        user_data = db.get_user_by_id(user_id)
        
        if not user_data:
            return jsonify({'authenticated': False})
        
        return jsonify({
            'authenticated': True,
            'user_id': user_id,
            'created_at': user_data[2],
            'last_active': user_data[4]
        })
    
    except Exception as e:
        print(f"Error getting user session: {e}")
        return jsonify({'error': 'Failed to fetch user session'}), 500

@app.errorhandler(401)
def unauthorized(e):
    # Respond with JSON for AJAX/fetch/XHR requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
       request.accept_mimetypes['application/json'] > request.accept_mimetypes['text/html']:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    # Otherwise, do normal redirect for browser loads
    return redirect(url_for('set_user'))

if __name__ == '__main__':
    print("Starting Rice Classification Web Application...")
    print(f"Model loaded: {'Yes' if model else 'No'}")
    print("Available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

