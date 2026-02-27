"""
Flask Server for Virtual Try-On with Gender Detection
Serves video stream with real-time gender detection overlay
"""

from flask import Flask, Response, render_template_string, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import sys
import os
import mysql.connector
from mysql.connector import Error

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gender_detector import GenderDetector
from tryon_engine import TryOnEngine

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Serve static files from frontend directory
import os
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching during development

# Global variables
camera = None
camera_lock = threading.Lock()
gender_detector = None
current_gender = "Detecting..."
is_running = False
tryon_engine = None
current_tryon = {
    "filename": "",
    "user_size_shirt": "--",
    "user_size_pant": "--",
    "item_type": "Shirt",
    "size_locked": False,
    "distance_valid": False,
    "distance_warning": "",
    "countdown": 0,
    "state": "STANDBY",
}

# Configuration
CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def init_camera():
    """Initialize camera"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(CAM_INDEX)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            camera.set(cv2.CAP_PROP_FPS, 30)
            
            if camera.isOpened():
                print(f"[OK] Camera {CAM_INDEX} initialized")
                return True
            else:
                print(f"[ERROR] Failed to open camera {CAM_INDEX}")
                return False
    return True


def release_camera():
    """Release camera resources"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            print("[INFO] Camera released")


def generate_frames():
    """Generate video frames with try-on and gender detection overlay"""
    global current_gender, is_running, current_tryon
    
    is_running = True
    
    if not init_camera():
        is_running = False
        return
    
    while is_running:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break
                
            ret, frame = camera.read()
        
        if not ret:
            time.sleep(0.1)
            continue
        
        # Mirror the frame for selfie view
        frame = cv2.flip(frame, 1)
        
        # Try-on overlay
        if tryon_engine is not None:
            frame, info = tryon_engine.process_frame(frame)
            current_tryon = info

            label = f"{info['item_type']}: {info['filename']}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"Shirt: {info['user_size_shirt']}  |  Pant: {info['user_size_pant']}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Detect gender
        if gender_detector is not None and gender_detector.model_loaded:
            results = gender_detector.detect_gender(frame)
            frame = gender_detector.draw_results(frame, results)

            if results:
                current_gender = results[0]['gender']
            else:
                current_gender = "No face detected"
        else:
            current_gender = "Model not loaded"
            cv2.putText(
                frame,
                "Gender model not loaded",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "Run: python train_gender_model.py",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
            )
        
        # Add timestamp overlay
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (frame.shape[1] - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    is_running = False


@app.route('/')
def serve_frontend():
    """Serve the frontend HTML"""
    try:
        with open(os.path.join(frontend_path, 'index.html'), 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Virtual Try-On Server</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    background: #1a1a1a; 
                    color: white;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 40px;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <h1>✓ Virtual Try-On Server Running</h1>
            <p>Frontend not found. Please access from the correct URL or check frontend directory.</p>
            <p>API Endpoints available:</p>
            <ul style="text-align: left; display: inline-block;">
                <li>/api/products - Get all products from database</li>
                <li>/api/products/&lt;id&gt; - Get specific product</li>
                <li>/api/products/category/&lt;category&gt; - Get products by category</li>
                <li>/tryon_feed - Video streaming feed</li>
                <li>/api/gender - Get current gender detection</li>
                <li>/api/tryon/status - Get try-on status</li>
            </ul>
        </body>
        </html>
        ''')


@app.route('/api/index')
def fallback_index():
    """Simple test page"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gender Detection Server</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: #1a1a1a; 
                color: white;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
            }
            h1 { color: #3b82f6; }
            .video-container {
                border: 2px solid #3b82f6;
                border-radius: 12px;
                overflow: hidden;
                margin: 20px 0;
            }
            img { display: block; }
            .status {
                background: rgba(59, 130, 246, 0.2);
                padding: 10px 20px;
                border-radius: 8px;
                margin: 10px 0;
            }
            .info {
                color: #888;
                font-size: 14px;
                text-align: center;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>🎭 Gender Detection Server</h1>
        <p class="status">Server is running on port 5000</p>
        
        <div class="video-container">
            <img src="/tryon_feed" width="640" height="480" alt="Video Feed">
        </div>
        
        <p id="gender-status">Current Gender: Loading...</p>
        
        <div class="info">
            <p>API Endpoints:</p>
            <p><code>/tryon_feed</code> - MJPEG video stream</p>
            <p><code>/api/gender</code> - Current gender detection result</p>
            <p><code>/api/status</code> - Server status</p>
        </div>
        
        <script>
            setInterval(async () => {
                try {
                    const response = await fetch('/api/gender');
                    const data = await response.json();
                    document.getElementById('gender-status').textContent = 
                        'Current Gender: ' + data.gender;
                } catch (e) {}
            }, 500);
        </script>
    </body>
    </html>
    ''')


@app.route('/tryon_feed')
def tryon_feed():
    """Try-on video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/gender')
def get_gender():
    """API endpoint to get current detected gender"""
    return jsonify({
        'gender': current_gender,
        'model_loaded': gender_detector.model_loaded if gender_detector else False
    })


@app.route('/api/status')
def get_status():
    """API endpoint to get server status"""
    return jsonify({
        'running': is_running,
        'camera_index': CAM_INDEX,
        'model_loaded': gender_detector.model_loaded if gender_detector else False
    })


@app.route('/api/tryon/status')
def get_tryon_status():
    """API endpoint to get current try-on state"""
    return jsonify({
        'filename': current_tryon.get('filename', ''),
        'user_size_shirt': current_tryon.get('user_size_shirt', '--'),
        'user_size_pant': current_tryon.get('user_size_pant', '--'),
        'item_type': current_tryon.get('item_type', 'Shirt'),
        'size_locked': current_tryon.get('size_locked', False),
        'distance_valid': current_tryon.get('distance_valid', False),
        'distance_warning': current_tryon.get('distance_warning', ''),
        'countdown': current_tryon.get('countdown', 0),
        'state': current_tryon.get('state', 'STANDBY'),
    })


@app.route('/api/tryon/next')
def tryon_next():
    if tryon_engine is not None:
        tryon_engine.next_shirt()
    return jsonify({'status': 'ok'})


@app.route('/api/tryon/prev')
def tryon_prev():
    if tryon_engine is not None:
        tryon_engine.prev_shirt()
    return jsonify({'status': 'ok'})


@app.route('/api/tryon/reload')
def tryon_reload():
    """Reload clothing from folder (after adding/deleting files)"""
    if tryon_engine is not None:
        count = tryon_engine.reload_clothing()
        return jsonify({'status': 'ok', 'count': count, 'items': tryon_engine.get_clothing_list()})
    return jsonify({'status': 'error', 'message': 'Engine not loaded'})


@app.route('/api/tryon/reset')
def tryon_reset():
    """Reset detection state for next user"""
    global current_tryon
    if tryon_engine is not None:
        tryon_engine.reset_for_next_user()
        current_tryon = {
            "filename": current_tryon.get('filename', ''),
            "user_size_shirt": "--",
            "user_size_pant": "--",
            "item_type": current_tryon.get('item_type', 'Shirt'),
            "size_locked": False,
            "distance_valid": False,
            "distance_warning": "",
            "countdown": 5,
            "state": "STANDBY",
        }
        return jsonify({'status': 'ok', 'message': 'Reset for next user'})
    return jsonify({'status': 'error', 'message': 'Engine not loaded'})


@app.route('/api/tryon/list')
def tryon_list():
    """Get list of all available clothing items"""
    if tryon_engine is not None:
        return jsonify({'items': tryon_engine.get_clothing_list(), 'count': len(tryon_engine.shirt_cache)})
    return jsonify({'items': [], 'count': 0})


# ======================== DATABASE FUNCTIONS ========================

def get_db_connection():
    """Create a MySQL database connection"""
    try:
        # Try with no password first
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',  # No password
            database='outfit_prediction'
        )
        print("[OK] Database connected successfully")
        return connection
    except Error as err:
        # Try with other possible configurations
        try:
            connection = mysql.connector.connect(
                host='127.0.0.1',
                user='root',
                password='root',  # Try common password
                database='outfit_prediction'
            )
            print("[OK] Database connected with password")
            return connection
        except Error:
            print(f"[WARNING] Database connection failed: {err}")
            print("[INFO] MySQL Products will not be available")
            return None


@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products from database"""
    try:
        connection = get_db_connection()
        if connection is None:
            return jsonify({'error': 'Database connection failed', 'products': []}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM products ORDER BY id DESC")
        products = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return jsonify({'products': products, 'count': len(products)})
    except Error as err:
        print(f"[ERROR] Query failed: {err}")
        return jsonify({'error': str(err), 'products': []}), 500


@app.route('/api/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    """Get a specific product by ID"""
    try:
        connection = get_db_connection()
        if connection is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM products WHERE id = %s", (product_id,))
        product = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if product:
            return jsonify({'product': product})
        else:
            return jsonify({'error': 'Product not found'}), 404
    except Error as err:
        print(f"[ERROR] Query failed: {err}")
        return jsonify({'error': str(err)}), 500


@app.route('/api/products/category/<category>', methods=['GET'])
def get_products_by_category(category):
    """Get products by category"""
    try:
        connection = get_db_connection()
        if connection is None:
            return jsonify({'error': 'Database connection failed', 'products': []}), 500
        
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM products WHERE category = %s ORDER BY id DESC", (category,))
        products = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return jsonify({'products': products, 'count': len(products)})
    except Error as err:
        print(f"[ERROR] Query failed: {err}")
        return jsonify({'error': str(err), 'products': []}), 500


@app.route('/api/image/<path:filename>', methods=['GET'])
def serve_image(filename):
    """Serve image files from database paths"""
    try:
        # Construct the full path from the workspace root
        # filename will be like 'virtual-tryon/shirts/shirt1.png'
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        image_path = os.path.join(workspace_root, filename)
        
        # Security: Prevent directory traversal attacks
        image_path = os.path.abspath(image_path)
        if not image_path.startswith(os.path.abspath(workspace_root)):
            return jsonify({'error': 'Invalid path'}), 400
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found: {image_path}")
            return jsonify({'error': 'Image not found'}), 404
        
        # Determine the MIME type
        mime_type = 'image/png'
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        elif filename.lower().endswith('.gif'):
            mime_type = 'image/gif'
        elif filename.lower().endswith('.webp'):
            mime_type = 'image/webp'
        
        from flask import send_file
        return send_file(image_path, mimetype=mime_type)
    except Exception as err:
        print(f"[ERROR] Failed to serve image: {err}")
        return jsonify({'error': str(err)}), 500


@app.route('/api/start')
def start_stream():
    """Start the video stream"""
    global is_running
    if not is_running:
        is_running = True
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already running'})


@app.route('/api/stop')
def stop_stream():
    """Stop the video stream"""
    global is_running
    is_running = False
    release_camera()
    return jsonify({'status': 'stopped'})


def main():
    global gender_detector, tryon_engine
    
    print("=" * 50)
    print("Gender Detection Server")
    print("=" * 50)
    
    # Initialize gender detector
    print("\n[INFO] Loading gender detection model...")
    gender_detector = GenderDetector()

    print("\n[INFO] Loading try-on engine...")
    tryon_engine = TryOnEngine(shirt_dir="shirts")
    
    if not gender_detector.model_loaded:
        print("\n[WARNING] Gender model not found!")
        print("[INFO] To train the model, run:")
        print("       python train_gender_model.py")
        print("\n[INFO] Server will start but gender detection will be disabled.")
    
    print(f"\n[INFO] Starting server on http://localhost:5000")
    print("[INFO] Press Ctrl+C to stop\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
