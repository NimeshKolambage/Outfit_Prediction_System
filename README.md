# WearMe - Virtual Try-On System 👕

An AI-powered virtual try-on platform that predicts user body measurements and filters clothing recommendations in real-time. Combines computer vision, gender detection, and smart filtering to create a personalized shopping experience.

## 🌟 Features

### Core Functionality
- **Real-Time Gender Detection** - Identifies user gender from live video stream using ML
- **Body Measurement Prediction** - Predicts shirt and pants sizes from user body landmarks
- **Virtual Try-On Overlay** - Real-time clothing overlay on user video feed
- **Smart Product Filtering** - Automatically filters products by predicted sizes
- **Live Video Streaming** - MJPEG stream from Flask backend to web frontend
- **Product Database Integration** - MySQL database with product images and sizes

### Technical Features
- **MediaPipe Pose Detection** - Accurate body landmark detection
- **OpenCV Processing** - Real-time video frame processing
- **Flask REST API** - Backend with multiple endpoints for data delivery
- **Responsive Web UI** - Modern frontend with Tailwind-inspired design
- **Image Serving** - Secure endpoint for product image delivery
- **CORS Enabled** - Cross-origin resource sharing for frontend access

## 📁 Project Structure

```
Outfit_Prediction_System/
├── frontend/
│   ├── index.html              # Web UI with video streaming & product grid
│   └── assets/
│       └── wearme.png          # Logo
├── virtual-tryon/
│   ├── server.py               # Flask backend with REST API
│   ├── app.py                  # Standalone camera app with OpenCV
│   ├── tryon_engine.py         # Core virtual try-on logic
│   ├── gender_detector.py      # Gender classification model
│   ├── body_measurements.py    # Body size prediction logic
│   ├── mediapipe_compat.py     # MediaPipe compatibility layer
│   ├── train_gender_model.py   # Gender model training script
│   ├── test_body_measurements.py # Unit tests
│   ├── requirements_server.txt # Python dependencies
│   └── shirts/                 # Clothing images for demo
├── README.md                   # This file
└── venv/                       # Python virtual environment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MySQL Server (optional, for product database)
- Webcam
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone repository & navigate to project**
```bash
cd "C:\ML module\Project\Outfit_Prediction_System"
```

2. **Create virtual environment**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac
```

3. **Install dependencies**
```bash
cd virtual-tryon
pip install -r requirements_server.txt
```

4. **Start Flask server**
```bash
python server.py
```
Server runs at: `http://localhost:5000`

5. **Open web browser**
Navigate to `http://localhost:5000` and start trying on clothes!

## 🎯 Usage

### Web Interface Flow
1. Click **"Try On Now"** button on landing page
2. Select clothing type (T-Shirt, Shirt Long/Short, Pants, etc.)
3. Click **"Continue"** to access camera
4. Stand 6-7 feet from camera for accurate measurements
5. Wait for **5-second countdown** and size detection
6. Press **"View Products"** to see filtered clothing
7. Use **"Prev/Next Shirt"** to browse items
8. Click **"Try On Now"** to select a product

### Database (Optional)
The system includes MySQL endpoints for product management:
- `GET /api/products` - Fetch all products
- `GET /api/products/<id>` - Get specific product
- `GET /api/products/category/<category>` - Filter by category
- `GET /api/image/<path>` - Serve product images

## 🤖 How It Works

### 1. Gender Detection
- Live video stream from webcam
- CNN/ML model analyzes facial features
- Returns: "Male", "Female", or "Detecting..."
- Displayed in real-time badge

### 2. Size Prediction
- **MediaPipe Pose Detection** extracts 33 body landmarks per frame
- **Body Measurements** module calculates dimensions:
  - Shoulder width
  - Chest circumference
  - Torso length
  - Leg length & width
- **ML Model** maps measurements → Shirt & Pants sizes (S/M/L/XL)
- **5-Second Countdown** ensures stable pose before locking size

### 3. Product Filtering
- When sizes are locked, frontend calls `applyPredictedFilter()`
- Filters grid to show only matching products:
  - Shirt category: Show only matching shirt size
  - Pants category: Show only matching pants size
- User can click **"Clear Filter"** to see all products
- Filter updates dynamically as new sizes predicted

### 4. Virtual Try-On
- **TryOnEngine** loads .png clothing images from `shirts/` folder
- Overlays clothing on detected body landmarks in real-time
- Uses OpenCV transformation for perspective matching
- Displays countdown timer and distance feedback

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI page |
| GET | `/tryon_feed` | MJPEG video stream (try-on overlay) |
| GET | `/api/gender` | Current detected gender |
| GET | `/api/status` | Server status |
| GET | `/api/tryon/status` | Current try-on status (size, item, state) |
| POST | `/api/tryon/next` | Next clothing item |
| POST | `/api/tryon/prev` | Previous clothing item |
| POST | `/api/tryon/reset` | Reset for next user |
| GET | `/api/products` | All products from DB |
| GET | `/api/products/<id>` | Specific product |
| GET | `/api/products/category/<cat>` | Products by category |
| GET | `/api/image/<path>` | Serve product image |

## 📱 Frontend Features

### UI Components
- **Hero Section** - Landing page with YouTube video
- **Brand Showcase** - Featured clothing brands
- **Camera Section** - Live video feed with overlays
  - Gender badge (top-left)
  - Size info badge (bottom-right)
  - Live indicator
- **Product Grid** - 4-column responsive card layout
  - Product images from database
  - Category, size, price info
  - Try-on buttons
- **Filter Controls** - "Showing Shirt M | Pants L" label

### Controls
- **Prev/Next Shirt** - Navigate clothing items
- **Change Clothing** - Reopen clothing type selector
- **Next User** - Reset measurements for new person
- **View Products** - Load & filter product grid
- **Back to Home** - Return to landing page

## 🔧 Configuration

### Key Settings in `server.py`
```python
CAM_INDEX = 0                  # Webcam index (0=default)
FRAME_WIDTH = 640             # Video frame width
FRAME_HEIGHT = 480            # Video frame height
```

### Database Configuration (Optional)
```python
host='localhost'
user='root'
password=''                    # Set your password
database='outfit_prediction'
```

## 📈 Model Details

### Gender Detection Model
- **Input**: Video frames (RGB)
- **Architecture**: CNN-based classifier
- **Output**: Probability scores for Male/Female
- **Training**: Run `python train_gender_model.py`

### Size Prediction Model
- **Input**: 33 MediaPipe body landmarks + frame dimensions
- **Features**: Calculated measurements (shoulder, chest, length)
- **Output**: Shirt size (S/M/L/XL) & Pants size (S/M/L/XL)
- **Algorithm**: Classification model trained on boot data

## 🧪 Testing

Run unit tests for body measurements:
```bash
python test_body_measurements.py
```

## ⚙️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not found | Check camera permissions, try CAM_INDEX=1 in server.py |
| "Server not running" error | Ensure Flask app is started: `python server.py` |
| Images not displaying | Verify image paths in DB, check `/api/image/<path>` endpoint |
| Gender always "Detecting..." | Train model: `python train_gender_model.py` |
| Size not locking | Stand 6-7 feet from camera, maintain pose for countdown |
| Products not filtering | Check data-category and data-size attributes on cards |

## 📦 Dependencies

### Main
- `flask` - Web framework
- `flask-cors` - Cross-origin requests
- `opencv-python` - Video processing
- `mediapipe` - Pose detection
- `mysql-connector-python` - Database
- `numpy` - Array operations

### Optional (for training)
- `tensorflow` / `torch` - Deep learning
- `scikit-learn` - ML utilities

Full list: `virtual-tryon/requirements_server.txt`

## 🎓 ML Pipeline

```
Webcam → OpenCV → MediaPipe Pose → Measurements → Size Classifier → Product Filter → Display
         ↓
      Gender CNN → Gender Badge
```

## 🔐 Security Features

- Directory traversal protection in `/api/image/<path>` endpoint
- CORS enabled for frontend access
- Input validation for database queries
- Secure file serving with MIME type detection

## 🚀 Future Enhancements

- [ ] Support for more clothing types (jackets, dresses, shoes)
- [ ] 3D body model visualization
- [ ] Multiple pose angles for better accuracy
- [ ] Product wishlist/favorites
- [ ] Order integration
- [ ] Mobile app (React Native)
- [ ] Real-time model improvements with user feedback
- [ ] Color/style recommendations
- [ ] Social sharing features

## 👥 Contributors

**Project Members:**
- ML/AI Development
- Backend Development
- Frontend Development

## 📄 License

This project is proprietary and confidential. All rights reserved.

## 📞 Support

For issues or questions about the system, please refer to the troubleshooting section above or contact the development team.

---

**Last Updated:** February 27, 2026 | **Version:** 1.0.0
