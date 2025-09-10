# 🎯 YOLO11 Integration Deployment Guide for Raspberry Pi 5

## 🚀 Quick Deploy Commands

### 1. SSH into your Pi 5
```bash
ssh pi@10.0.0.249
```

### 2. Complete Installation Script
```bash
# Update system
sudo apt update && sudo apt full-upgrade -y

# Install essential dependencies
sudo apt install -y python3-picamera2 --no-install-recommends
sudo apt install -y python3-opencv ffmpeg git

# Clone the enhanced repository
cd ~
mkdir -p prg
cd prg
git clone https://github.com/mariagorskikh/pi_camera_yolo.git
cd pi_camera_yolo

# Create virtual environment with system packages
python -m venv --system-site-packages .venv
source .venv/bin/activate

# Install Python dependencies
pip install "Flask>=3,<4"
pip install numpy
pip install "matplotlib<3.8"
pip install flask-jwt-extended
pip install ultralytics

# Initialize database
flask --app raspiCamSrv init-db

# Start the server
flask --app raspiCamSrv run --port 5000 --host=0.0.0.0
```

## 🎛️ Configuration Steps

### 1. Access Web Interface
Open browser and go to: `http://10.0.0.249:5000`

### 2. Initial Setup
1. **Register first user** (becomes SuperUser)
2. **Go to Settings** → Select active camera
3. **Go to Trigger** → Motion tab
4. **Select Algorithm 5: YOLO11** from dropdown
5. **Configure YOLO11 parameters**:
   - Confidence Threshold: 0.5 (default)
   - Motion Threshold: 50 pixels
   - Size Change Threshold: 0.3 (30%)
6. **Submit** configuration
7. **Start Motion Detection**

### 3. Test YOLO11
1. Go to **Trigger** → Motion tab
2. Click **"Test Motion Detection"** button
3. You should see 4 test frames:
   - Original Frame
   - YOLO11 Detections  
   - Motion Analysis
   - Tracked Objects

## 🔧 Advanced Configuration

### YOLO11 Model Options
The system will automatically download `yolo11n.pt` (nano model). For better accuracy:

```bash
# In the Pi terminal, with venv activated:
cd ~/prg/pi_camera_yolo
python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"  # Small model
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"  # Medium model
```

### Performance Optimization for Pi 5
The integration automatically:
- Converts models to NCNN format for better ARM performance
- Optimizes frame processing for Pi 5 capabilities
- Provides ~10-15 FPS with yolo11n on Pi 5

### Object Class Filtering
To detect only specific objects, modify the YOLO11 detector:

```python
# Edit: ~/prg/pi_camera_yolo/raspiCamSrv/motionAlgoYOLO11.py
# Line ~25: Change track_classes
self.track_classes = ['person', 'car', 'bicycle']  # Only detect these classes
```

## 🎥 Dual Camera Setup (Pi 5)

Your Pi 5 supports dual cameras! Configure both:

1. **Connect second camera** to the second CSI port
2. **Go to Settings** → Cameras
3. **Enable second camera**
4. **Access streams**:
   - Primary: `http://10.0.0.249:5000/video_feed`
   - Secondary: `http://10.0.0.249:5000/video_feed2`

## 🔄 Auto-Start Service (Optional)

To start raspi-cam-srv automatically on boot:

```bash
# Copy service template
cp ~/prg/pi_camera_yolo/config/raspiCamSrv.service ~/raspiCamSrv.service

# Edit for your user (replace 'pi' with your username if different)
sed -i 's/<user>/pi/g' ~/raspiCamSrv.service

# Install service
sudo cp ~/raspiCamSrv.service /etc/systemd/system/
sudo systemctl enable raspiCamSrv.service
sudo systemctl start raspiCamSrv.service

# Check status
sudo systemctl status raspiCamSrv.service
```

## 📊 Performance Expectations

| Model | Input Size | Pi 5 FPS | Accuracy | Use Case |
|-------|------------|----------|----------|----------|
| yolo11n | 640x640 | 15-20 | Good | Real-time monitoring |
| yolo11s | 640x640 | 10-15 | Better | Balanced performance |
| yolo11m | 640x640 | 5-10 | Best | High accuracy needs |

## 🔍 Troubleshooting

### YOLO11 Not Available
```bash
# Check if ultralytics is installed
pip list | grep ultralytics

# Reinstall if needed
pip install --upgrade ultralytics
```

### Low Performance
```bash
# Check system resources
htop

# Reduce model size or frame rate in settings
# Or switch to yolo11n if using larger model
```

### Camera Issues
```bash
# Test camera directly
libcamera-hello --timeout 5000

# Check camera detection
vcgencmd get_camera
```

## 🎯 YOLO11 Features

### Motion Detection Capabilities
- **Object Detection**: 80+ COCO classes (person, car, bicycle, etc.)
- **Motion Analysis**: Tracks object movement, appearance, disappearance
- **Size Change Detection**: Detects objects getting larger/smaller
- **Multi-Object Tracking**: Handles multiple objects simultaneously

### Trigger Actions
Configure actions when motion/objects detected:
- **Photo Capture**: Take photos with YOLO11 annotations
- **Video Recording**: Record videos with bounding boxes
- **Email Notifications**: Send alerts with detection details
- **GPIO Control**: Trigger lights, alarms, etc.

### API Access
Use the raspi-cam-srv API for automation:
- `GET /api/trigger/status` - Check detection status
- `POST /api/trigger/start` - Start motion detection  
- `POST /api/trigger/stop` - Stop motion detection
- `GET /api/info/detections` - Get current detections

## 🌟 What's New with YOLO11

This enhanced version adds:
- **State-of-the-art object detection** with YOLO11
- **Intelligent motion analysis** beyond simple pixel changes
- **Object classification** in motion events
- **Performance optimization** for Raspberry Pi 5
- **Real-time visualization** of detection process
- **Seamless integration** with existing raspi-cam-srv features

## 📞 Support

If you encounter issues:
1. Check the **server logs** in the web interface
2. Verify **camera connectivity** with `libcamera-hello`
3. Test **YOLO11 standalone** with: `python test_yolo11_standalone.py`
4. Check **system resources** with `htop`

---

**🎉 Enjoy your enhanced Pi 5 camera system with YOLO11 object detection!**
