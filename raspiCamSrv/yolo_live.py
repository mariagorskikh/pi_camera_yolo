"""
YOLO11 Live Detection Blueprint
Provides a dedicated page for real-time YOLO11 object detection
"""

from flask import Blueprint, render_template, Response, g, request, flash, jsonify
from raspiCamSrv.auth import login_required
from raspiCamSrv.camera_pi import Camera
from raspiCamSrv.camCfg import CameraCfg
from raspiCamSrv import version
import logging
import time
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

bp = Blueprint("yolo_live", __name__, url_prefix="/yolo")

class YOLOLiveDetector:
    """Live YOLO11 detector for streaming"""
    
    def __init__(self):
        self.model = None
        self.model_initialized = False
        self.confidence_threshold = 0.5
        self.show_labels = True
        self.show_confidence = True
        self.detection_stats = {
            'objects_detected': 0,
            'frame_count': 0,
            'fps': 0,
            'last_detection_time': time.time()
        }
    
    def initialize_model(self, model_name="yolo11n.pt"):
        """Initialize YOLO11 model"""
        if not YOLO_AVAILABLE:
            logger.error("YOLO11 not available")
            return False
        
        try:
            self.model = YOLO(model_name)
            # Test with dummy frame
            test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            results = self.model(test_frame, verbose=False)
            self.model_initialized = True
            logger.info(f"YOLO11 model initialized: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize YOLO11: {e}")
            return False
    
    def detect_and_annotate(self, frame):
        """Detect objects and annotate frame - ALWAYS returns the frame with overlay"""
        annotated_frame = frame.copy()
        detections = []
        
        # Always add a basic overlay even if YOLO fails
        self._add_info_overlay(annotated_frame, [])
        
        if not self.model_initialized:
            if not self.initialize_model():
                # Add "initializing" message
                cv2.putText(annotated_frame, "YOLO11 Initializing...", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                return annotated_frame, []
        
        try:
            # Run detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Extract detections
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[cls]
                    
                    detection = {
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    }
                    detections.append(detection)
                    
                    # Draw bounding box
                    color = self._get_class_color(class_name)
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    if self.show_labels:
                        label = class_name
                        if self.show_confidence:
                            label += f": {conf:.2f}"
                        
                        # Calculate label size and position
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10), 
                                    (int(x1) + label_size[0], int(y1)), color, -1)
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update stats
            self.detection_stats['objects_detected'] = len(detections)
            self.detection_stats['frame_count'] += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.detection_stats['last_detection_time'] > 1.0:
                self.detection_stats['fps'] = self.detection_stats['frame_count'] / (current_time - self.detection_stats['last_detection_time'])
                self.detection_stats['frame_count'] = 0
                self.detection_stats['last_detection_time'] = current_time
            
            # Always add info overlay (replaces the basic one added earlier)
            self._add_info_overlay(annotated_frame, detections)
            
            return annotated_frame, detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            # Even on error, return the frame with error message
            cv2.putText(annotated_frame, f"YOLO11 Error: {str(e)[:50]}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return annotated_frame, []
    
    def _get_class_color(self, class_name):
        """Get consistent color for object class"""
        colors = {
            'person': (255, 0, 0),      # Red
            'car': (0, 255, 0),         # Green
            'truck': (0, 255, 0),       # Green
            'bus': (0, 255, 255),       # Cyan
            'bicycle': (255, 255, 0),   # Yellow
            'motorcycle': (255, 255, 0), # Yellow
            'cat': (255, 0, 255),       # Magenta
            'dog': (255, 128, 0),       # Orange
            'bird': (128, 255, 0),      # Light Green
            'bottle': (0, 128, 255),    # Blue
            'cup': (128, 0, 255),       # Purple
            'laptop': (0, 255, 128),    # Teal
            'cell phone': (255, 128, 128), # Pink
        }
        return colors.get(class_name, (128, 128, 128))  # Default gray
    
    def _add_info_overlay(self, frame, detections):
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Background for info
        overlay_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text info
        info_lines = [
            f"YOLO11 Live Detection - Objects: {len(detections)}",
            f"FPS: {self.detection_stats['fps']:.1f} | Confidence: {self.confidence_threshold}",
        ]
        
        # List detected objects
        if detections:
            object_names = [f"{d['class']} ({d['confidence']:.2f})" for d in detections[:3]]  # Show first 3
            if len(detections) > 3:
                object_names.append(f"...and {len(detections)-3} more")
            info_lines.append("Detected: " + ", ".join(object_names))
        else:
            info_lines.append("No objects detected - monitoring...")
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 20 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# Global detector instance
yolo_detector = YOLOLiveDetector()

@bp.route("/")
@login_required
def index():
    """YOLO11 live detection main page"""
    g.hostname = request.host
    g.version = version
    cfg = CameraCfg()
    sc = cfg.serverConfig
    sc.curMenu = "yolo"  # Set active menu
    
    return render_template("yolo_live/index.html", sc=sc, yolo_available=YOLO_AVAILABLE)

@bp.route("/video_feed")
@login_required
def video_feed():
    """Video streaming route with YOLO11 detection"""
    return Response(generate_yolo_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    """YOLO11 detection settings"""
    global yolo_detector
    
    if request.method == "POST":
        # Update settings
        confidence = float(request.form.get("confidence", 0.5))
        show_labels = request.form.get("show_labels") is not None
        show_confidence = request.form.get("show_confidence") is not None
        
        yolo_detector.confidence_threshold = confidence
        yolo_detector.show_labels = show_labels
        yolo_detector.show_confidence = show_confidence
        
        flash("YOLO11 settings updated!")
    
    g.hostname = request.host
    g.version = version
    cfg = CameraCfg()
    sc = cfg.serverConfig
    sc.curMenu = "yolo"  # Set active menu
    
    return render_template("yolo_live/settings.html", 
                         sc=sc, 
                         detector=yolo_detector,
                         yolo_available=YOLO_AVAILABLE)

@bp.route("/stats")
@login_required
def stats():
    """Get current detection statistics"""
    return jsonify(yolo_detector.detection_stats)

def generate_yolo_frames():
    """Generate frames with YOLO11 detection - ALWAYS shows video feed"""
    camera = Camera()
    
    # Ensure live stream is started
    from raspiCamSrv.camCfg import CameraCfg
    cfg = CameraCfg()
    if not cfg.serverConfig.isLiveStream:
        camera.startLiveStream()
        time.sleep(1)  # Give camera time to start
    
    logger.info("YOLO video stream started")
    
    while True:
        try:
            # Get frame from camera using the same method as Live page
            frame_bytes = camera.get_frame()
            if frame_bytes is None:
                time.sleep(0.05)  # Shorter sleep for better responsiveness
                continue
            
            # Convert MJPEG bytes to numpy array
            if isinstance(frame_bytes, bytes):
                # Decode MJPEG frame to numpy array
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            else:
                frame = frame_bytes
            
            if frame is None:
                time.sleep(0.05)
                continue
            
            # ALWAYS run YOLO11 detection and annotation (even if no objects detected)
            annotated_frame, detections = yolo_detector.detect_and_annotate(frame)
            
            # ALWAYS encode and yield frame (never skip)
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes_out = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes_out + b'\r\n')
            else:
                logger.error("Failed to encode frame")
                time.sleep(0.05)
                   
        except Exception as e:
            logger.error(f"Error in YOLO frame generation: {e}")
            # Even on error, try to show a basic frame
            try:
                # Create a simple error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "YOLO11 Stream Error", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except:
                pass
            time.sleep(0.1)
