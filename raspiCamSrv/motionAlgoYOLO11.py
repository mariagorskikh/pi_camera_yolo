"""
YOLO11-based Motion Detection Algorithm for raspiCamSrv
Integrates YOLO11 object detection with the existing motion detection framework
"""

import cv2
import numpy as np
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    
import copy
from _thread import get_ident
import logging
import os
from datetime import datetime
from raspiCamSrv.motionAlgoIB import MotionDetectAlgoIB

logger = logging.getLogger(__name__)

class MotionDetectYOLO11(MotionDetectAlgoIB):
    """YOLO11-based object detection and motion analysis"""
    
    def __init__(self):
        super().__init__()
        
        # Algorithm reference
        self.algoReferenceTit = "Ultralytics YOLO11 Object Detection"
        self.algoReferenceURL = "https://docs.ultralytics.com/"
        self.testFrame1Title = "Original Frame"
        self.testFrame2Title = "YOLO11 Detections"
        self.testFrame3Title = "Motion Analysis"
        self.testFrame4Title = "Tracked Objects"
        
        # YOLO11 model
        self.model = None
        self.model_initialized = False
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
        self.track_classes = []  # Empty = all classes, or specify: ['person', 'car', 'dog']
        
        # Motion tracking
        self.previous_detections = []
        self.motion_threshold = 50  # pixels for significant movement
        self.size_change_threshold = 0.3  # 30% size change
        self.new_object_threshold = 0.5  # confidence for new objects
        
        # Performance settings
        self.input_size = 640
        self.max_detections = 50
        
        logger.info("YOLO11 Motion Detector initialized")
    
    def initialize_model(self, model_path="yolo11n.pt"):
        """Initialize YOLO11 model"""
        if not YOLO_AVAILABLE:
            logger.error("YOLO11 not available - install ultralytics package")
            return False
            
        try:
            logger.info(f"Loading YOLO11 model: {model_path}")
            self.model = YOLO(model_path)
            
            # Test the model with a dummy image
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            results = self.model(test_img, verbose=False)
            
            self.model_initialized = True
            logger.info(f"YOLO11 model loaded successfully: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO11 model: {e}")
            return False
    
    def detectMotion(self, frame2, frame1):
        """Detect objects and analyze motion between frames"""
        motion = False
        self.frame2 = copy.copy(frame2)
        self.frame1 = copy.copy(frame1)
        
        # Initialize model if not done
        if not self.model_initialized:
            if not self.initialize_model():
                trigger = {
                    "trigger": "YOLO11 Object Detection",
                    "triggertype": "Initialization Error",
                    "triggerparam": {"error": "Model not loaded"}
                }
                return (False, trigger)
        
        try:
            # Run YOLO11 detection on current frame
            # Resize frame for consistent processing
            h, w = frame2.shape[:2]
            if w > self.input_size or h > self.input_size:
                scale = self.input_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                processed_frame = cv2.resize(frame2, (new_w, new_h))
            else:
                processed_frame = frame2
                scale = 1.0
            
            # Run detection
            results = self.model(processed_frame, 
                               conf=self.confidence_threshold, 
                               iou=self.iou_threshold,
                               max_det=self.max_detections,
                               verbose=False)
            
            # Scale detections back to original frame size
            current_detections = self._extract_detections(results[0], scale)
            
            if self.test:
                self.testFrame1 = self._frameToStream(frame2)
                # Create annotated frame
                annotated_frame = self._draw_detections(frame2.copy(), current_detections)
                self.testFrame2 = self._frameToStream(annotated_frame)
            
            # Analyze motion between previous and current detections
            motion_data = self._analyze_motion(current_detections)
            
            if motion_data['motion_detected']:
                motion = True
                if self.test:
                    motion_frame = self._draw_motion_analysis(frame2.copy(), motion_data)
                    self.testFrame3 = self._frameToStream(motion_frame)
                    
                    # Draw tracking visualization
                    tracking_frame = self._draw_tracking(frame2.copy(), current_detections, motion_data)
                    self.testFrame4 = self._frameToStream(tracking_frame)
            
            # Update detection history
            self.previous_detections = current_detections
            
            # Set detections for potential video recording with bboxes
            self.detections = self._convert_detections_for_recording(current_detections)
            
            trigger = {
                "trigger": "YOLO11 Object Detection",
                "triggertype": "Object Motion Analysis",
                "triggerparam": {
                    "objects_detected": len(current_detections),
                    "motion_events": len(motion_data['motion_events']),
                    "confidence_thr": self.confidence_threshold,
                    "classes_detected": list(set([d['class'] for d in current_detections]))
                }
            }
            
            return (motion, trigger)
            
        except Exception as e:
            logger.error(f"Error in YOLO11 detectMotion: {e}")
            trigger = {
                "trigger": "YOLO11 Object Detection",
                "triggertype": "Detection Error",
                "triggerparam": {"error": str(e)}
            }
            return (False, trigger)
    
    def _extract_detections(self, results, scale=1.0):
        """Extract detection data from YOLO results"""
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls]
                
                # Filter by class if specified
                if self.track_classes and class_name not in self.track_classes:
                    continue
                
                # Scale coordinates back to original frame size
                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                    
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': class_name,
                    'center': [(x1+x2)/2, (y1+y2)/2],
                    'area': (x2-x1) * (y2-y1),
                    'id': f"{class_name}_{len(detections)}"  # Simple ID for tracking
                })
        return detections
    
    def _analyze_motion(self, current_detections):
        """Analyze motion between current and previous detections"""
        motion_events = []
        motion_detected = False
        
        if not self.previous_detections:
            # First frame - consider any confident detection as motion
            if current_detections:
                motion_events.append({
                    'type': 'initial_detection',
                    'objects': [d['class'] for d in current_detections]
                })
                motion_detected = True
            return {'motion_detected': motion_detected, 'motion_events': motion_events}
        
        # Match detections between frames (simple distance-based matching)
        matched_pairs = []
        for curr_det in current_detections:
            best_match = None
            min_distance = float('inf')
            
            for prev_det in self.previous_detections:
                if curr_det['class'] == prev_det['class']:
                    distance = np.sqrt((curr_det['center'][0] - prev_det['center'][0])**2 + 
                                     (curr_det['center'][1] - prev_det['center'][1])**2)
                    if distance < min_distance and distance < 200:  # Max matching distance
                        min_distance = distance
                        best_match = prev_det
            
            if best_match:
                matched_pairs.append((curr_det, best_match, min_distance))
        
        # Analyze matched pairs for movement and size changes
        for curr_det, prev_det, distance in matched_pairs:
            # Check for significant movement
            if distance > self.motion_threshold:
                motion_events.append({
                    'type': 'movement',
                    'object': curr_det['class'],
                    'distance': distance,
                    'from': prev_det['center'],
                    'to': curr_det['center'],
                    'confidence': curr_det['confidence']
                })
                motion_detected = True
            
            # Check size change
            if prev_det['area'] > 0:
                size_change = abs(curr_det['area'] - prev_det['area']) / prev_det['area']
                if size_change > self.size_change_threshold:
                    motion_events.append({
                        'type': 'size_change',
                        'object': curr_det['class'],
                        'change_ratio': size_change,
                        'confidence': curr_det['confidence']
                    })
                    motion_detected = True
        
        # Check for new objects (unmatched current detections)
        matched_current = [pair[0] for pair in matched_pairs]
        new_objects = [det for det in current_detections if det not in matched_current]
        if new_objects:
            for obj in new_objects:
                if obj['confidence'] > self.new_object_threshold:
                    motion_events.append({
                        'type': 'new_object',
                        'object': obj['class'],
                        'confidence': obj['confidence'],
                        'position': obj['center']
                    })
                    motion_detected = True
        
        # Check for disappeared objects
        matched_previous = [pair[1] for pair in matched_pairs]
        disappeared_objects = [det for det in self.previous_detections if det not in matched_previous]
        if disappeared_objects:
            for obj in disappeared_objects:
                motion_events.append({
                    'type': 'disappeared_object',
                    'object': obj['class'],
                    'last_position': obj['center']
                })
                motion_detected = True
        
        return {'motion_detected': motion_detected, 'motion_events': motion_events}
    
    def _draw_detections(self, frame, detections):
        """Draw detection bounding boxes and labels"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Choose color based on class
            color = self._get_class_color(det['class'])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _draw_motion_analysis(self, frame, motion_data):
        """Draw motion analysis visualization"""
        for event in motion_data['motion_events']:
            if event['type'] == 'movement':
                # Draw movement vector
                start = (int(event['from'][0]), int(event['from'][1]))
                end = (int(event['to'][0]), int(event['to'][1]))
                cv2.arrowedLine(frame, start, end, (0, 255, 0), 3)
                cv2.putText(frame, f"{event['object']}: {event['distance']:.1f}px", 
                           (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            elif event['type'] == 'new_object':
                # Draw circle for new objects
                pos = (int(event['position'][0]), int(event['position'][1]))
                cv2.circle(frame, pos, 20, (255, 0, 0), 3)
                cv2.putText(frame, f"NEW: {event['object']}", 
                           (pos[0] + 25, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame
    
    def _draw_tracking(self, frame, detections, motion_data):
        """Draw object tracking visualization"""
        # Draw all detections with tracking info
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center = (int(det['center'][0]), int(det['center'][1]))
            
            # Draw center point
            cv2.circle(frame, center, 5, (255, 255, 0), -1)
            
            # Draw ID
            cv2.putText(frame, det['id'], 
                       (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add motion summary
        summary_text = f"Motion Events: {len(motion_data['motion_events'])}"
        cv2.putText(frame, summary_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _get_class_color(self, class_name):
        """Get consistent color for object class"""
        colors = {
            'person': (255, 0, 0),
            'car': (0, 255, 0), 
            'truck': (0, 255, 0),
            'bicycle': (255, 255, 0),
            'motorcycle': (255, 255, 0),
            'bus': (0, 255, 255),
            'cat': (255, 0, 255),
            'dog': (255, 128, 0),
            'bird': (128, 255, 0),
        }
        return colors.get(class_name, (128, 128, 128))
    
    def _convert_detections_for_recording(self, detections):
        """Convert detections to format expected by recording system"""
        # Convert to simple bbox format for video recording
        bboxes = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            bboxes.append([x1, y1, x2, y2])
        return np.array(bboxes) if bboxes else None
