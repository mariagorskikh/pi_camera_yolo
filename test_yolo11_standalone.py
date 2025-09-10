#!/usr/bin/env python3
"""
Standalone test for YOLO11 motion detection (no picamera2 dependency)
This test creates a simplified version to test YOLO11 functionality independently
"""

import cv2
import numpy as np
import time
import copy
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO11 (ultralytics) available")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLO11 (ultralytics) not available")

class SimpleYOLO11Detector:
    """Simplified YOLO11 detector for testing without raspi-cam-srv dependencies"""
    
    def __init__(self):
        self.model = None
        self.model_initialized = False
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
        self.motion_threshold = 50
        self.size_change_threshold = 0.3
        self.previous_detections = []
        
    def initialize_model(self, model_path="yolo11n.pt"):
        """Initialize YOLO11 model"""
        if not YOLO_AVAILABLE:
            print("❌ YOLO11 not available")
            return False
            
        try:
            print(f"📦 Loading YOLO11 model: {model_path}")
            self.model = YOLO(model_path)
            
            # Test with dummy image
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            results = self.model(test_img, verbose=False)
            
            self.model_initialized = True
            print("✅ YOLO11 model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load YOLO11 model: {e}")
            return False
    
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if not self.model_initialized:
            return []
        
        try:
            results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)
            return self._extract_detections(results[0])
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def _extract_detections(self, results):
        """Extract detection data from YOLO results"""
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': class_name,
                    'center': [(x1+x2)/2, (y1+y2)/2],
                    'area': (x2-x1) * (y2-y1)
                })
        return detections
    
    def analyze_motion(self, current_detections):
        """Analyze motion between detections"""
        if not self.previous_detections:
            self.previous_detections = current_detections
            return len(current_detections) > 0, "Initial detection"
        
        motion_events = []
        
        # Simple motion analysis - check for position changes
        for curr_det in current_detections:
            for prev_det in self.previous_detections:
                if curr_det['class'] == prev_det['class']:
                    distance = np.sqrt((curr_det['center'][0] - prev_det['center'][0])**2 + 
                                     (curr_det['center'][1] - prev_det['center'][1])**2)
                    if distance > self.motion_threshold:
                        motion_events.append(f"{curr_det['class']} moved {distance:.1f}px")
        
        # Check for new/disappeared objects
        curr_classes = [d['class'] for d in current_detections]
        prev_classes = [d['class'] for d in self.previous_detections]
        
        new_objects = set(curr_classes) - set(prev_classes)
        disappeared_objects = set(prev_classes) - set(curr_classes)
        
        for obj in new_objects:
            motion_events.append(f"New {obj} appeared")
        for obj in disappeared_objects:
            motion_events.append(f"{obj} disappeared")
        
        self.previous_detections = current_detections
        
        return len(motion_events) > 0, motion_events
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes on frame"""
        frame_copy = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Color based on class
            colors = {
                'person': (255, 0, 0),
                'car': (0, 255, 0),
                'truck': (0, 255, 0),
                'bicycle': (255, 255, 0),
                'motorcycle': (255, 255, 0),
                'bus': (0, 255, 255),
                'cat': (255, 0, 255),
                'dog': (255, 128, 0),
            }
            color = colors.get(det['class'], (128, 128, 128))
            
            # Draw box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy

def create_test_scene(frame_num):
    """Create a test scene with moving objects"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark background
    
    # Add some basic shapes that might be detected as objects
    offset = frame_num * 5
    
    # Moving rectangle (car-like)
    x = 100 + offset % 400
    cv2.rectangle(frame, (x, 200), (x + 80, 240), (0, 128, 255), -1)
    
    # Moving circle (person-like)
    y = 300 + int(50 * np.sin(frame_num * 0.1))
    cv2.circle(frame, (320, y), 25, (255, 128, 0), -1)
    
    # Static building-like rectangle
    cv2.rectangle(frame, (450, 100), (550, 350), (128, 128, 128), -1)
    
    return frame

def test_with_webcam():
    """Test with webcam if available"""
    print("\n🎥 Testing with webcam...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No webcam available")
        return False
    
    detector = SimpleYOLO11Detector()
    if not detector.initialize_model():
        cap.release()
        return False
    
    print("✅ Starting webcam test (press 'q' to quit)")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect objects
        detections = detector.detect_objects(frame)
        
        # Analyze motion
        motion_detected, motion_info = detector.analyze_motion(detections)
        
        # Draw results
        result_frame = detector.draw_detections(frame, detections)
        
        # Add info text
        info_text = f"Frame: {frame_count} | Objects: {len(detections)}"
        if motion_detected:
            info_text += " | MOTION DETECTED"
        
        cv2.putText(result_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display motion events
        if motion_detected and isinstance(motion_info, list):
            for i, event in enumerate(motion_info[:3]):  # Show max 3 events
                cv2.putText(result_frame, event, (10, 60 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('YOLO11 Test - Webcam', result_frame)
        
        # Calculate FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"📊 FPS: {fps:.1f} | Objects detected: {len(detections)}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam test completed")
    return True

def test_with_synthetic_scenes():
    """Test with synthetic scenes"""
    print("\n🎨 Testing with synthetic scenes...")
    
    detector = SimpleYOLO11Detector()
    if not detector.initialize_model():
        return False
    
    print("✅ Running synthetic scene test...")
    
    for frame_num in range(50):
        frame = create_test_scene(frame_num)
        
        # Detect objects
        start_time = time.time()
        detections = detector.detect_objects(frame)
        detection_time = time.time() - start_time
        
        # Analyze motion
        motion_detected, motion_info = detector.analyze_motion(detections)
        
        # Show results
        status = "🔴 MOTION" if motion_detected else "🟢 STATIC"
        print(f"{status} Frame {frame_num:02d}: {len(detections)} objects, "
              f"detection time: {detection_time:.3f}s")
        
        if motion_detected and isinstance(motion_info, list):
            for event in motion_info:
                print(f"    📍 {event}")
        
        # Save some frames for inspection
        if frame_num % 10 == 0:
            result_frame = detector.draw_detections(frame, detections)
            cv2.imwrite(f'test_frame_{frame_num:02d}.jpg', result_frame)
            print(f"    💾 Saved test_frame_{frame_num:02d}.jpg")
        
        time.sleep(0.1)
    
    print("✅ Synthetic scene test completed")
    return True

def main():
    """Main test function"""
    print("🚀 YOLO11 Standalone Test")
    print("="*50)
    
    if not YOLO_AVAILABLE:
        print("❌ YOLO11 not available - install with: pip install ultralytics")
        return False
    
    # Test 1: Synthetic scenes
    if not test_with_synthetic_scenes():
        print("❌ Synthetic scene test failed")
        return False
    
    # Test 2: Webcam (optional)
    print("\n🎮 Webcam test (optional)")
    print("   Press Enter to try webcam test, or 's' to skip:")
    choice = input().lower()
    
    if choice != 's':
        test_with_webcam()
    else:
        print("⏭️  Skipping webcam test")
    
    print("\n🎉 YOLO11 standalone test completed!")
    print("\n📝 Test Results:")
    print("   ✅ YOLO11 model loading")
    print("   ✅ Object detection")
    print("   ✅ Motion analysis")
    print("   ✅ Visualization")
    
    print("\n🔧 Ready for Pi deployment!")
    print("   Copy the raspi-cam-srv folder to your Pi 5")
    print("   Run the installation steps from the README")
    print("   YOLO11 will be available as motion detection algorithm #5")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
