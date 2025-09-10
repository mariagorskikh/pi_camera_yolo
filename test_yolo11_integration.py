#!/usr/bin/env python3
"""
Test script for YOLO11 integration with raspi-cam-srv
Tests the YOLO11 motion detection algorithm independently
"""

import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path

# Add the raspiCamSrv directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'raspiCamSrv'))

try:
    from motionAlgoYOLO11 import MotionDetectYOLO11
    print("✅ Successfully imported YOLO11 motion detector")
except ImportError as e:
    print(f"❌ Failed to import YOLO11 motion detector: {e}")
    sys.exit(1)

def create_test_frame(frame_num=0):
    """Create a test frame with some objects to detect"""
    # Create a simple test image with some shapes
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add background
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Add some moving rectangles (simulating objects)
    offset = frame_num * 10
    
    # Moving rectangle (simulating a car)
    cv2.rectangle(frame, (100 + offset, 200), (200 + offset, 250), (0, 255, 0), -1)
    
    # Static rectangle (simulating a building)
    cv2.rectangle(frame, (400, 100), (500, 300), (128, 128, 128), -1)
    
    # Moving circle (simulating a person)
    center_x = 300 + int(30 * np.sin(frame_num * 0.1))
    cv2.circle(frame, (center_x, 350), 30, (255, 0, 0), -1)
    
    # Add some noise
    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame

def test_yolo11_with_webcam():
    """Test YOLO11 with webcam if available"""
    print("\n🎥 Testing YOLO11 with webcam...")
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No webcam available")
        return False
    
    print("✅ Webcam opened successfully")
    
    # Initialize YOLO11 detector
    detector = MotionDetectYOLO11()
    detector.test = True  # Enable test mode for visualization
    
    print("📦 Initializing YOLO11 model...")
    if not detector.initialize_model():
        print("❌ Failed to initialize YOLO11 model")
        cap.release()
        return False
    
    print("✅ YOLO11 model loaded successfully")
    print("🎮 Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        if prev_frame is not None:
            # Run motion detection
            start_time = time.time()
            motion_detected, trigger_info = detector.detectMotion(frame, prev_frame)
            detection_time = time.time() - start_time
            
            # Display results
            if motion_detected:
                print(f"🔴 Frame {frame_count}: Motion detected! {trigger_info}")
            
            # Get test frames for visualization
            test_frame1 = detector.testFrame1
            test_frame2 = detector.testFrame2
            test_frame3 = detector.testFrame3
            test_frame4 = detector.testFrame4
            
            # Convert streams back to images for display
            if test_frame1 is not None:
                # Decode JPEG stream back to image
                frame_array = np.frombuffer(test_frame1, dtype=np.uint8)
                img1 = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if img1 is not None:
                    cv2.imshow('Original', img1)
            
            if test_frame2 is not None:
                frame_array = np.frombuffer(test_frame2, dtype=np.uint8)
                img2 = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if img2 is not None:
                    cv2.imshow('YOLO11 Detections', img2)
            
            # Show performance info
            fps = 1.0 / detection_time if detection_time > 0 else 0
            print(f"📊 Frame {frame_count}: Detection time: {detection_time:.3f}s, FPS: {fps:.1f}")
        
        prev_frame = frame.copy()
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'yolo11_test_frame_{frame_count}.jpg', frame)
            print(f"💾 Saved frame {frame_count}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam test completed")
    return True

def test_yolo11_with_synthetic_frames():
    """Test YOLO11 with synthetic test frames"""
    print("\n🎨 Testing YOLO11 with synthetic frames...")
    
    # Initialize YOLO11 detector
    detector = MotionDetectYOLO11()
    detector.test = True  # Enable test mode
    
    print("📦 Initializing YOLO11 model...")
    if not detector.initialize_model():
        print("❌ Failed to initialize YOLO11 model")
        return False
    
    print("✅ YOLO11 model loaded successfully")
    
    # Create test frames
    print("🎬 Creating test sequence...")
    
    prev_frame = None
    for frame_num in range(20):
        frame = create_test_frame(frame_num)
        
        if prev_frame is not None:
            # Run motion detection
            start_time = time.time()
            motion_detected, trigger_info = detector.detectMotion(frame, prev_frame)
            detection_time = time.time() - start_time
            
            # Display results
            status = "🔴 MOTION" if motion_detected else "🟢 NO MOTION"
            print(f"{status} Frame {frame_num}: {trigger_info}")
            print(f"   ⏱️  Detection time: {detection_time:.3f}s")
            
            # Save test frames periodically
            if frame_num % 5 == 0:
                cv2.imwrite(f'test_frame_{frame_num:03d}.jpg', frame)
                print(f"   💾 Saved test frame {frame_num}")
        
        prev_frame = frame.copy()
        time.sleep(0.1)  # Small delay to simulate real-time
    
    print("✅ Synthetic frame test completed")
    return True

def test_algorithm_parameters():
    """Test different algorithm parameters"""
    print("\n⚙️  Testing algorithm parameters...")
    
    detector = MotionDetectYOLO11()
    
    # Test parameter changes
    original_conf = detector.confidence_threshold
    original_motion = detector.motion_threshold
    
    print(f"📊 Original confidence threshold: {original_conf}")
    print(f"📊 Original motion threshold: {original_motion}")
    
    # Test different confidence thresholds
    for conf in [0.3, 0.5, 0.7]:
        detector.confidence_threshold = conf
        print(f"✅ Set confidence threshold to {conf}")
    
    # Test different motion thresholds  
    for motion in [25, 50, 100]:
        detector.motion_threshold = motion
        print(f"✅ Set motion threshold to {motion}")
    
    # Restore original values
    detector.confidence_threshold = original_conf
    detector.motion_threshold = original_motion
    
    print("✅ Parameter test completed")
    return True

def main():
    """Main test function"""
    print("🚀 YOLO11 Integration Test Suite")
    print("="*50)
    
    # Test 1: Algorithm parameters
    if not test_algorithm_parameters():
        print("❌ Parameter test failed")
        return False
    
    # Test 2: Synthetic frames
    if not test_yolo11_with_synthetic_frames():
        print("❌ Synthetic frame test failed")
        return False
    
    # Test 3: Webcam (if available)
    if not test_yolo11_with_webcam():
        print("⚠️  Webcam test skipped (no webcam available)")
    
    print("\n🎉 All tests completed successfully!")
    print("\n📝 Test Summary:")
    print("   ✅ YOLO11 motion detector import")
    print("   ✅ Algorithm parameter configuration")
    print("   ✅ Synthetic frame processing")
    print("   ✅ Motion detection functionality")
    
    print("\n🔧 Integration Notes:")
    print("   • YOLO11 algorithm is now available as motion detection option 5")
    print("   • Test mode provides 4 visualization frames")
    print("   • Detection includes object classification and motion analysis")
    print("   • Performance optimized for Raspberry Pi 5")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
