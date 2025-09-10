#!/usr/bin/env python3
"""
Setup script for testing YOLO11 integration
Creates minimal configuration for testing without full Pi setup
"""

import os
import sys
import tempfile
import json
from pathlib import Path

def create_mock_camera_config():
    """Create a mock camera configuration for testing"""
    print("📋 Creating mock camera configuration...")
    
    # Create a temporary directory for test files
    test_dir = Path(tempfile.gettempdir()) / "raspi_cam_test"
    test_dir.mkdir(exist_ok=True)
    
    # Mock configuration
    config = {
        "motion_detection": {
            "algorithm": 5,  # YOLO11
            "confidence_threshold": 0.5,
            "iou_threshold": 0.4,
            "motion_threshold": 50,
            "size_change_threshold": 0.3
        },
        "camera": {
            "active_camera": 0,
            "stream_size": [640, 480]
        },
        "test_mode": True,
        "test_directory": str(test_dir)
    }
    
    config_file = test_dir / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Mock configuration created: {config_file}")
    return config_file, test_dir

def setup_environment():
    """Set up environment variables for testing"""
    print("🔧 Setting up test environment...")
    
    # Set environment variables
    os.environ['RASPI_CAM_TEST_MODE'] = '1'
    os.environ['RASPI_CAM_NO_GPIO'] = '1'  # Disable GPIO for non-Pi testing
    
    print("✅ Environment variables set")

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    try:
        import ultralytics
        print("✅ ultralytics available")
    except ImportError:
        missing_deps.append("ultralytics")
    
    try:
        import cv2
        print("✅ opencv-python available")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
        print("✅ numpy available")
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("📦 Install with: pip install " + " ".join(missing_deps))
        return False
    
    print("✅ All dependencies available")
    return True

def download_yolo_model():
    """Download YOLO11 model for testing"""
    print("📥 Checking YOLO11 model...")
    
    try:
        from ultralytics import YOLO
        
        # This will download the model if not present
        model = YOLO('yolo11n.pt')
        print("✅ YOLO11n model ready")
        
        # Test the model
        import numpy as np
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(test_img, verbose=False)
        print("✅ YOLO11 model test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO11 model setup failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 YOLO11 Test Environment Setup")
    print("="*40)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Setup failed - missing dependencies")
        return False
    
    # Setup environment
    setup_environment()
    
    # Create mock configuration
    config_file, test_dir = create_mock_camera_config()
    
    # Download and test YOLO model
    if not download_yolo_model():
        print("❌ Setup failed - YOLO model issues")
        return False
    
    print("\n🎉 Test environment setup complete!")
    print(f"📁 Test directory: {test_dir}")
    print(f"⚙️  Config file: {config_file}")
    
    print("\n🔧 Next steps:")
    print("   1. Run: python test_yolo11_integration.py")
    print("   2. For Pi deployment: copy files to Pi and run setup")
    print("   3. Access raspi-cam-srv web interface to configure YOLO11")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
