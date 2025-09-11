#!/usr/bin/env python3
"""
Debug script to test YOLO import and show detailed error info
"""

print("🔍 Debugging YOLO11 import...")
print("=" * 50)

# Test 1: Basic import
print("1. Testing basic ultralytics import...")
try:
    import ultralytics
    print(f"✅ ultralytics imported successfully: {ultralytics.__version__}")
except ImportError as e:
    print(f"❌ ultralytics import failed: {e}")
    exit(1)

# Test 2: YOLO class import
print("\n2. Testing YOLO class import...")
try:
    from ultralytics import YOLO
    print("✅ YOLO class imported successfully")
except ImportError as e:
    print(f"❌ YOLO class import failed: {e}")
    exit(1)

# Test 3: Model loading
print("\n3. Testing YOLO model loading...")
try:
    model = YOLO('yolo11n.pt')
    print("✅ YOLO11n model loaded successfully")
except Exception as e:
    print(f"❌ YOLO model loading failed: {e}")

# Test 4: Check what the yolo_live module sees
print("\n4. Testing yolo_live module import logic...")
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO_AVAILABLE = True (same logic as yolo_live.py)")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLO_AVAILABLE = False (same logic as yolo_live.py)")

print(f"\n🎯 Final result: YOLO_AVAILABLE = {YOLO_AVAILABLE}")

# Test 5: Check virtual environment
print("\n5. Checking Python environment...")
import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

# Test 6: Check ultralytics installation location
print("\n6. Checking ultralytics installation...")
try:
    import ultralytics
    print(f"Ultralytics location: {ultralytics.__file__}")
except:
    print("❌ Cannot find ultralytics location")

print("\n" + "=" * 50)
print("🏁 Debug complete!")
