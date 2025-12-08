#!/usr/bin/env python3
"""
DEMO REAL-TIME - Camera Display Only
All logic is imported from main.py
"""

import cv2
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main system
from main import FaceRecognitionAttendance

class DemoRealtime:
    def __init__(self):
        """Initialize the demo with camera only"""
        print("="*60)
        print("DEMO REAL-TIME - Camera Display")
        print("="*60)
        
        # Initialize the main system
        self.system = FaceRecognitionAttendance()
        
        # Camera settings
        self.camera_index = 0
        self.frame_width = 1280
        self.frame_height = 720
        self.fps = 30
        
        # Display settings
        self.show_confidence = True
        self.frame_count = 0
        self.frame_skip = 2  # Process every 2nd frame
        
        print("\n⚙️  Demo Settings:")
        print(f"   • Camera: {self.camera_index}")
        print(f"   • Resolution: {self.frame_width}x{self.frame_height}")
        print(f"   • FPS: {self.fps}")
        print(f"   • Frame skip: {self.frame_skip}")
        print(f"   • Show confidence: {self.show_confidence}")
        print("="*60)
    
    def open_camera(self):
        """Open and configure camera"""
        print("🔧 Opening camera...")
        
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ Cannot open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera opened successfully!")
        print(f"   • Actual resolution: {actual_width}x{actual_height}")
        print(f"   • Actual FPS: {actual_fps:.1f}")
        
        return True
    
    def run(self):
        """Main camera loop"""
        if not self.open_camera():
            return
        
        print("\n🎬 Starting camera display...")
        print("📋 CONTROLS:")
        print("   [Q] - Quit program")
        print("   [S] - Show attendance summary")
        print("   [R] - Reset tracking")
        print("   [C] - Toggle confidence display")
        print("   [T] - Change confidence threshold")
        print("="*60)
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Cannot read frame from camera")
                    break
                
                self.frame_count += 1
                
                # Mirror effect for natural viewing
                frame = cv2.flip(frame, 1)
                
                # Process frame (skip some for performance)
                if self.frame_count % self.frame_skip == 0:
                    # Detect and recognize faces
                    results = self.system.detect_and_recognize(frame)
                    
                    # Auto-mark attendance for recognized faces
                    for result in results:
                        if result['name'] != "UNKNOWN" and not result['already_marked']:
                            self.system.auto_mark_attendance(result['name'], result['confidence'])
                    
                    # Draw results on frame
                    frame = self.system.draw_results_on_frame(frame, results, self.show_confidence)
                
                # Add info panel
                frame = self.system.add_info_panel(frame, self.frame_count)
                
                # Add demo-specific info
                self.add_demo_info(frame)
                
                # Display frame
                cv2.imshow("DEMO: Face Recognition & Attendance", frame)
                
                # Handle keyboard input
                key = self.handle_keyboard_input()
                if key == 'quit':
                    break
        
        except KeyboardInterrupt:
            print("\n⚠️ Program interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.cleanup()
    
    def add_demo_info(self, frame):
        """Add demo-specific information to frame"""
        # Demo title
        cv2.putText(frame, "DEMO MODE - REAL-TIME", 
                   (frame.shape[1] - 300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Frame info
        cv2.putText(frame, f"Frame: {self.frame_count}",
                   (frame.shape[1] - 300, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Camera status
        status_text = "Camera: ✅ ACTIVE"
        cv2.putText(frame, status_text,
                   (frame.shape[1] - 300, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def handle_keyboard_input(self):
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\n👋 Quitting program...")
            return 'quit'
        
        elif key == ord('s'):
            self.system.show_summary()
            print("\n📊 Summary shown in terminal")
        
        elif key == ord('r'):
            self.system.reset_tracking()
        
        elif key == ord('c'):
            self.show_confidence = not self.show_confidence
            status = "ON" if self.show_confidence else "OFF"
            print(f"\n👁️  Confidence display: {status}")
        
        elif key == ord('t'):
            self.system.change_threshold()
        
        return 'continue'
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            print("✅ Camera released")
        
        cv2.destroyAllWindows()
        
        # Final statistics
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Present today: {len(self.system.already_marked)}/{len(self.system.class_names)}")
        
        # Show absent people
        absent = set(self.system.class_names) - self.system.already_marked
        if absent:
            print(f"\n📋 ABSENT TODAY ({len(absent)}):")
            for name in sorted(list(absent))[:5]:
                print(f"   • {name}")
            if len(absent) > 5:
                print(f"   ... and {len(absent) - 5} more")
        
        print("\n✅ Demo finished successfully!")

def main():
    """Main function to run the demo"""
    print("\n" + "="*60)
    print("🚀 LAUNCHING DEMO - REAL TIME CAMERA")
    print("="*60)
    
    # Check for required files
    required_files = ['face_recognition_model.h5', 'class_names.txt']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n📝 Please run training first!")
        return
    
    # Run the demo
    demo = DemoRealtime()
    demo.run()
    
    print("\n" + "="*60)
    print("✨ DEMO COMPLETED")
    print("="*60)

if __name__ == "__main__":
    # Show help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("\n📋 DEMO REAL-TIME - HELP")
        print("="*60)
        print("Usage:")
        print("  python demo_realtime.py          # Run the demo")
        print("  python demo_realtime.py --help   # Show this help")
        print("\nControls during demo:")
        print("  [Q] - Quit program")
        print("  [S] - Show attendance summary")
        print("  [R] - Reset tracking (memory only)")
        print("  [C] - Toggle confidence display")
        print("  [T] - Change confidence threshold")
        print("="*60)
        sys.exit(0)
    
    # Run main
    main()