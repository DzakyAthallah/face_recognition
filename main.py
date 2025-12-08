"""
MAIN PROGRAM - Sistem Face Recognition dengan Attendance
Semua logika dan fungsi ada di sini
"""

import cv2
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime, date
from tensorflow.keras.models import load_model

class FaceRecognitionAttendance:
    def __init__(self, model_path='face_recognition_model.h5', class_names_file='class_names.txt'):
        """Initialize the complete system"""
        print("="*60)
        print("MAIN SYSTEM - Face Recognition & Attendance")
        print("="*60)
        
        # Load model
        self.model = load_model(model_path)
        self.img_size = (100, 100)
        
        # Load class names
        with open(class_names_file, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"✅ Model loaded: {model_path}")
        print(f"📊 People in database: {len(self.class_names)}")
        
        # Face detector
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Attendance system
        self.init_attendance_system()
        
        # Recognition settings
        self.confidence_threshold = 70.0
        self.cooldown_seconds = 10
        self.frame_skip = 2
        
        print("✅ Main system initialized successfully!")
        print("="*60)
    
    def init_attendance_system(self):
        """Initialize attendance tracking system"""
        # Create data folder
        self.data_folder = "attendance_data"
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
        # Today's date and file
        self.today_date = date.today()
        self.date_str = self.today_date.strftime("%Y-%m-%d")
        self.attendance_file = os.path.join(self.data_folder, f"attendance_{self.date_str}.csv")
        
        # Tracking dictionaries
        self.already_marked = set()  # Who has attended today
        self.cooldown_dict = {}      # For spam prevention
        self.attendance_data = pd.DataFrame()  # DataFrame for data
        
        # Load or create attendance file
        self.load_or_create_attendance_file()
        
        print(f"📅 Date: {self.date_str}")
        print(f"📁 Attendance file: {self.attendance_file}")
    
    def load_or_create_attendance_file(self):
        """Load existing attendance or create new file"""
        if os.path.exists(self.attendance_file):
            try:
                self.attendance_data = pd.read_csv(self.attendance_file)
                
                # Get who's already marked today
                if not self.attendance_data.empty:
                    today_mask = self.attendance_data['Tanggal'] == self.date_str
                    if today_mask.any():
                        today_names = self.attendance_data[today_mask]['Nama'].tolist()
                        self.already_marked = set(today_names)
                        print(f"📊 Already marked today: {len(self.already_marked)} people")
            except Exception as e:
                print(f"⚠️ Error loading attendance: {e}")
                self.create_new_attendance_file()
        else:
            self.create_new_attendance_file()
    
    def create_new_attendance_file(self):
        """Create new attendance CSV file"""
        with open(self.attendance_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Nama', 'Waktu', 'Tanggal', 'Confidence', 'Status'])
        print("📄 Created new attendance file")
    
    # ===== FACE PROCESSING METHODS =====
    def preprocess_face(self, face_img):
        """Preprocess face image for model (SAME as training)"""
        # 1. Resize to model input size
        face_resized = cv2.resize(face_img, self.img_size)
        
        # 2. Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize to 0-1
        face_normalized = face_rgb.astype('float32') / 255.0
        
        # 4. Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def detect_and_recognize(self, frame):
        """Detect faces in frame and recognize them"""
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            try:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Skip if face is too small
                if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                    continue
                
                # Preprocess and predict
                processed = self.preprocess_face(face_roi)
                predictions = self.model.predict(processed, verbose=0)[0]
                
                # Get best match
                max_idx = np.argmax(predictions)
                confidence = predictions[max_idx] * 100
                name = self.class_names[max_idx]
                
                # Add to results
                results.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'name': name,
                    'confidence': confidence,
                    'already_marked': name in self.already_marked
                })
                
            except Exception as e:
                # print(f"Warning: {e}")
                continue
        
        return results
    
    # ===== ATTENDANCE METHODS =====
    def mark_attendance(self, name, confidence):
        """Mark attendance for a person"""
        current_time = datetime.now()
        
        # 1. Check cooldown
        if name in self.cooldown_dict:
            elapsed = (current_time - self.cooldown_dict[name]).seconds
            if elapsed < self.cooldown_seconds:
                return False, "In cooldown"
        
        # 2. Check if already marked today
        if name in self.already_marked:
            return False, "Already marked today"
        
        # 3. Determine status based on time
        hour = current_time.hour
        if hour < 8:
            status = "Very Early"
        elif hour < 9:
            status = "On Time"
        elif hour < 10:
            status = "Slightly Late"
        elif hour < 11:
            status = "Late"
        else:
            status = "Very Late"
        
        # 4. Create record
        record = {
            'Nama': name,
            'Waktu': current_time.strftime("%H:%M:%S"),
            'Tanggal': self.date_str,
            'Confidence': f"{confidence:.1f}%",
            'Status': status
        }
        
        # 5. Save to CSV
        try:
            with open(self.attendance_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['Nama', 'Waktu', 'Tanggal', 'Confidence', 'Status'])
                writer.writerow(record)
            
            # Update tracking
            self.already_marked.add(name)
            self.cooldown_dict[name] = current_time
            
            # Update DataFrame
            new_row = pd.DataFrame([record])
            self.attendance_data = pd.concat([self.attendance_data, new_row], ignore_index=True)
            
            return True, f"Marked as {status}"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    def auto_mark_attendance(self, name, confidence):
        """Automatically mark attendance if confidence is high enough"""
        if confidence >= self.confidence_threshold:
            success, message = self.mark_attendance(name, confidence)
            if success:
                return True, "Attendance marked"
            else:
                return False, message
        return False, f"Low confidence ({confidence:.1f}%)"
    
    # ===== DISPLAY METHODS =====
    def draw_results_on_frame(self, frame, results, show_confidence=True):
        """Draw detection results on the frame"""
        for result in results:
            x, y, w, h = result['x'], result['y'], result['w'], result['h']
            name = result['name']
            confidence = result['confidence']
            already_marked = result['already_marked']
            
            # Determine colors
            if name == "UNKNOWN" or confidence < 30:
                color = (0, 0, 255)  # Red: unknown/low confidence
                status_color = (0, 0, 255)
            elif already_marked:
                color = (255, 255, 0)  # Yellow: already marked
                status_color = (255, 255, 0)
            else:
                color = (0, 255, 0)  # Green: ready to mark
                status_color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw name
            name_text = name
            cv2.putText(frame, name_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw confidence if enabled
            if show_confidence:
                conf_text = f"{confidence:.1f}%"
                cv2.putText(frame, conf_text, (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # Draw status
            if already_marked:
                status_text = "✓ Already marked"
            elif confidence >= self.confidence_threshold:
                status_text = "✓ Ready to mark"
            else:
                status_text = "✗ Low confidence"
            
            cv2.putText(frame, status_text, (x, y+h+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        return frame
    
    def add_info_panel(self, frame, frame_count=0):
        """Add information panel to frame"""
        # System info
        info_y = 30
        line_height = 25
        
        info_lines = [
            f"Date: {self.date_str}",
            f"Present: {len(self.already_marked)}/{len(self.class_names)}",
            f"Threshold: {self.confidence_threshold}%",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, info_y + (i * line_height)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls
        controls = "[Q]Quit [S]Summary [R]Reset [C]Confidence [T]Threshold"
        cv2.putText(frame, controls, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Frame counter
        if frame_count > 0:
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    # ===== UTILITY METHODS =====
    def show_summary(self):
        """Show attendance summary"""
        print("\n" + "="*60)
        print(f"📊 ATTENDANCE SUMMARY - {self.date_str}")
        print("="*60)
        
        if self.attendance_data.empty:
            print("📭 No attendance data yet")
            return
        
        # Filter for today
        today_data = self.attendance_data[self.attendance_data['Tanggal'] == self.date_str]
        
        if today_data.empty:
            print("📭 No attendance today")
            return
        
        print(f"{'No.':<4} {'Name':<15} {'Time':<10} {'Status':<15} {'Confidence':<10}")
        print("-" * 60)
        
        for i, (_, row) in enumerate(today_data.iterrows(), 1):
            print(f"{i:<4} {row['Nama']:<15} {row['Waktu']:<10} "
                  f"{row['Status']:<15} {row['Confidence']:<10}")
        
        # Statistics
        total = len(today_data)
        total_people = len(self.class_names)
        
        print(f"\n📈 STATISTICS:")
        print(f"   Total Present: {total}/{total_people}")
        print(f"   Attendance Rate: {(total/total_people*100):.1f}%")
        
        # Status breakdown
        if 'Status' in today_data.columns:
            status_counts = today_data['Status'].value_counts()
            for status, count in status_counts.items():
                print(f"   • {status}: {count}")
        
        # Export to Excel
        try:
            excel_file = os.path.join(self.data_folder, f"summary_{self.date_str}.xlsx")
            today_data.to_excel(excel_file, index=False)
            print(f"\n💾 Summary saved to: {excel_file}")
        except Exception as e:
            print(f"⚠️ Could not save Excel: {e}")
        
        print("="*60)
    
    def reset_tracking(self):
        """Reset attendance tracking (memory only)"""
        print("\n⚠️ RESET TRACKING")
        print("   This only clears memory, CSV file remains intact.")
        
        confirm = input("   Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            self.already_marked.clear()
            self.cooldown_dict.clear()
            print("   ✅ Tracking reset")
            return True
        else:
            print("   ❌ Reset cancelled")
            return False
    
    def change_threshold(self):
        """Change confidence threshold"""
        print(f"\n⚙️ CHANGE CONFIDENCE THRESHOLD")
        print(f"   Current: {self.confidence_threshold}%")
        
        try:
            new_thresh = float(input("   New threshold (0-100): ").strip())
            if 0 <= new_thresh <= 100:
                old_thresh = self.confidence_threshold
                self.confidence_threshold = new_thresh
                print(f"   ✅ Changed: {old_thresh}% → {new_thresh}%")
                return True
            else:
                print("   ❌ Must be between 0-100")
                return False
        except ValueError:
            print("   ❌ Invalid input")
            return False

# ===== SIMPLE RUN FUNCTION =====
def run_simple_demo():
    """Simple function to run the system (for testing)"""
    print("Running simple demo...")
    
    # Initialize system
    system = FaceRecognitionAttendance()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("\n🎥 Camera opened successfully!")
    print("Press 'q' to quit")
    
    frame_count = 0
    show_confidence = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 2nd frame for performance
        if frame_count % 2 == 0:
            # Detect and recognize faces
            results = system.detect_and_recognize(frame)
            
            # Auto-mark attendance for recognized faces
            for result in results:
                if result['name'] != "UNKNOWN" and not result['already_marked']:
                    system.auto_mark_attendance(result['name'], result['confidence'])
            
            # Draw results
            frame = system.draw_results_on_frame(frame, results, show_confidence)
        
        # Add info panel
        frame = system.add_info_panel(frame, frame_count)
        
        # Display
        cv2.imshow("Face Recognition System", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Demo finished")

if __name__ == "__main__":
    print("This is the main system module.")
    print("Import this module and use FaceRecognitionAttendance class.")
    print("Or run demo_realtime.py for the full application.")