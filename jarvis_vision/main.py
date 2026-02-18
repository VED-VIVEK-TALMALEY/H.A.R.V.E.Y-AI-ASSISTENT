"""
JARVIS Vision Module - Main Integration
Day 1: Complete Vision Pipeline

Brings together all Day 1 components into a working system.
"""

import cv2
import sys
import time
from vision_core.camera_manager import CameraManager
from vision_core.frame_preprocessing import FramePreprocessor
from vision_core.hand_tracking import HandTracker
from vision_core.calibration import CalibrationManager


class JARVISVisionDay1:
    """
    Day 1 complete integration of JARVIS vision system.
    
    Features:
    - Threaded camera capture
    - Frame preprocessing
    - Real-time hand tracking
    - Calibration management
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize JARVIS Vision System."""
        print("\n" + "=" * 60)
        print("🤖 JARVIS VISION MODULE - DAY 1 INTEGRATION")
        print("=" * 60 + "\n")
        
        # Load calibration
        self.calibration = CalibrationManager(config_dir="./config")
        profile = self.calibration.current_profile
        
        print(f"✓ Loaded profile: {profile.profile_name}")
        
        # Initialize components
        self.camera = CameraManager(
            camera_index=profile.camera_index,
            resolution=profile.resolution
        )
        
        self.preprocessor = FramePreprocessor(
            target_brightness=profile.brightness_target,
            enable_denoising=profile.enable_denoising,
            enhance_contrast=profile.enhance_contrast
        )
        
        self.tracker = HandTracker(
            max_hands=profile.max_hands,
            min_detection_confidence=profile.min_detection_confidence,
            min_tracking_confidence=profile.min_tracking_confidence
        )
        
        # State
        self.running = False
        self.show_landmarks = True
        self.show_fps = True
        self.frame_count = 0
        self.start_time = None
        
        print("✓ Camera Manager initialized")
        print("✓ Frame Preprocessor initialized")
        print("✓ Hand Tracker initialized")
        print("✓ System ready\n")
    
    def start(self):
        """Start the vision system."""
        if not self.camera.start():
            print("✗ Failed to start camera")
            return False
        
        self.running = True
        self.start_time = time.time()
        print("✓ JARVIS Vision System started\n")
        print("Controls:")
        print("  'q' - Quit")
        print("  'l' - Toggle landmarks")
        print("  'f' - Toggle FPS display")
        print("  'c' - Show calibration info")
        print("  's' - Save screenshot")
        print("  'r' - Reset statistics")
        print()
        
        return True
    
    def process_frame(self):
        """Process a single frame through the pipeline."""
        # Get frame from camera
        frame = self.camera.get_frame()
        if frame is None:
            return None
        
        # Preprocess frame
        frame_rgb = self.preprocessor.process(frame)
        
        # Detect hands
        hands_data = self.tracker.detect(frame_rgb)
        
        # Convert back to BGR for display
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks if enabled
        if self.show_landmarks:
            display_frame = self.tracker.draw_landmarks(
                display_frame, 
                hands_data,
                draw_connections=True
            )
        
        # Add info overlay
        if self.show_fps:
            self._add_info_overlay(display_frame, hands_data)
        
        self.frame_count += 1
        return display_frame
    
    def _add_info_overlay(self, frame, hands_data):
        """Add information overlay to frame."""
        # Background for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 25
        
        # FPS
        fps = self.camera.get_fps()
        fps_text = f"FPS: {fps:.1f}"
        color = (0, 255, 0) if fps >= 30 else (0, 165, 255) if fps >= 20 else (0, 0, 255)
        cv2.putText(frame, fps_text, (10, y_offset), font, 0.7, color, 2)
        
        # Hands detected
        y_offset += 30
        hands_text = f"Hands: {len(hands_data)}"
        cv2.putText(frame, hands_text, (10, y_offset), font, 0.6, (255, 255, 255), 2)
        
        # Hand details
        for idx, hand in enumerate(hands_data):
            y_offset += 25
            hand_info = f"  {hand.handedness}"
            if hasattr(hand, 'confidence'):
                hand_info += f" ({hand.confidence:.2f})"
            cv2.putText(frame, hand_info, (10, y_offset), font, 0.5, (200, 200, 200), 1)
        
        # Detection rate
        y_offset = 25
        x_offset = 220
        detection_rate = self.tracker.get_detection_rate()
        rate_text = f"Detection: {detection_rate:.1f}%"
        cv2.putText(frame, rate_text, (x_offset, y_offset), font, 0.6, (255, 255, 0), 2)
        
        # Session time
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            time_text = f"Time: {mins:02d}:{secs:02d}"
            cv2.putText(frame, time_text, (x_offset, y_offset + 30), font, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Main run loop."""
        if not self.start():
            return
        
        try:
            while self.running:
                # Process frame
                display_frame = self.process_frame()
                
                if display_frame is not None:
                    cv2.imshow('JARVIS Vision System - Day 1', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nShutting down...")
                    break
                    
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                    
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                    print(f"FPS Display: {'ON' if self.show_fps else 'OFF'}")
                    
                elif key == ord('c'):
                    self.calibration.print_current_profile()
                    
                elif key == ord('s'):
                    filename = f"jarvis_screenshot_{int(time.time())}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"✓ Screenshot saved: {filename}")
                    
                elif key == ord('r'):
                    self.tracker.reset_statistics()
                    self.frame_count = 0
                    self.start_time = time.time()
                    print("✓ Statistics reset")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the vision system."""
        self.running = False
        self.camera.stop()
        self.tracker.close()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final statistics."""
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        
        camera_status = self.camera.get_status()
        elapsed = int(time.time() - self.start_time) if self.start_time else 0
        
        print(f"Session Duration: {elapsed // 60}m {elapsed % 60}s")
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Average FPS: {camera_status['fps']}")
        print(f"Hand Detection Rate: {self.tracker.get_detection_rate():.1f}%")
        print(f"Successful Detections: {self.tracker.successful_detections}")
        
        print("=" * 60 + "\n")
        print("✓ JARVIS Vision System shutdown complete")
        print("\n🎉 DAY 1 COMPLETE! Great work!")


def main():
    """Main entry point."""
    # Create and run JARVIS Vision System
    jarvis = JARVISVisionDay1()
    jarvis.run()


if __name__ == "__main__":
    main()