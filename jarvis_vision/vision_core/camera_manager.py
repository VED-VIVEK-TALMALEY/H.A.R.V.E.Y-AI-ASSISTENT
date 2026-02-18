"""
JARVIS Vision Module - Camera Manager
Day 1: Foundation Module

Handles webcam initialization, threaded capture, and frame management.
"""

import cv2
import threading
import time
from typing import Optional, Tuple
import numpy as np


class CameraManager:
    """
    Manages webcam video capture with threading for optimal performance.
    
    Features:
    - Threaded video capture (non-blocking)
    - Auto-reconnect on camera failure
    - FPS monitoring
    - Resolution management
    - Frame buffering
    """
    
    def __init__(self, camera_index: int = 0, resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize the camera manager.
        
        Args:
            camera_index: Index of the camera device (0 for default)
            resolution: Desired resolution as (width, height)
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.capture = None
        self.frame = None
        self.running = False
        self.thread = None
        
        # Performance metrics
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Status
        self.camera_available = False
        self.last_error = None
        
    def start(self) -> bool:
        """
        Start the camera capture thread.
        
        Returns:
            True if successful, False otherwise
        """
        if self.running:
            print("Camera already running")
            return True
            
        # Initialize camera
        if not self._initialize_camera():
            return False
            
        # Start capture thread
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        print(f"✓ Camera started at {self.resolution[0]}x{self.resolution[1]}")
        return True
        
    def _initialize_camera(self) -> bool:
        """
        Initialize the camera device.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            
            if not self.capture.isOpened():
                self.last_error = "Failed to open camera"
                print(f"✗ {self.last_error}")
                return False
                
            # Set resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Verify settings
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if (actual_width, actual_height) != self.resolution:
                print(f"⚠ Requested {self.resolution}, got ({actual_width}, {actual_height})")
                self.resolution = (actual_width, actual_height)
            
            self.camera_available = True
            return True
            
        except Exception as e:
            self.last_error = f"Camera initialization error: {str(e)}"
            print(f"✗ {self.last_error}")
            return False
            
    def _capture_loop(self):
        """
        Main capture loop (runs in separate thread).
        Continuously reads frames from the camera.
        """
        consecutive_failures = 0
        max_failures = 30
        
        while self.running:
            try:
                ret, frame = self.capture.read()
                
                if ret:
                    # Successfully captured frame
                    with self.lock:
                        self.frame = frame
                        self.frame_count += 1
                    consecutive_failures = 0
                    
                else:
                    # Failed to capture
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_failures:
                        print("⚠ Camera connection lost. Attempting to reconnect...")
                        self._reconnect()
                        consecutive_failures = 0
                        
                # Update FPS
                self._update_fps()
                
                # Small delay to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                print(f"✗ Capture error: {str(e)}")
                consecutive_failures += 1
                time.sleep(0.1)
                
    def _reconnect(self):
        """Attempt to reconnect to the camera."""
        if self.capture:
            self.capture.release()
            
        time.sleep(1)
        
        if self._initialize_camera():
            print("✓ Camera reconnected successfully")
        else:
            print("✗ Failed to reconnect camera")
            
    def _update_fps(self):
        """Calculate current FPS."""
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
            
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the camera.
        
        Returns:
            Latest frame as numpy array, or None if unavailable
        """
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
            
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            Current frames per second
        """
        return round(self.fps, 1)
        
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self.running and self.camera_available
        
    def stop(self):
        """Stop the camera capture thread."""
        if not self.running:
            return
            
        print("Stopping camera...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        if self.capture:
            self.capture.release()
            
        print("✓ Camera stopped")
        
    def get_status(self) -> dict:
        """Get current camera status."""
        return {
            'running': self.running,
            'available': self.camera_available,
            'fps': self.get_fps(),
            'resolution': self.resolution,
            'frame_count': self.frame_count,
            'error': self.last_error
        }
        
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()


# Test function
def test_camera():
    """Test the camera manager."""
    print("=== Camera Manager Test ===\n")
    
    camera = CameraManager()
    
    if not camera.start():
        print("✗ Failed to start camera")
        return
        
    print("✓ Camera started successfully")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            frame = camera.get_frame()
            
            if frame is not None:
                # Add FPS display
                fps_text = f"FPS: {camera.get_fps()}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('JARVIS Camera Test', frame)
                
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        
    # Print final status
    print("\n=== Final Status ===")
    status = camera.get_status()
    for key, value in status.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_camera()