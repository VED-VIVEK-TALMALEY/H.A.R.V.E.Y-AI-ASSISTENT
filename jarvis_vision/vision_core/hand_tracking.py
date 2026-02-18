"""
JARVIS Vision Module - Hand Tracking (Using cvzone)
Day 1: Hand Detection & Landmark Extraction
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

try:
    from cvzone.HandTrackingModule import HandDetector
    CVZONE_AVAILABLE = True
except ImportError:
    CVZONE_AVAILABLE = False
    print("⚠ cvzone not available, install with: pip install cvzone")


class HandLandmark(Enum):
    """Hand landmark indices."""
    WRIST = 0
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_TIP = 20


@dataclass
class HandData:
    """Container for hand tracking data."""
    landmarks: np.ndarray
    landmarks_pixel: np.ndarray
    handedness: str
    confidence: float
    palm_center: Tuple[int, int] = None
    
    def __post_init__(self):
        if self.landmarks_pixel is not None and len(self.landmarks_pixel) >= 21:
            wrist = self.landmarks_pixel[0]
            index_mcp = self.landmarks_pixel[5]
            pinky_mcp = self.landmarks_pixel[17]
            self.palm_center = tuple(
                np.mean([wrist, index_mcp, pinky_mcp], axis=0).astype(int)
            )


class HandTracker:
    """Hand tracker using cvzone."""
    
    def __init__(self, max_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        if not CVZONE_AVAILABLE:
            raise ImportError("cvzone not installed. Run: pip install cvzone")
        
        self.detector = HandDetector(
            detectionCon=min_detection_confidence,
            maxHands=max_hands
        )
        self.total_detections = 0
        self.successful_detections = 0
        print("✓ cvzone HandDetector initialized")
    
    def detect(self, frame_rgb: np.ndarray) -> List[HandData]:
        """Detect hands in frame."""
        if frame_rgb is None:
            return []
        
        self.total_detections += 1
        
        # cvzone expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Detect hands
        hands, img = self.detector.findHands(frame_bgr, draw=False)
        
        hands_data = []
        if hands:
            self.successful_detections += 1
            
            for hand in hands:
                # Extract landmarks
                lm_list = hand['lmList']  # List of 21 landmarks [x, y, z]
                hand_type = hand['type']  # "Left" or "Right"
                
                landmarks = np.array([[lm[0]/frame_bgr.shape[1], 
                                      lm[1]/frame_bgr.shape[0], 
                                      lm[2]/frame_bgr.shape[1]] for lm in lm_list])
                landmarks_pixel = np.array([[lm[0], lm[1]] for lm in lm_list])
                
                hand_data = HandData(
                    landmarks=landmarks,
                    landmarks_pixel=landmarks_pixel,
                    handedness=hand_type,
                    confidence=0.9  # cvzone doesn't provide confidence
                )
                hands_data.append(hand_data)
        
        return hands_data
    
    def draw_landmarks(self, frame, hands_data, draw_connections=True):
        """Draw hand landmarks."""
        annotated = frame.copy()
        
        for hand_data in hands_data:
            for idx, (x, y) in enumerate(hand_data.landmarks_pixel):
                if idx in [4, 8, 12, 16, 20]:  # Finger tips
                    color = (0, 255, 0)
                    radius = 6
                elif idx == 0:  # Wrist
                    color = (255, 0, 0)
                    radius = 8
                else:
                    color = (0, 0, 255)
                    radius = 4
                
                cv2.circle(annotated, (int(x), int(y)), radius, color, -1)
                cv2.circle(annotated, (int(x), int(y)), radius + 2, (255, 255, 255), 1)
            
            if draw_connections:
                self._draw_connections(annotated, hand_data.landmarks_pixel)
            
            if hand_data.palm_center:
                cv2.circle(annotated, hand_data.palm_center, 8, (255, 255, 0), -1)
            
            wrist = hand_data.landmarks_pixel[0]
            label = f"{hand_data.handedness}"
            cv2.putText(annotated, label, (int(wrist[0]) - 50, int(wrist[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def _draw_connections(self, frame, landmarks_pixel):
        """Draw connections between landmarks."""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for start, end in connections:
            if start < len(landmarks_pixel) and end < len(landmarks_pixel):
                pt1 = tuple(landmarks_pixel[start].astype(int))
                pt2 = tuple(landmarks_pixel[end].astype(int))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
    
    def get_detection_rate(self):
        if self.total_detections == 0:
            return 0.0
        return (self.successful_detections / self.total_detections) * 100
    
    def reset_statistics(self):
        self.total_detections = 0
        self.successful_detections = 0
    
    def close(self):
        pass
    
    def __del__(self):
        pass


def test_hand_tracker():
    """Test hand tracking."""
    import sys
    sys.path.append('..')
    from vision_core.camera_manager import CameraManager
    from vision_core.frame_preprocessing import FramePreprocessor
    
    print("=== Hand Tracker Test ===\n")
    
    camera = CameraManager()
    preprocessor = FramePreprocessor()
    
    try:
        tracker = HandTracker(max_hands=2)
    except ImportError as e:
        print(f"✗ {e}")
        return
    
    if not camera.start():
        print("✗ Failed to start camera")
        return
    
    print("✓ Camera started")
    print("\nShow your hand(s) to the camera")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            
            frame_rgb = preprocessor.process(frame)
            hands_data = tracker.detect(frame_rgb)
            
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            annotated = tracker.draw_landmarks(frame_bgr, hands_data)
            
            info = f"FPS: {camera.get_fps()} | Hands: {len(hands_data)} | Rate: {tracker.get_detection_rate():.1f}%"
            cv2.putText(annotated, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            y = 60
            for idx, hand in enumerate(hands_data):
                text = f"Hand {idx+1}: {hand.handedness}"
                cv2.putText(annotated, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y += 25
            
            cv2.imshow('JARVIS Hand Tracker', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        tracker.close()
        camera.stop()
        cv2.destroyAllWindows()
    
    print(f"\n✓ Test completed - Detection rate: {tracker.get_detection_rate():.1f}%")


if __name__ == "__main__":
    test_hand_tracker()