"""
JARVIS Vision Module - Landmark Parser (CORRECTED with Thumb Angle)
Day 2: Geometric Feature Extraction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
sys.path.append('..')
from vision_core.hand_tracking import HandData, HandLandmark
from gesture_engine.gesture_config import GestureConfig as cfg


@dataclass
class HandFeatures:
    """Container for extracted hand features."""
    fingers_extended: List[bool]
    fingers_count: int
    
    # Distances
    thumb_index_distance: float
    palm_to_fingertips: Dict[str, float]
    fingertips_to_palm_center: Dict[str, float]  # NEW
    
    # Angles
    finger_angles: Dict[str, float]
    thumb_angle: float  # NEW: Specific thumb angle
    hand_orientation: float
    
    # Palm properties
    palm_center: Tuple[int, int]
    palm_size: float
    palm_compactness: float  # NEW: How compact the hand is (fist detection)
    
    # Hand direction
    is_vertical: bool
    is_horizontal: bool
    
    # Raw data reference
    hand_data: HandData


class LandmarkParser:
    """Parses hand landmarks to extract geometric features."""
    
    def __init__(self):
        """Initialize landmark parser."""
        self.finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        self.finger_pips = {
            'thumb': 2,
            'index': 6,
            'middle': 10,
            'ring': 14,
            'pinky': 18
        }
        
        self.finger_mcps = {
            'thumb': 1,
            'index': 5,
            'middle': 9,
            'ring': 13,
            'pinky': 17
        }
    
    def parse(self, hand_data: HandData) -> HandFeatures:
        """Parse hand landmarks into geometric features."""
        # Extract finger extension states
        fingers_extended = self._get_fingers_extended(hand_data)
        fingers_count = sum(fingers_extended)
        
        # Calculate distances
        thumb_index_dist = self._calculate_thumb_index_distance(hand_data)
        palm_distances = self._calculate_palm_to_fingertip_distances(hand_data)
        fingertip_palm_distances = self._calculate_fingertips_to_palm_center(hand_data)
        
        # Calculate angles
        finger_angles = self._calculate_finger_angles(hand_data)
        thumb_angle = self._calculate_thumb_angle(hand_data)
        hand_orientation = self._calculate_hand_orientation(hand_data)
        
        # Palm properties
        palm_center = hand_data.palm_center if hand_data.palm_center else (0, 0)
        palm_size = self._estimate_palm_size(hand_data)
        palm_compactness = self._calculate_palm_compactness(hand_data)
        
        # Hand direction
        is_vertical = abs(hand_orientation) < 30 or abs(hand_orientation) > 150
        is_horizontal = 60 < abs(hand_orientation) < 120
        
        return HandFeatures(
            fingers_extended=fingers_extended,
            fingers_count=fingers_count,
            thumb_index_distance=thumb_index_dist,
            palm_to_fingertips=palm_distances,
            fingertips_to_palm_center=fingertip_palm_distances,
            finger_angles=finger_angles,
            thumb_angle=thumb_angle,
            hand_orientation=hand_orientation,
            palm_center=palm_center,
            palm_size=palm_size,
            palm_compactness=palm_compactness,
            is_vertical=is_vertical,
            is_horizontal=is_horizontal,
            hand_data=hand_data
        )
    
    def _get_fingers_extended(self, hand_data: HandData) -> List[bool]:
        """Determine which fingers are extended."""
        landmarks = hand_data.landmarks
        landmarks_pixel = hand_data.landmarks_pixel
        
        fingers = []
        
        # Thumb (uses angle-based detection)
        thumb_extended = self._is_thumb_extended_by_angle(hand_data)
        fingers.append(thumb_extended)
        
        # Other fingers
        wrist = landmarks_pixel[0]
        
        for finger_name in ['index', 'middle', 'ring', 'pinky']:
            tip_idx = self.finger_tips[finger_name]
            pip_idx = self.finger_pips[finger_name]
            mcp_idx = self.finger_mcps[finger_name]
            
            tip = landmarks_pixel[tip_idx]
            pip = landmarks_pixel[pip_idx]
            mcp = landmarks_pixel[mcp_idx]
            
            # Distance-based detection
            tip_dist_y = abs(tip[1] - wrist[1])
            pip_dist_y = abs(pip[1] - wrist[1])
            mcp_dist_y = abs(mcp[1] - wrist[1])
            
            tip_above_pip = tip[1] < pip[1]
            tip_farther = tip_dist_y > pip_dist_y * cfg.FINGER_EXTENSION_RATIO
            good_alignment = tip_dist_y > mcp_dist_y * cfg.FINGER_ALIGNMENT_RATIO
            
            extended = tip_above_pip and tip_farther and good_alignment
            fingers.append(extended)
        
        return fingers
    
    def _is_thumb_extended_by_angle(self, hand_data: HandData) -> bool:
        """
        Check if thumb is extended using ANGLE method.
        This is more reliable than distance-based detection.
        """
        # Calculate thumb angle
        thumb_angle = self._calculate_thumb_angle(hand_data)
        
        # Thumb is extended if angle is large (thumb pointing away from palm)
        # Angle > 140° means thumb is extended
        angle_extended = thumb_angle > cfg.THUMB_ANGLE_EXTENDED_MIN
        
        # Also check distance as secondary validation
        landmarks_pixel = hand_data.landmarks_pixel
        thumb_tip = landmarks_pixel[4]
        wrist = landmarks_pixel[0]
        index_mcp = landmarks_pixel[5]
        
        # Horizontal distance from palm line
        palm_line_x = abs(index_mcp[0] - wrist[0])
        thumb_offset_x = abs(thumb_tip[0] - wrist[0])
        distance_extended = thumb_offset_x > palm_line_x * cfg.THUMB_HORIZONTAL_THRESHOLD
        
        # Both conditions should be true
        return angle_extended and distance_extended
    
    def _calculate_thumb_angle(self, hand_data: HandData) -> float:
        landmarks_pixel = hand_data.landmarks_pixel
        index_mcp = landmarks_pixel[5]   # Base of index finger
        thumb_cmc = landmarks_pixel[1]   # Base of thumb (vertex)
        thumb_tip = landmarks_pixel[4]   # Thumb tip
        
        # Create vectors
        # Vector from thumb base toward index (palm direction)
        v1 = index_mcp - thumb_cmc
        # Vector from thumb base to thumb tip
        v2 = thumb_tip - thumb_cmc
        
        # Calculate angle
        angle = self._calculate_angle_between_vectors(v1, v2)
        
        return angle

    def _calculate_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return float(np.degrees(angle))
    
    def _calculate_thumb_index_distance(self, hand_data: HandData) -> float:
        """Calculate normalized distance between thumb and index finger tips."""
        landmarks = hand_data.landmarks
        thumb_tip = landmarks[4][:2]
        index_tip = landmarks[8][:2]
        distance = np.linalg.norm(thumb_tip - index_tip)
        return float(distance)
    
    def _calculate_palm_to_fingertip_distances(self, hand_data: HandData) -> Dict[str, float]:
        """Calculate distance from wrist to each fingertip."""
        landmarks = hand_data.landmarks
        palm_pos = landmarks[0][:2]
        
        distances = {}
        for finger_name, tip_idx in self.finger_tips.items():
            tip_pos = landmarks[tip_idx][:2]
            distance = np.linalg.norm(tip_pos - palm_pos)
            distances[finger_name] = float(distance)
        
        return distances
    
    def _calculate_fingertips_to_palm_center(self, hand_data: HandData) -> Dict[str, float]:
        """Calculate distance from palm center to each fingertip (for fist detection)."""
        landmarks_pixel = hand_data.landmarks_pixel
        palm_center = hand_data.palm_center
        
        if palm_center is None:
            return {name: 0.0 for name in self.finger_tips.keys()}
        
        distances = {}
        palm_array = np.array(palm_center)
        
        for finger_name, tip_idx in self.finger_tips.items():
            tip_pos = landmarks_pixel[tip_idx]
            distance = np.linalg.norm(tip_pos - palm_array)
            distances[finger_name] = float(distance)
        
        return distances
    
    def _calculate_palm_compactness(self, hand_data: HandData) -> float:
        """
        Calculate how compact the hand is.
        Low value = fist (fingertips close to palm)
        High value = open hand (fingertips far from palm)
        """
        fingertip_distances = self._calculate_fingertips_to_palm_center(hand_data)
        
        # Average distance (normalized by palm size)
        avg_distance = np.mean(list(fingertip_distances.values()))
        palm_size = self._estimate_palm_size(hand_data)
        
        # Normalize
        if palm_size > 0:
            compactness = avg_distance / np.sqrt(palm_size)
        else:
            compactness = 0.0
        
        return float(compactness)
    
    def _calculate_finger_angles(self, hand_data: HandData) -> Dict[str, float]:
        """Calculate bend angle for each finger."""
        landmarks = hand_data.landmarks
        angles = {}
        
        for finger_name in ['index', 'middle', 'ring', 'pinky']:
            tip_idx = self.finger_tips[finger_name]
            pip_idx = self.finger_pips[finger_name]
            mcp_idx = self.finger_mcps[finger_name]
            
            tip = landmarks[tip_idx][:2]
            pip = landmarks[pip_idx][:2]
            mcp = landmarks[mcp_idx][:2]
            
            v1 = mcp - pip
            v2 = tip - pip
            angle = self._calculate_angle_between_vectors(v1, v2)
            angles[finger_name] = angle
        
        return angles
    
    def _calculate_hand_orientation(self, hand_data: HandData) -> float:
        """Calculate overall hand orientation angle."""
        landmarks = hand_data.landmarks
        wrist = landmarks[0][:2]
        middle_mcp = landmarks[9][:2]
        
        vector = middle_mcp - wrist
        angle = np.arctan2(vector[1], vector[0])
        
        return float(np.degrees(angle))
    
    def _estimate_palm_size(self, hand_data: HandData) -> float:
        """Estimate palm size."""
        landmarks = hand_data.landmarks
        wrist = landmarks[0][:2]
        middle_mcp = landmarks[9][:2]
        
        palm_length = np.linalg.norm(middle_mcp - wrist)
        palm_area = palm_length ** 2
        
        return float(palm_area)


def test_landmark_parser():
    """Test the landmark parser."""
    print("=== Landmark Parser Test ===\n")
    
    from vision_core.camera_manager import CameraManager
    from vision_core.frame_preprocessing import FramePreprocessor
    from vision_core.hand_tracking import HandTracker
    import cv2
    
    camera = CameraManager()
    preprocessor = FramePreprocessor()
    tracker = HandTracker(max_hands=1)
    parser = LandmarkParser()
    
    if not camera.start():
        print("✗ Failed to start camera")
        return
    
    print("✓ All initialized")
    print("\nPress 'q' to quit\n")
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            
            frame_rgb = preprocessor.process(frame)
            hands_data = tracker.detect(frame_rgb)
            
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            if hands_data:
                hand = hands_data[0]
                features = parser.parse(hand)
                
                annotated = tracker.draw_landmarks(frame_bgr, [hand])
                
                y = 30
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                finger_names = ['T', 'I', 'M', 'R', 'P']
                finger_status = ''.join([name if ext else '-' 
                                        for name, ext in zip(finger_names, features.fingers_extended)])
                
                info_lines = [
                    f"Fingers: {finger_status} ({features.fingers_count})",
                    f"Thumb Angle: {features.thumb_angle:.1f}°",
                    f"Thumb-Index: {features.thumb_index_distance:.3f}",
                    f"Compactness: {features.palm_compactness:.3f}",
                ]
                
                for line in info_lines:
                    cv2.putText(annotated, line, (10, y), font, 0.6, (0, 255, 0), 2)
                    y += 30
                
                cv2.imshow('Landmark Parser Test', annotated)
            else:
                cv2.imshow('Landmark Parser Test', frame_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        tracker.close()
        camera.stop()
        cv2.destroyAllWindows()
    
    print("\n✓ Test completed")


if __name__ == "__main__":
    test_landmark_parser()