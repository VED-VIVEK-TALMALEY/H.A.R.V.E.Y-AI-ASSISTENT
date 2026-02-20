"""
JARVIS Vision Module - Gesture Rules (CORRECTED)
Day 2: Gesture Classification with improved accuracy
"""

import numpy as np
from typing import Optional, List
from enum import Enum
from dataclasses import dataclass
import sys
sys.path.append('..')
from gesture_engine.landmark_parser import HandFeatures
from gesture_engine.gesture_config import GestureConfig as cfg


class GestureType(Enum):
    """Supported gesture types."""
    UNKNOWN = "unknown"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    POINTING = "pointing"
    OK_SIGN = "ok_sign"
    PINCH = "pinch"
    PINCH_MAJOR = "pinch_major"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"


@dataclass
class GestureResult:
    """Result of gesture recognition."""
    gesture: GestureType
    confidence: float
    features: HandFeatures
    description: str


class GestureRules:
    """Rule-based gesture recognition with improved accuracy."""
    
    def __init__(self):
        """Initialize gesture rules."""
        self.pinch_threshold = cfg.PINCH_THRESHOLD
        self.confidence_threshold = cfg.MIN_CONFIDENCE
        
        # Swipe tracking
        self.previous_palm_positions: List[tuple] = []
        self.max_history = cfg.SWIPE_HISTORY_FRAMES
        self.swipe_distance_threshold = cfg.SWIPE_DISTANCE_THRESHOLD
        
        # Gesture stability
        self.last_gesture = GestureType.UNKNOWN
        self.gesture_hold_count = 0
        self.hold_threshold = cfg.GESTURE_HOLD_FRAMES
        
        # Swipe cooldown
        self.last_swipe_frame = 0
        self.swipe_cooldown = cfg.SWIPE_COOLDOWN_FRAMES
        self.current_frame = 0
    
    def recognize(self, features: HandFeatures) -> GestureResult:
        """Recognize gesture from hand features."""
        self.current_frame += 1
        self._update_movement_history(features.palm_center)
        
        # Priority 1: Check swipe (moving gestures)
        swipe_result = self._check_swipe(features)
        if swipe_result:
            return swipe_result
        
        # Priority 2: Pinch (very specific)
        result = self._check_pinch(features)
        if result and result.confidence >= self.confidence_threshold:
            return self._stabilize_gesture(result)
        
        # Priority 3: Fist (strict - no fingers up)
        result = self._check_fist(features)
        if result:
            return self._stabilize_gesture(result)
        
        # Priority 4: Specific finger patterns
        result = self._check_peace(features)
        if result:
            return self._stabilize_gesture(result)
        
        result = self._check_pointing(features)
        if result:
            return self._stabilize_gesture(result)
        
        result = self._check_thumbs_up(features)
        if result:
            return self._stabilize_gesture(result)
        
        # Priority 5: OK sign (thumb + index close, others up)
        result = self._check_ok_sign(features)
        if result:
            return self._stabilize_gesture(result)
        
        # Priority 6: Open palm
        result = self._check_open_palm(features)
        if result:
            return self._stabilize_gesture(result)
        
        return GestureResult(
            gesture=GestureType.UNKNOWN,
            confidence=0.0,
            features=features,
            description="No gesture"
        )
    
    def _stabilize_gesture(self, result: GestureResult) -> GestureResult:
        """Stabilize gesture detection."""
        if result.gesture == self.last_gesture:
            self.gesture_hold_count += 1
        else:
            self.gesture_hold_count = 0
            self.last_gesture = result.gesture
        
        if self.gesture_hold_count >= self.hold_threshold:
            result.confidence = min(1.0, result.confidence + cfg.STABILITY_CONFIDENCE_BOOST)
        
        return result
    
    def _check_fist(self, features: HandFeatures) -> Optional[GestureResult]:
        """
        Check for fist - STRICT version.
        Must have ALL fingers down AND hand must be compact.
        """
        # Absolutely no fingers extended
        if features.fingers_count != 0:
            return None
        
        # Hand must be compact (fingertips close to palm)
        if features.palm_compactness > cfg.FIST_PALM_COMPACTNESS:
            return None
        
        # Additional check: thumb angle should be small (thumb tucked in)
        if features.thumb_angle > 100:  # Thumb too extended
            return None
        
        return GestureResult(
            gesture=GestureType.FIST,
            confidence=cfg.CONFIDENCE_FIST,
            features=features,
            description="Fist ✊"
        )
    
    def _check_open_palm(self, features: HandFeatures) -> Optional[GestureResult]:
        """Check if hand is open palm."""
        if features.fingers_count >= 4:
            confidence = cfg.CONFIDENCE_OPEN_PALM_5 if features.fingers_count == 5 else cfg.CONFIDENCE_OPEN_PALM_4
            return GestureResult(
                gesture=GestureType.OPEN_PALM,
                confidence=confidence,
                features=features,
                description="Open Palm ✋"
            )
        return None
    
    def _check_peace(self, features: HandFeatures) -> Optional[GestureResult]:
        """Check for peace sign."""
        fingers = features.fingers_extended
        
        if (fingers[1] and fingers[2] and
            not fingers[0] and not fingers[3] and not fingers[4]):
            return GestureResult(
                gesture=GestureType.PEACE,
                confidence=cfg.CONFIDENCE_PEACE,
                features=features,
                description="Peace ✌️"
            )
        return None
    
    def _check_thumbs_up(self, features: HandFeatures) -> Optional[GestureResult]:
        """Check for thumbs up."""
        fingers = features.fingers_extended
        
        if fingers[0] and not any(fingers[1:]):
            angle = abs(features.hand_orientation)
            if angle < 45 or angle > 135:
                return GestureResult(
                    gesture=GestureType.THUMBS_UP,
                    confidence=cfg.CONFIDENCE_THUMBS_UP,
                    features=features,
                    description="Thumbs Up 👍"
                )
        return None
    
    def _check_pointing(self, features: HandFeatures) -> Optional[GestureResult]:
        """Check for pointing gesture."""
        fingers = features.fingers_extended
        
        if (fingers[1] and
            not fingers[0] and not fingers[2] and 
            not fingers[3] and not fingers[4]):
            return GestureResult(
                gesture=GestureType.POINTING,
                confidence=cfg.CONFIDENCE_POINTING,
                features=features,
                description="Pointing 👉"
            )
        return None
    
    def _check_ok_sign(self, features: HandFeatures) -> Optional[GestureResult]:
        """
        Check for OK sign - IMPROVED.
        Thumb and index form circle, other fingers extended.
        """
        fingers = features.fingers_extended
        distance = features.thumb_index_distance
        
        # Thumb and index must be close (forming circle)
        if distance > cfg.OK_SIGN_THUMB_INDEX_MAX:
            return None
        
        # Count other extended fingers (middle, ring, pinky)
        other_fingers_up = sum(fingers[2:5])
        
        # At least 2 of the other 3 fingers must be up
        if other_fingers_up >= cfg.OK_SIGN_OTHER_FINGERS_MIN:
            return GestureResult(
                gesture=GestureType.OK_SIGN,
                confidence=cfg.CONFIDENCE_OK_SIGN,
                features=features,
                description="OK Sign 👌"
            )
        
        return None
    
    def _check_pinch(self, features: HandFeatures) -> Optional[GestureResult]:
        """
        Check for pinch - IMPROVED to avoid confusion with fist.
        Must have thumb and index close, but other fingers can be slightly extended.
        """
        distance = features.thumb_index_distance
        fingers = features.fingers_extended
        
        # Distance check
        if distance >= self.pinch_threshold:
            return None
        
        # Count other fingers (middle, ring, pinky)
        other_fingers = sum(fingers[2:5])
        
        # Too many other fingers up = not a pinch
        if other_fingers > cfg.PINCH_MAX_OTHER_FINGERS:
            return None
        
        # Hand should NOT be as compact as a fist
        if features.palm_compactness < cfg.FIST_PALM_COMPACTNESS * 1.5:
            return None  # Too compact, probably a fist
        
        # Check thumb angle - thumb should be somewhat extended for pinch
        if features.thumb_angle < 100:  # Thumb too tucked
            return None
        
        # Determine pinch tightness
        if distance < self.pinch_threshold * cfg.PINCH_MAJOR_RATIO:
            return GestureResult(
                gesture=GestureType.PINCH_MAJOR,
                confidence=cfg.CONFIDENCE_PINCH_MAJOR,
                features=features,
                description="Tight Pinch 🤏"
            )
        else:
            return GestureResult(
                gesture=GestureType.PINCH,
                confidence=cfg.CONFIDENCE_PINCH,
                features=features,
                description="Pinch 🤏"
            )
    
    def _check_swipe(self, features: HandFeatures) -> Optional[GestureResult]:
        """
        Check for swipe gestures - IMPROVED with speed check.
        """
        if len(self.previous_palm_positions) < cfg.SWIPE_ANALYSIS_FRAMES:
            return None
        
        if self.current_frame - self.last_swipe_frame < self.swipe_cooldown:
            return None
        
        # Analyze recent movement
        positions = self.previous_palm_positions[-cfg.SWIPE_ANALYSIS_FRAMES:]
        start_pos = np.array(positions[0])
        end_pos = np.array(positions[-1])
        movement = end_pos - start_pos
        
        distance = np.linalg.norm(movement)
        
        # Check minimum distance
        if distance < self.swipe_distance_threshold:
            return None
        
        # Check speed (distance per frame)
        speed = distance / cfg.SWIPE_ANALYSIS_FRAMES
        if speed < cfg.SWIPE_MIN_SPEED:
            return None  # Too slow, not a deliberate swipe
        
        # Determine direction
        angle = np.arctan2(movement[1], movement[0])
        angle_deg = np.degrees(angle)
        
        if -45 <= angle_deg <= 45:
            gesture = GestureType.SWIPE_RIGHT
            desc = "Swipe Right →"
        elif 45 < angle_deg <= 135:
            gesture = GestureType.SWIPE_DOWN
            desc = "Swipe Down ↓"
        elif angle_deg > 135 or angle_deg < -135:
            gesture = GestureType.SWIPE_LEFT
            desc = "Swipe Left ←"
        else:
            gesture = GestureType.SWIPE_UP
            desc = "Swipe Up ↑"
        
        confidence = min(0.95, cfg.CONFIDENCE_SWIPE_MIN + (speed / 30))
        
        self.last_swipe_frame = self.current_frame
        self.previous_palm_positions.clear()
        
        return GestureResult(
            gesture=gesture,
            confidence=confidence,
            features=features,
            description=desc
        )
    
    def _update_movement_history(self, palm_center: tuple):
        """Update palm position history."""
        self.previous_palm_positions.append(palm_center)
        
        if len(self.previous_palm_positions) > self.max_history:
            self.previous_palm_positions.pop(0)
    
    def reset_movement_history(self):
        """Reset movement tracking."""
        self.previous_palm_positions.clear()
        self.last_gesture = GestureType.UNKNOWN
        self.gesture_hold_count = 0


def test_gesture_rules():
    """Test gesture recognition."""
    print("=== Gesture Rules Test (FINAL CORRECTED) ===\n")
    
    from vision_core.camera_manager import CameraManager
    from vision_core.frame_preprocessing import FramePreprocessor
    from vision_core.hand_tracking import HandTracker
    from gesture_engine.landmark_parser import LandmarkParser
    import cv2
    
    camera = CameraManager()
    preprocessor = FramePreprocessor()
    tracker = HandTracker(max_hands=1)
    parser = LandmarkParser()
    gesture_rules = GestureRules()
    
    if not camera.start():
        print("✗ Failed to start camera")
        return
    
    cfg.print_current_config()
    
    print("✓ All components initialized")
    print("\n🎯 GESTURES TO TRY:")
    print("  ✊ Fist - ALL fingers closed, compact hand")
    print("  ✋ Open Palm - All fingers extended")
    print("  ✌️ Peace - Index + middle ONLY")
    print("  👍 Thumbs Up - Thumb ONLY")
    print("  👉 Pointing - Index ONLY")
    print("  👌 OK Sign - Thumb+index touching, 2+ others up")
    print("  🤏 Pinch - Thumb+index close, thumb extended")
    print("  ← → ↑ ↓ Swipes - Move hand FAST (15+ px/frame)")
    print("\n💡 Tips:")
    print("  - Hold gestures steady for 4 frames")
    print("  - Fist: Make hand very compact")
    print("  - Pinch: Extend thumb slightly")
    print("  - OK: Keep other fingers clearly up")
    print("  - Swipes: Move quickly and deliberately")
    print("\nPress 'q' to quit\n")
    
    gesture_display_frames = 0
    current_display = None
    
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
                result = gesture_rules.recognize(features)
                
                annotated = tracker.draw_landmarks(frame_bgr, [hand])
                
                if result.gesture != GestureType.UNKNOWN:
                    current_display = result
                    gesture_display_frames = 60
                
                if gesture_display_frames > 0 and current_display:
                    gesture_display_frames -= 1
                    
                    cv2.rectangle(annotated, (5, 45), (550, 130), (0, 0, 0), -1)
                    cv2.rectangle(annotated, (5, 45), (550, 130), (0, 255, 0), 2)
                    
                    cv2.putText(annotated, current_display.description, (15, 85), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
                    
                    conf_text = f"Confidence: {current_display.confidence:.0%}"
                    cv2.putText(annotated, conf_text, (15, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.putText(annotated, f"FPS: {camera.get_fps():.0f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Debug info
                y = 160
                debug_info = [
                    f"Fingers: {''.join(['TIMRP'[i] if features.fingers_extended[i] else '-' for i in range(5)])}",
                    f"Thumb∠: {features.thumb_angle:.0f}°",
                    f"T-I: {features.thumb_index_distance:.3f}",
                    f"Compact: {features.palm_compactness:.3f}",
                ]
                
                for info in debug_info:
                    cv2.putText(annotated, info, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y += 20
                
                cv2.imshow('Gesture Recognition', annotated)
            else:
                cv2.imshow('Gesture Recognition', frame_bgr)
            
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
    test_gesture_rules()