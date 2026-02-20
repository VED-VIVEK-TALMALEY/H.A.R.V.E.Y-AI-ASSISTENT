"""
Debug pinch detection
"""
import sys
sys.path.append('..')
from vision_core.camera_manager import CameraManager
from vision_core.frame_preprocessing import FramePreprocessor
from vision_core.hand_tracking import HandTracker
from gesture_engine.landmark_parser import LandmarkParser
from gesture_engine.gesture_config import GestureConfig as cfg
import cv2

camera = CameraManager()
preprocessor = FramePreprocessor()
tracker = HandTracker(max_hands=1)
parser = LandmarkParser()

camera.start()

print("=== PINCH DEBUG MODE ===")
print(f"Current pinch threshold: {cfg.PINCH_THRESHOLD}")
print("\nMake a pinch gesture and watch the values")
print("Press 'q' to quit\n")

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
            
            # Get pinch-related values
            distance = features.thumb_index_distance
            thumb_angle = features.thumb_angle
            compactness = features.palm_compactness
            fingers = features.fingers_extended
            other_fingers = sum(fingers[2:5])
            
            # Determine if it would be detected as pinch
            is_pinch_distance = distance < cfg.PINCH_THRESHOLD
            is_thumb_angle_ok = thumb_angle > 90
            is_not_too_compact = compactness > cfg.FIST_PALM_COMPACTNESS * 1.5
            is_not_too_many_fingers = other_fingers <= cfg.PINCH_MAX_OTHER_FINGERS
            
            # Display info
            y = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Thumb-Index Distance (KEY VALUE)
            distance_color = (0, 255, 0) if is_pinch_distance else (0, 0, 255)
            cv2.putText(annotated, f"Thumb-Index Distance: {distance:.4f}", (10, y),
                       font, 0.7, distance_color, 2)
            y += 35
            
            cv2.putText(annotated, f"Threshold: {cfg.PINCH_THRESHOLD:.4f}", (10, y),
                       font, 0.6, (255, 255, 0), 2)
            y += 30
            
            # Thumb Angle
            angle_color = (0, 255, 0) if is_thumb_angle_ok else (0, 0, 255)
            cv2.putText(annotated, f"Thumb Angle: {thumb_angle:.1f}° (need >90)", (10, y),
                       font, 0.6, angle_color, 1)
            y += 25
            
            # Compactness
            compact_color = (0, 255, 0) if is_not_too_compact else (0, 0, 255)
            cv2.putText(annotated, f"Compactness: {compactness:.3f} (need >{cfg.FIST_PALM_COMPACTNESS * 1.5:.3f})", 
                       (10, y), font, 0.6, compact_color, 1)
            y += 25
            
            # Other fingers
            fingers_color = (0, 255, 0) if is_not_too_many_fingers else (0, 0, 255)
            cv2.putText(annotated, f"Other fingers up: {other_fingers} (max {cfg.PINCH_MAX_OTHER_FINGERS})", 
                       (10, y), font, 0.6, fingers_color, 1)
            y += 30
            
            # Overall status
            all_conditions = is_pinch_distance and is_thumb_angle_ok and is_not_too_compact and is_not_too_many_fingers
            status_text = "✓ PINCH DETECTED!" if all_conditions else "✗ Not detected"
            status_color = (0, 255, 0) if all_conditions else (0, 0, 255)
            cv2.putText(annotated, status_text, (10, y), font, 1.0, status_color, 2)
            
            # Show which conditions failed
            y += 40
            if not all_conditions:
                cv2.putText(annotated, "Failed checks:", (10, y), font, 0.6, (255, 255, 255), 1)
                y += 25
                if not is_pinch_distance:
                    cv2.putText(annotated, "  - Distance too far", (10, y), font, 0.5, (0, 0, 255), 1)
                    y += 20
                if not is_thumb_angle_ok:
                    cv2.putText(annotated, "  - Thumb angle too small", (10, y), font, 0.5, (0, 0, 255), 1)
                    y += 20
                if not is_not_too_compact:
                    cv2.putText(annotated, "  - Hand too compact (fist?)", (10, y), font, 0.5, (0, 0, 255), 1)
                    y += 20
                if not is_not_too_many_fingers:
                    cv2.putText(annotated, "  - Too many other fingers up", (10, y), font, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('Pinch Debug', annotated)
        else:
            cv2.imshow('Pinch Debug', frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    pass
finally:
    tracker.close()
    camera.stop()
    cv2.destroyAllWindows()

print("\n✓ Debug complete")