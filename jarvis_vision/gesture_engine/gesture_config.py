"""
JARVIS Vision Module - Gesture Configuration
Tweakable parameters for gesture recognition accuracy
"""

class GestureConfig:
    """
    Central configuration for all gesture recognition parameters.
    """
    
    # ==========================================
    # FINGER DETECTION THRESHOLDS
    # ==========================================
    
    FINGER_EXTENSION_RATIO = 0.85
    THUMB_HORIZONTAL_THRESHOLD = 0.6
    THUMB_LENGTH_THRESHOLD = 1.3
    THUMB_ANGLE_EXTENDED_MIN = 20  # NEW: Thumb must be at this angle to be "extended"
    FINGER_ALIGNMENT_RATIO = 1.15
    
    # ==========================================
    # PINCH DETECTION (More strict to avoid confusion with fist)
    # ==========================================
    
    PINCH_THRESHOLD = 0.10           # Tighter threshold
    PINCH_MAJOR_RATIO = 0.6          # Very tight pinch
    PINCH_MAX_OTHER_FINGERS = 2   # Max other fingers that can be up during pinch
    
    # ==========================================
    # OK SIGN DETECTION
    # ==========================================
    
    OK_SIGN_THUMB_INDEX_MAX = 0.10   # Thumb-index must be close
    OK_SIGN_OTHER_FINGERS_MIN = 2    # At least 2 other fingers must be up
    
    # ==========================================
    # FIST DETECTION (Strict - all fingers must be down)
    # ==========================================
    
    FIST_MAX_FINGERS_UP = 0          # Absolutely no fingers up for fist
    FIST_PALM_COMPACTNESS = 0.15     # All fingertips close to palm
    
    # ==========================================
    # GESTURE STABILITY
    # ==========================================
    
    GESTURE_HOLD_FRAMES = 4          # Increased for more stability
    STABILITY_CONFIDENCE_BOOST = 0.1
    MIN_CONFIDENCE = 0.75
    
    # ==========================================
    # SWIPE DETECTION
    # ==========================================
    
    SWIPE_DISTANCE_THRESHOLD = 100   # Increased for deliberate swipes only
    SWIPE_HISTORY_FRAMES = 12
    SWIPE_COOLDOWN_FRAMES = 25
    SWIPE_ANALYSIS_FRAMES = 8
    SWIPE_MIN_SPEED = 15             # NEW: Minimum pixels per frame
    
    # ==========================================
    # GESTURE-SPECIFIC CONFIDENCE
    # ==========================================
    
    CONFIDENCE_FIST = 0.98
    CONFIDENCE_OPEN_PALM_5 = 0.95
    CONFIDENCE_OPEN_PALM_4 = 0.85
    CONFIDENCE_PEACE = 0.95
    CONFIDENCE_THUMBS_UP = 0.92
    CONFIDENCE_POINTING = 0.92
    CONFIDENCE_OK_SIGN = 0.88
    CONFIDENCE_PINCH_MAJOR = 0.96
    CONFIDENCE_PINCH = 0.92
    CONFIDENCE_SWIPE_MIN = 0.75
    
    @classmethod
    def print_current_config(cls):
        """Print current configuration."""
        print("\n" + "="*60)
        print("CURRENT GESTURE CONFIGURATION")
        print("="*60)
        print(f"Finger Extension Ratio: {cls.FINGER_EXTENSION_RATIO}")
        print(f"Thumb Angle Min: {cls.THUMB_ANGLE_EXTENDED_MIN}°")
        print(f"Pinch Threshold: {cls.PINCH_THRESHOLD}")
        print(f"Gesture Hold Frames: {cls.GESTURE_HOLD_FRAMES}")
        print(f"Min Confidence: {cls.MIN_CONFIDENCE}")
        print(f"Swipe Distance: {cls.SWIPE_DISTANCE_THRESHOLD}px")
        print(f"Swipe Min Speed: {cls.SWIPE_MIN_SPEED}px/frame")
        print("="*60 + "\n")