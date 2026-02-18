"""
JARVIS Vision Module - Calibration
Day 1: Camera & Sensitivity Calibration

Handles user calibration for optimal hand tracking performance.
"""

import json
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CalibrationProfile:
    """Calibration settings profile."""
    profile_name: str
    
    # Camera settings
    brightness_target: int = 140
    camera_index: int = 0
    resolution: Tuple[int, int] = (640, 480)
    
    # Hand tracking settings
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    max_hands: int = 1
    
    # Sensitivity settings
    gesture_sensitivity: float = 1.0
    volume_sensitivity: float = 1.0
    scroll_sensitivity: float = 1.0
    
    # Lighting conditions
    lighting_condition: str = "normal"
    enable_denoising: bool = True
    enhance_contrast: bool = True
    
    # Metadata
    created_at: str = ""
    last_modified: str = ""
    
    def __post_init__(self):
        """Set timestamps if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.last_modified:
            self.last_modified = datetime.now().isoformat()


class CalibrationManager:
    """
    Manages calibration profiles and settings.
    
    Features:
    - Save/load calibration profiles
    - Preset profiles
    - Sensitivity adjustment
    """
    
    def __init__(self, config_dir: str = "./config"):
        """Initialize calibration manager."""
        self.config_dir = config_dir
        self.profiles_file = os.path.join(config_dir, "calibration_profiles.json")
        self.current_profile: Optional[CalibrationProfile] = None
        
        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Load or create default profile
        self._load_or_create_default()
    
    def _load_or_create_default(self):
        """Load existing profiles or create default."""
        if os.path.exists(self.profiles_file):
            self.load_profile("default")
        else:
            self.current_profile = CalibrationProfile(profile_name="default")
            self.save_profile()
    
    def save_profile(self, profile_name: Optional[str] = None):
        """Save current calibration profile."""
        if profile_name:
            self.current_profile.profile_name = profile_name
        
        # Update timestamp
        self.current_profile.last_modified = datetime.now().isoformat()
        
        # Load existing profiles
        profiles = self._load_all_profiles()
        
        # Update or add current profile
        profiles[self.current_profile.profile_name] = asdict(self.current_profile)
        
        # Save to file
        with open(self.profiles_file, 'w') as f:
            json.dump(profiles, f, indent=2)
        
        print(f"✓ Profile '{self.current_profile.profile_name}' saved")
    
    def load_profile(self, profile_name: str) -> bool:
        """Load a calibration profile."""
        profiles = self._load_all_profiles()
        
        if profile_name not in profiles:
            print(f"✗ Profile '{profile_name}' not found")
            return False
        
        # Convert dict to CalibrationProfile
        profile_dict = profiles[profile_name]
        # Convert resolution tuple if it's stored as list
        if 'resolution' in profile_dict and isinstance(profile_dict['resolution'], list):
            profile_dict['resolution'] = tuple(profile_dict['resolution'])
        
        self.current_profile = CalibrationProfile(**profile_dict)
        print(f"✓ Profile '{profile_name}' loaded")
        return True
    
    def _load_all_profiles(self) -> Dict:
        """Load all profiles from file."""
        if not os.path.exists(self.profiles_file):
            return {}
        
        try:
            with open(self.profiles_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠ Warning: Corrupted profiles file, creating new")
            return {}
    
    def list_profiles(self) -> list:
        """Get list of available profile names."""
        profiles = self._load_all_profiles()
        return list(profiles.keys())
    
    def create_preset(self, preset_type: str):
        """Create a preset calibration profile."""
        presets = {
            "media": CalibrationProfile(
                profile_name="media_mode",
                gesture_sensitivity=1.2,
                volume_sensitivity=1.5,
                scroll_sensitivity=0.8,
                max_hands=1
            ),
            "productivity": CalibrationProfile(
                profile_name="productivity_mode",
                gesture_sensitivity=0.8,
                volume_sensitivity=1.0,
                scroll_sensitivity=1.2,
                max_hands=1
            ),
            "presentation": CalibrationProfile(
                profile_name="presentation_mode",
                gesture_sensitivity=1.5,
                volume_sensitivity=0.5,
                scroll_sensitivity=1.5,
                max_hands=1,
                min_detection_confidence=0.8
            )
        }
        
        if preset_type in presets:
            self.current_profile = presets[preset_type]
            self.save_profile()
            print(f"✓ Preset '{preset_type}' created and loaded")
        else:
            print(f"✗ Unknown preset type: {preset_type}")
    
    def print_current_profile(self):
        """Print current profile settings."""
        print(f"\n=== Calibration Profile: {self.current_profile.profile_name} ===")
        print(f"Camera: Index {self.current_profile.camera_index}, "
              f"{self.current_profile.resolution[0]}x{self.current_profile.resolution[1]}")
        print(f"Brightness Target: {self.current_profile.brightness_target}")
        print(f"Lighting: {self.current_profile.lighting_condition}")
        print(f"Detection Confidence: {self.current_profile.min_detection_confidence}")
        print(f"Gesture Sensitivity: {self.current_profile.gesture_sensitivity}")
        print(f"Volume Sensitivity: {self.current_profile.volume_sensitivity}")
        print(f"Scroll Sensitivity: {self.current_profile.scroll_sensitivity}")
        print(f"Last Modified: {self.current_profile.last_modified}")
        print("=" * 50)


# Test function
def test_calibration():
    """Test calibration manager."""
    print("=== Calibration Manager Test ===\n")
    
    manager = CalibrationManager(config_dir="../config")
    
    # Show current profile
    manager.print_current_profile()
    
    # Create presets
    print("\nCreating presets...")
    manager.create_preset("media")
    manager.create_preset("productivity")
    manager.create_preset("presentation")
    
    # List profiles
    print("\nAvailable profiles:")
    for profile in manager.list_profiles():
        print(f"  - {profile}")
    
    # Load a profile
    print("\nLoading 'media_mode' profile...")
    manager.load_profile("media_mode")
    manager.print_current_profile()
    
    print("\n✓ Test completed")


if __name__ == "__main__":
    test_calibration()