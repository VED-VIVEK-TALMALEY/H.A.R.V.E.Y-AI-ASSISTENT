"""
JARVIS Vision Module - Frame Preprocessing
Day 1: Image Enhancement Pipeline

Handles frame preprocessing, normalization, and optimization.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class FramePreprocessor:
    """
    Preprocesses camera frames for optimal hand tracking.
    
    Features:
    - RGB conversion
    - Brightness normalization
    - Noise reduction
    - Contrast enhancement
    - Resolution management
    """
    
    def __init__(self, 
                 target_brightness: int = 128,
                 enable_denoising: bool = True,
                 enhance_contrast: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            target_brightness: Target average brightness (0-255)
            enable_denoising: Whether to apply noise reduction
            enhance_contrast: Whether to enhance contrast
        """
        self.target_brightness = target_brightness
        self.enable_denoising = enable_denoising
        self.enhance_contrast = enhance_contrast
        
        # Adaptive brightness parameters
        self.brightness_alpha = 0.1
        self.current_brightness = target_brightness
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Processed frame (RGB format)
        """
        if frame is None:
            return None
            
        # Step 1: Convert BGR to RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Step 2: Brightness normalization
        if self.target_brightness > 0:
            frame_rgb = self._normalize_brightness(frame_rgb)
        
        # Step 3: Denoise (optional)
        if self.enable_denoising:
            frame_rgb = self._reduce_noise(frame_rgb)
        
        # Step 4: Contrast enhancement (optional)
        if self.enhance_contrast:
            frame_rgb = self._enhance_contrast(frame_rgb)
        
        return frame_rgb
        
    def _normalize_brightness(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame brightness to target level.
        
        Args:
            frame: Input RGB frame
            
        Returns:
            Brightness-normalized frame
        """
        # Calculate current brightness
        current = np.mean(frame)
        
        # Smooth the brightness value
        self.current_brightness = (
            self.brightness_alpha * current + 
            (1 - self.brightness_alpha) * self.current_brightness
        )
        
        # Calculate adjustment factor
        if self.current_brightness > 0:
            factor = self.target_brightness / self.current_brightness
            factor = np.clip(factor, 0.7, 1.5)
        else:
            factor = 1.0
        
        # Apply adjustment
        adjusted = frame * factor
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
        
    def _reduce_noise(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction filter.
        
        Args:
            frame: Input RGB frame
            
        Returns:
            Denoised frame
        """
        # Use bilateral filter - preserves edges while reducing noise
        denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)
        return denoised
        
    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame contrast using CLAHE.
        
        Args:
            frame: Input RGB frame
            
        Returns:
            Contrast-enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb


# Test function
def test_preprocessor():
    """Test the frame preprocessor with live camera."""
    import sys
    sys.path.append('..')
    from vision_core.camera_manager import CameraManager
    
    print("=== Frame Preprocessor Test ===\n")
    
    camera = CameraManager()
    preprocessor = FramePreprocessor(
        target_brightness=140,
        enable_denoising=True,
        enhance_contrast=True
    )
    
    if not camera.start():
        print("✗ Failed to start camera")
        return
    
    print("✓ Camera started")
    print("✓ Preprocessor initialized")
    print("\nPress 'q' to quit")
    print("Press 'c' to toggle comparison view\n")
    
    show_comparison = False
    
    try:
        while True:
            frame = camera.get_frame()
            
            if frame is not None:
                # Process frame
                processed = preprocessor.process(frame)
                
                # Display
                if show_comparison:
                    # Side-by-side comparison
                    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    
                    # Add labels
                    cv2.putText(frame, "Raw", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(processed_bgr, "Processed", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Combine horizontally
                    display = np.hstack([frame, processed_bgr])
                else:
                    # Just processed frame
                    display = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    
                    # Add info
                    brightness = np.mean(frame)
                    fps_text = f"FPS: {camera.get_fps()}"
                    bright_text = f"Brightness: {brightness:.0f}"
                    
                    cv2.putText(display, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, bright_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('JARVIS Preprocessor Test', display)
                
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_comparison = not show_comparison
                print(f"Comparison view: {'ON' if show_comparison else 'OFF'}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        
    print("\n✓ Test completed")


if __name__ == "__main__":
    test_preprocessor()