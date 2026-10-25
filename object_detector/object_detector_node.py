#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
from datetime import datetime

class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')
        
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('dataset_path', '~/yolo_sample_dataset')
        self.declare_parameter('output_dir', '~/detection_outputs')
        
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        image_topic = self.get_parameter('image_topic').value
        self.dataset_path = self.get_parameter('dataset_path').value
        self.output_dir = self.get_parameter('output_dir').value
        
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        
        # Create output directory
        self.output_path = Path(os.path.expanduser(self.output_dir))
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f'Output directory: {self.output_path}')
        
        # Create window with resizable capability
        self.window_name = 'Object Detection - ROS2'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 760)
        
        self.use_dataset = False
        self.dataset_images = []
        self.current_idx = 0
        self.is_fullscreen = False
        
        # Store current frames for saving
        self.current_original = None
        self.current_detected = None
        self.current_dual_view = None
        
        if self.dataset_path:
            self.load_dataset()
        
        self.get_logger().info('Object Detector Node initialized')
        
    def load_dataset(self):
        dataset_path = Path(os.path.expanduser(self.dataset_path))
        
        if dataset_path.is_dir():
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
            for ext in extensions:
                self.dataset_images.extend(list(dataset_path.glob(f'*{ext}')))
            
            if self.dataset_images:
                self.use_dataset = True
                self.get_logger().info(f'Loaded {len(self.dataset_images)} images from dataset')
            else:
                self.get_logger().warn(f'No images found in {dataset_path}')
        elif dataset_path.is_file():
            self.dataset_images = [dataset_path]
            self.use_dataset = True
            self.get_logger().info(f'Loaded single image: {dataset_path}')
    
    def process_frame(self, frame):
        # Store original frame
        self.current_original = frame.copy()
        
        # Detect
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detected = results[0].plot()
        
        # Store detected frame
        self.current_detected = detected.copy()
        
        # Display
        self.display_dual_layout(frame, detected)
    
    def display_dual_layout(self, original, detected):
        """Create a responsive dual-view layout that adapts to window size"""
        
        # Reserve space for labels and separator
        label_height = 40
        separator_width = 10
        
        # Get original image dimensions
        img_h, img_w = original.shape[:2]
        
        # Calculate optimal window size to fit screen better when maximized
        max_display_width = 1920
        max_display_height = 1080
        
        # Calculate scale to fit dual images side by side
        total_width_needed = (img_w * 2) + separator_width
        total_height_needed = img_h + label_height
        
        scale_w = (max_display_width * 0.95) / total_width_needed
        scale_h = (max_display_height * 0.90) / total_height_needed
        scale = min(scale_w, scale_h, 1.0)
        
        # Calculate final dimensions
        new_width = int(img_w * scale)
        new_height = int(img_h * scale)
        
        # Resize images
        frame_resized = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_AREA)
        detected_resized = cv2.resize(detected, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create containers for each side
        container_height = new_height + label_height
        container_width = new_width
        
        labeled_orig = np.zeros((container_height, container_width, 3), dtype=np.uint8)
        labeled_det = np.zeros((container_height, container_width, 3), dtype=np.uint8)
        
        # Set background color
        labeled_orig[:] = (40, 40, 40)
        labeled_det[:] = (40, 40, 40)
        
        # Add label backgrounds
        labeled_orig[:label_height] = (60, 60, 60)
        labeled_det[:label_height] = (60, 60, 60)
        
        # Calculate font scale based on image size
        font_scale = max(0.5, min(new_width / 800, 1.0))
        font_thickness = max(1, int(font_scale * 2))
        
        # Add labels
        cv2.putText(labeled_orig, 'ORIGINAL', (10, int(label_height * 0.7)), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), font_thickness)
        cv2.putText(labeled_det, 'DETECTED', (10, int(label_height * 0.7)), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 0), font_thickness)
        
        # Add image info text
        info_text = f'Image: {self.current_idx + 1}/{len(self.dataset_images)}' if self.use_dataset else ''
        if info_text:
            cv2.putText(labeled_orig, info_text, 
                       (container_width - 150, int(label_height * 0.7)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (200, 200, 200), 1)
        
        # Place images directly below labels
        labeled_orig[label_height:label_height+new_height, :] = frame_resized
        labeled_det[label_height:label_height+new_height, :] = detected_resized
        
        # Create separator
        separator = np.ones((container_height, separator_width, 3), dtype=np.uint8) * 255
        
        # Combine horizontally
        dual_view = np.hstack([labeled_orig, separator, labeled_det])
        
        # Store current dual view for saving
        self.current_dual_view = dual_view.copy()
        
        # Show the combined view
        cv2.imshow(self.window_name, dual_view)
    
    def save_current_output(self):
        """Save the current detection output with timestamp"""
        if self.current_detected is None:
            self.get_logger().warn('No image to save!')
            return
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get base filename from current image if available
        if self.use_dataset and self.dataset_images:
            base_name = Path(self.dataset_images[self.current_idx]).stem
            filename_prefix = f"{base_name}_{timestamp}"
        else:
            filename_prefix = f"detection_{timestamp}"
        
        # Save detected image
        detected_path = self.output_path / f"{filename_prefix}_detected.jpg"
        cv2.imwrite(str(detected_path), self.current_detected)
        
        # Save dual view (side-by-side comparison)
        dual_path = self.output_path / f"{filename_prefix}_comparison.jpg"
        cv2.imwrite(str(dual_path), self.current_dual_view)
        
        # Optionally save original
        original_path = self.output_path / f"{filename_prefix}_original.jpg"
        cv2.imwrite(str(original_path), self.current_original)
        
        self.get_logger().info(f'Saved outputs to:')
        self.get_logger().info(f'  - {detected_path.name}')
        self.get_logger().info(f'  - {dual_path.name}')
        self.get_logger().info(f'  - {original_path.name}')
    
    def show_help(self):
        """Display help information"""
        help_text = [
            "Keyboard Controls:",
            "  q - Quit",
            "  n - Next image",
            "  p - Previous image", 
            "  s - Save current output (Ctrl+S also works)",
            "  f - Toggle fullscreen",
            "  w - Exit fullscreen (windowed mode)",
            "  h - Show this help"
        ]
        for line in help_text:
            self.get_logger().info(line)
    
    def process_dataset_image(self):
        if self.use_dataset and self.dataset_images:
            img_path = str(self.dataset_images[self.current_idx])
            frame = cv2.imread(img_path)
            if frame is not None:
                self.get_logger().info(f'Processing [{self.current_idx+1}/{len(self.dataset_images)}]: {os.path.basename(img_path)}')
                self.process_frame(frame)
            else:
                self.get_logger().error(f'Could not read image: {img_path}')
    
    def run_dataset_mode(self):
        if not self.use_dataset:
            self.get_logger().error('No dataset loaded!')
            return
        
        self.show_help()
        
        # Process first image
        self.process_dataset_image()
        
        # Keep processing with key events
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)
            
            # Check for key presses
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                self.get_logger().info('Quit requested')
                break
            elif key == ord('n'):
                self.current_idx = (self.current_idx + 1) % len(self.dataset_images)
                self.get_logger().info(f'Next image: {self.current_idx + 1}/{len(self.dataset_images)}')
                self.process_dataset_image()
            elif key == ord('p'):
                self.current_idx = (self.current_idx - 1) % len(self.dataset_images)
                self.get_logger().info(f'Previous image: {self.current_idx + 1}/{len(self.dataset_images)}')
                self.process_dataset_image()
            elif key == ord('s') or key == 19:  # 's' or Ctrl+S (ASCII 19)
                self.save_current_output()
            elif key == ord('f'):
                # Toggle fullscreen
                if not self.is_fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    self.is_fullscreen = True
                    self.get_logger().info('Fullscreen mode enabled')
                else:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    self.is_fullscreen = False
                    self.get_logger().info('Windowed mode enabled')
                # Refresh display after mode change
                self.process_dataset_image()
            elif key == ord('w'):
                # Exit fullscreen
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                self.is_fullscreen = False
                self.get_logger().info('Windowed mode enabled')
                # Refresh display after mode change
                self.process_dataset_image()
            elif key == ord('h'):
                self.show_help()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    
    if node.use_dataset:
        try:
            node.run_dataset_mode()
        except KeyboardInterrupt:
            pass
    else:
        node.get_logger().info('No dataset specified. Use dataset_path parameter.')
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
    
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
