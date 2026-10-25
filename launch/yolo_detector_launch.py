from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value='yolov8n.pt'),
        DeclareLaunchArgument('confidence', default_value='0.5'),
        DeclareLaunchArgument('dataset_path', default_value=''),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        
        Node(
            package='yolo_detector',
            executable='yolo_detector_node',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence'),
                'dataset_path': LaunchConfiguration('dataset_path'),
                'image_topic': LaunchConfiguration('image_topic'),
            }]
        ),
    ])
