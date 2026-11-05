from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ch04_ros2_communication'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Başlatma dosyaları, robot modelleri, YAML ve RViz ayarları
        # kurulum dizinine kopyalanır; böylece get_package_share_directory
        # ile her yerden bulunabilirler.
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cem Süha Yılmaz',
    maintainer_email='cshyilmaz@gmail.com',
    description='ROS 2 Kitabı Bölüm 4 örnekleri',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'add_two_ints_server = '
            'ch04_ros2_communication.add_two_ints_server:main',
            'add_two_ints_client = '
            'ch04_ros2_communication.add_two_ints_client:main',
            'trigger_server = ch04_ros2_communication.trigger_server:main',
            'rotate_client = ch04_ros2_communication.rotate_client:main',
            'parameter_demo = ch04_ros2_communication.parameter_demo:main',
            'point_transformer = '
            'ch04_ros2_communication.point_transformer:main',
            'tcp_tf2_monitor = ch04_ros2_communication.tcp_tf2_monitor:main',
        ],
    },
)
