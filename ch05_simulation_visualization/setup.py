from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ch05_simulation_visualization'

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
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cem Süha Yılmaz',
    maintainer_email='cshyilmaz@gmail.com',
    description='ROS 2 Kitabı Bölüm 5 örnekleri',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'send_trajectory = '
            'ch05_simulation_visualization.send_trajectory:main',
        ],
    },
)
