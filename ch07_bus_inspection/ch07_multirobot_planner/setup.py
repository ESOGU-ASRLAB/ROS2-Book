from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ch07_multirobot_planner'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # Kitapta kullanılan çok robotlu plan (25 UR10e + 11 Kawasaki bakış noktası)
        (os.path.join('share', package_name, 'plans'),
            glob('plans/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cem Süha Yılmaz',
    maintainer_email='cshyilmaz@gmail.com',
    description='ROS 2 Kitabı Bölüm 7: çok robotlu bakış noktası planlaması ve muayene',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'multirobot_planner_node = '
            'ch07_multirobot_planner.multirobot_planner_node:main',
            'ur_inspection_node = ch07_multirobot_planner.ur_inspection_node:main',
            'kawasaki_inspection_node = '
            'ch07_multirobot_planner.kawasaki_inspection_node:main',
            'multirobot_viewpoint_visualizer = '
            'ch07_multirobot_planner.visualization:main',
            'multirobot_plan_visualizer = '
            'ch07_multirobot_planner.plan_visualizer:main',
        ],
    },
)
