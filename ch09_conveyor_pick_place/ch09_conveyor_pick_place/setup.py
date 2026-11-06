from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ch09_conveyor_pick_place'

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
        # Temel parametreler + sim/gerçek kip katmanları
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    # google-genai (Gemini SDK) rosdep'te yoktur; pip ile kurulur (bkz. README).
    # Yalnızca model çağrısı anında, tembel olarak içe aktarılır.
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cem Süha Yılmaz',
    maintainer_email='cshyilmaz@gmail.com',
    description='ROS 2 Kitabı Bölüm 9: Gemini Robotics ER 2 ile konveyörden al-bırak',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = ch09_conveyor_pick_place.perception_node:main',
            'pick_place_node = ch09_conveyor_pick_place.pick_place_node:main',
            'recorder_node = ch09_conveyor_pick_place.recorder_node:main',
            'er_probe = ch09_conveyor_pick_place.er_probe:main',
            'measure_tcp = ch09_conveyor_pick_place.tool_geometry:main',
        ],
    },
)
