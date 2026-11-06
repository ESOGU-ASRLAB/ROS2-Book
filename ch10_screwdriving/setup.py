from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ch10_screwdriving'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cem Süha Yılmaz',
    maintainer_email='cshyilmaz@gmail.com',
    description='ROS 2 Kitabı Bölüm 10: insan-robot iş birlikli vidalama',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_robot_collaboration_scenario = '
            'ch10_screwdriving.human_robot_collaboration_scenario:main',
        ],
    },
)
