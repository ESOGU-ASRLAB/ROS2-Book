from setuptools import find_packages, setup

package_name = 'book_pymoveit2'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'LICENSE']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cem Süha Yılmaz',
    maintainer_email='cshyilmaz@gmail.com',
    description='Bölüm 7-10 için ortak MoveIt 2 Python arayüzü',
    license='BSD-3-Clause',
    tests_require=['pytest'],
)
