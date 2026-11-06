from setuptools import find_packages, setup

package_name = 'ch08_door_inspection_ui'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    # Flask şablonları ve statik dosyalar modülün yanında durur; Flask onları
    # modül klasörüne göre bulur. Bu yüzden veri dosyası olarak değil paket verisi
    # olarak kurulurlar.
    package_data={
        package_name: ['templates/*.html', 'static/css/*.css', 'static/js/*.js'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Cem Süha Yılmaz',
    maintainer_email='cshyilmaz@gmail.com',
    description='ROS 2 Kitabı Bölüm 8: kapı görevinin operatör paneli (Flask + SocketIO)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'doors_dashboard = ch08_door_inspection_ui.app:main',
        ],
    },
)
