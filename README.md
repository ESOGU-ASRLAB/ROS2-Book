# ROS 2 ile Endüstriyel Robot Programlama — Kitap Kodları

Bu depo, *ROS 2 ile Endüstriyel Robot Programlama* kitabının resmî kod deposudur.
Her bölümün kodu, bölüm numarasıyla başlayan bir klasördedir (`chNN_...`).

| Klasör | Bölüm | İçerik |
|---|---|---|
| [`ch04_ros2_communication`](ch04_ros2_communication) | 4 — Gelişmiş ROS 2 İletişim Yapıları ve Araçlar | ROS 2 paketi: servis, eylem, parametre, başlatma dosyası, URDF/Xacro, TF2 |
| [`ch05_simulation_visualization`](ch05_simulation_visualization) | 5 — Simülasyon ve Görselleştirme Ortamları | ROS 2 paketi: RViz2, Gazebo Harmonic, ros_gz köprüsü, ros2_control kipleri |
| [`ch07_bus_inspection`](ch07_bus_inspection) | 7 — Otobüs Şasesinin Kalite Kontrolü İçin Hareket Planlama | Araştırma deposuna yönlendirme |
| [`ch08_door_inspection`](ch08_door_inspection) | 8 — Doblo Kapısı Yüzey Kalite Kontrolü | Araştırma deposuna yönlendirme |
| [`ch09_conveyor_pick_place`](ch09_conveyor_pick_place) | 9 — Konveyör Al-Yerleştir Otomasyonu | Araştırma deposuna yönlendirme |
| [`ch10_screwdriving`](ch10_screwdriving) | 10 — İnsan–Robot İş Birlikli Hassas Vidalama | Araştırma deposuna yönlendirme |

Bölüm 4 ve 5 kendi başına derlenip çalışan, donanım gerektirmeyen paketlerdir.
Bölüm 7–10'daki uygulamalar ise gerçek bir robot hücresine (UR10e, Kawasaki RS005L,
Festo doğrusal eksen, AGV, OnRobot ve VGC10 tutucular) ve onlarca sürücü paketine
dayandığı için kodları
[ESOGU-ASRLAB/ESOGU-HILTest-DualRobot](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot)
araştırma deposunda tutulur. Bu depodaki `ch07`–`ch10` klasörleri, bölümde anlatılan
her adımın araştırma deposundaki karşılığını gösterir.

## Gereksinimler

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Harmonic (yalnızca Bölüm 5'in Gazebo kipi için; kurulumu aşağıdaki
  **Bölüm 5: Gazebo Harmonic kurulumu** başlığındadır)

## Kurulum

```bash
$ mkdir -p ~/ros2_ws/src
$ cd ~/ros2_ws/src
$ git clone https://github.com/ESOGU-ASRLAB/ROS2-Book.git
$ cd ~/ros2_ws
$ source /opt/ros/humble/setup.bash
$ rosdep install --from-paths src --ignore-src -r -y
$ colcon build --symlink-install
$ source install/setup.bash
```

Kurulumu sınamak için Bölüm 4'ün TF2 alıştırmasını çalıştırın (iki ayrı terminal):

```bash
# 1. terminal
$ ros2 launch ch04_ros2_communication manipulator_rsp.launch.py

# 2. terminal
$ ros2 run ch04_ros2_communication tcp_tf2_monitor
[INFO] [manipulator_tcp_monitor]: Flanş (tool0) [m]: X=0.515, Y=0.000, Z=0.900 | TCP (tcp_link) [m]: X=0.660, Y=0.000, Z=0.900 (Quat w=0.71)
```

## Bölüm 7–10: araştırma deposunun eklenmesi

Uygulama bölümlerinin kodunu aynı çalışma alanına çekmek için
[`research.repos`](research.repos) dosyası kullanılır:

```bash
$ cd ~/ros2_ws
$ vcs import src < src/ROS2-Book/research.repos
$ rosdep install --from-paths src --ignore-src -r -y
$ colcon build --symlink-install
```

`vcs` komutu yoksa: `$ sudo apt install python3-vcstool`

## Bölüm 5: Gazebo Harmonic kurulumu

ROS 2 Humble'ın varsayılan Gazebo eşi Gazebo Fortress'tır; `rosdep`'in kurduğu
`ros-humble-ros-gz-*` ve `ros-humble-gz-ros2-control` ikili paketleri Fortress'a
bağlıdır. Bölüm 5'in Gazebo kipi Gazebo Harmonic ile çalışır. Bunun için Harmonic
kurulur ve iki köprü deposu ([`gazebo_harmonic.repos`](gazebo_harmonic.repos):
`ros_gz` ve `gz_ros2_control`, `humble` dalları) çalışma alanında Harmonic'e karşı
derlenir. Aşağıdaki adımlar yukarıdaki **Kurulum** adımlarından sonra bir kez
yapılır.

> Bölüm 4 ve Bölüm 5'in mock (donanımsız) kipi Gazebo gerektirmez; yalnızca bunlarla
> çalışacaksanız bu bölümü atlayabilirsiniz.

**1. Gazebo Harmonic'i kurun** (OSRF paket deposu):

```bash
$ sudo apt install curl lsb-release gnupg
$ sudo curl https://packages.osrfoundation.org/gazebo.gpg \
    --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
$ sudo apt update
$ sudo apt install gz-harmonic
```

**2. Köprü paketlerini çalışma alanına ekleyip Harmonic'e karşı derleyin:**

```bash
$ cd ~/ros2_ws
$ vcs import src < src/ROS2-Book/gazebo_harmonic.repos

# GZ_VERSION, derlemenin hangi Gazebo sürümüne bağlanacağını belirler.
# Kalıcı olması için ~/.bashrc dosyasına eklenir.
$ echo "export GZ_VERSION=harmonic" >> ~/.bashrc
$ export GZ_VERSION=harmonic

$ source /opt/ros/humble/setup.bash
# gz-* kütüphaneleri 1. adımda gz-harmonic ile kuruldu; rosdep bu anahtarları
# tanımadığı için atlanır.
$ rosdep install --from-paths src --ignore-src -r -y \
    --skip-keys "gz-math7 gz-msgs10 gz-plugin2 gz-sim8 gz-transport13"
$ colcon build --symlink-install --packages-up-to ch05_simulation_visualization
$ source install/setup.bash
```

`--packages-up-to`, yalnızca Bölüm 5 paketinin ihtiyaç duyduğu paketleri
(`ros_gz_sim`, `ros_gz_bridge`, `ros_gz_interfaces`, `gz_ros2_control`) derler.

**3. Denetleyin:**

```bash
$ ros2 launch ch05_simulation_visualization cell_bringup.launch.py
# 2. terminal: denetleyiciler 'active' görünmelidir
$ ros2 control list_controllers
```

Çalışma alanındaki `ros_gz` ve `gz_ros2_control`, `/opt/ros/humble` altındaki
Fortress paketlerinin önüne geçer; Fortress paketlerinin kaldırılması gerekmez.

## Kod standartları

- Değişken, fonksiyon, dosya ve paket adları İngilizce; kod içi yorumlar Türkçedir.
- Python kodları PEP 8 ve PEP 484 (tip ipuçları) kurallarına uyar. Her paket
  `ament_flake8` ve `ament_pep257` denetimlerinden geçer:

  ```bash
  $ colcon test --packages-select ch04_ros2_communication ch05_simulation_visualization
  $ colcon test-result --verbose
  ```

- Tüm bağımlılıklar `package.xml` içinde tanımlıdır; `rosdep install` hepsini kurar.

## Lisans

[Apache License 2.0](LICENSE)
