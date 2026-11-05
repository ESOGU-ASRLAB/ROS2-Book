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
- Gazebo Harmonic (yalnızca Bölüm 5'in Gazebo kipi için; aşağıdaki nota bakınız)

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

## Humble + Gazebo Harmonic notu

ROS 2 Humble'ın varsayılan Gazebo eşi Gazebo Fortress'tır. `rosdep`'in kurduğu
`ros-humble-ros-gz-*` ve `ros-humble-gz-ros2-control` ikili paketleri Fortress'a
bağlıdır. Bölüm 5'in Gazebo kipi bu kitapta Harmonic ile sınanmıştır. Harmonic
kullanmak için `ros_gz` ve `gz_ros2_control` paketlerinin Harmonic'e karşı derlenmiş
hâlleri gerekir:

- `ros_gz` ve `gz_ros2_control` paketlerini `humble` dallarından, `GZ_VERSION=harmonic`
  ortam değişkeniyle çalışma alanında kaynaktan derleyin, ya da
- Gazebo'nun resmî ROS kurulum rehberindeki Humble + Harmonic yönergelerini izleyin:
  <https://gazebosim.org/docs/harmonic/ros_installation>

Bölüm 4 ve Bölüm 5'in mock (donanımsız) kipi Gazebo gerektirmez.

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
