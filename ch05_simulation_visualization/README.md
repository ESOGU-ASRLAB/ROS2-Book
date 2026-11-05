# ch05_simulation_visualization — Bölüm 5: Simülasyon ve Görselleştirme Ortamları

Bölüm 4'teki 6-eksenli kolun, aynı URDF ile iki farklı donanım kipinde ayağa
kaldırılması:

| Kip | Argüman | Donanım eklentisi | Ne gerektirir |
|---|---|---|---|
| Gazebo simülasyonu | `use_fake_hardware:=false` (varsayılan) | `gz_ros2_control/GazeboSimSystem` | Gazebo Harmonic |
| Mock (donanımsız test) | `use_fake_hardware:=true` | `mock_components/GenericSystem` | Yalnız ROS 2 |

Her iki kipte de aynı denetleyiciler (`joint_state_broadcaster`,
`joint_trajectory_controller`) ve aynı yörünge düğümü kullanılır. Üst katmanın
donanım değişiminden etkilenmemesi, ros2_control soyutlamasının amacıdır.

> Gazebo kipi için ana [README](../README.md)'deki **Humble + Gazebo Harmonic notu**na bakınız.

## Kitaptaki listelerin depodaki karşılıkları

| Liste | Kitaptaki ad | Depodaki dosya | Not |
|---|---|---|---|
| 5a | `manipulator.urdf.xacro` | [`urdf/manipulator.urdf.xacro`](urdf/manipulator.urdf.xacro) | Tek eklemli parça 6 ekleme tamamlandı; Gazebo için kütle, eylemsizlik ve `gz_ros2_control` eklentisi eklendi |
| 5b | `cell_bringup.launch.py` | [`launch/cell_bringup.launch.py`](launch/cell_bringup.launch.py) | `my_robot_description` → `ch05_simulation_visualization`; denetleyiciler, Gazebo, köprü ve RViz2 eklendi |
| 6 | köprü örneği, `bridge.yaml` | [`config/bridge.yaml`](config/bridge.yaml) | `/clock`, kamera görüntüsü ve kamera bilgisi köprülenir |
| — | — | [`config/controllers.yaml`](config/controllers.yaml) | Bölüm 4, Liste 23'ün çalışan hâli |
| — | — | [`worlds/cell.sdf`](worlds/cell.sdf) | Zemin, ışık, iş parçası ve sabit kamera |
| — | — | [`ch05_simulation_visualization/send_trajectory.py`](ch05_simulation_visualization/send_trajectory.py) | Kolu üç ara noktadan geçiren yörünge istemcisi |

## Çalıştırma

### Mock kipi (donanımsız)

```bash
# 1. terminal
$ ros2 launch ch05_simulation_visualization cell_bringup.launch.py use_fake_hardware:=true

# 2. terminal
$ ros2 control list_controllers
joint_state_broadcaster     joint_state_broadcaster/JointStateBroadcaster          active
joint_trajectory_controller joint_trajectory_controller/JointTrajectoryController  active
$ ros2 run ch05_simulation_visualization send_trajectory
[INFO] [trajectory_sender]: Bitti, error_code=0
```

### Gazebo kipi

```bash
# 1. terminal
$ ros2 launch ch05_simulation_visualization cell_bringup.launch.py
# arayüzsüz çalıştırmak için: headless:=true

# 2. terminal: zaman Gazebo'dan geldiği için use_sim_time:=true
$ ros2 run ch05_simulation_visualization send_trajectory --ros-args -p use_sim_time:=true

# Köprünün çalıştığını denetleyin
$ ros2 topic hz /clock              # ~1000 Hz (1 ms adım)
$ ros2 topic hz /camera/image_raw   # ~15 Hz
```

RViz2 penceresinde robot modeli, TF ağacı ve hücre kamerasının görüntüsü açılır.
Sabit çerçeve (Fixed Frame) `world`'dür.

> ⚠️ **YAYGIN ÖĞRENCİ HATASI (COMMON PITFALL):**
> Gazebo `/clock` adlı bir konu duyurur ama üzerinden yayın yapmaz; asıl saat
> `/world/<dünya_adı>/clock` konusundadır. Köprüde yanlış konu seçilirse
> `use_sim_time:=true` olan her düğümün zamanlayıcısı **hata vermeden** hiç tetiklenmez.
> Bu paketteki köprü `/world/cell/clock` konusunu ROS 2 `/clock` konusuna bağlar.

### 5.7 Kayıt, yeniden oynatma ve rqt

```bash
# Sistem çalışırken eklem durumlarını ve kamerayı kaydet
$ ros2 bag record -o cell_run /joint_states /camera/image_raw
$ ros2 bag info cell_run
$ ros2 bag play cell_run

$ ros2 run rqt_graph rqt_graph
$ ros2 run rqt_console rqt_console
$ ros2 run rqt_topic rqt_topic
$ ros2 run rqt_reconfigure rqt_reconfigure
```
