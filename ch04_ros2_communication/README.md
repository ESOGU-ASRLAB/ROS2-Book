# ch04_ros2_communication — Bölüm 4: Gelişmiş ROS 2 İletişim Yapıları ve Araçlar

Bölüm 4'teki servis, eylem, parametre, başlatma dosyası, URDF/Xacro ve TF2
örnekleri. Donanım gerektirmez. Örneklerin çoğu `turtlesim` ile çalışır.

Paketin kurulumu için ana [README](../README.md) dosyasına bakınız. Aşağıdaki
komutlar `source ~/ros2_ws/install/setup.bash` yapılmış bir terminali varsayar.

## Kitaptaki listelerin depodaki karşılıkları

Editöryal standart gereği depodaki **adlar İngilizce, yorumlar Türkçedir**. Bu yüzden
bazı dosya, değişken ve fonksiyon adları kitaptaki listelerden farklıdır.

| Liste | Kitaptaki ad | Depodaki dosya | Ad değişiklikleri |
|---|---|---|---|
| 1 | — | (yalnız komutlar) | — |
| 3 | `add_two_ints_server.py` | [`ch04_ros2_communication/add_two_ints_server.py`](ch04_ros2_communication/add_two_ints_server.py) | — |
| 4 | `add_two_ints_client.py` | [`ch04_ros2_communication/add_two_ints_client.py`](ch04_ros2_communication/add_two_ints_client.py) | `topla()` → `add()`, `dugum` → `node` |
| 5 | `python3 add_two_ints_server.py` | — | `ros2 run ch04_ros2_communication add_two_ints_server` |
| 6 | Trigger sunucusu | [`ch04_ros2_communication/trigger_server.py`](ch04_ros2_communication/trigger_server.py) | servis `baslat` → `start`, `_isi_yap()` → `_do_work()` |
| 9, 10 | eylem istemcisi, `durdur()` | [`ch04_ros2_communication/rotate_client.py`](ch04_ros2_communication/rotate_client.py) | `hedef` → `goal`, `gonder` → `send_future`, `tutamac` → `goal_handle`, `sonuc` → `result_future`, `durdur()` → `stop()`, `_en_buyuk_hiz()` → `_angular_speed` |
| 12, 13, 15 | `MyNode` | [`ch04_ros2_communication/parameter_demo.py`](ch04_ros2_communication/parameter_demo.py) | `robot_adi` → `robot_name`, `yayin_frekansi` → `publish_rate`, `azami_hiz` → `max_speed`, `gunluk_acik` → `logging_enabled`, `calisma_kipi` → `operation_mode` |
| 14 | `config/ayarlar.yaml` | [`config/params.yaml`](config/params.yaml) | `my_pkg` → `ch04_ros2_communication` |
| 14 (hata) | — | [`config/params_wrong_types.yaml`](config/params_wrong_types.yaml) | YAML tip tuzağını bilerek tetikleyen dosya |
| 16 | `iki_kaplumbaga.launch.py` | [`launch/two_turtles.launch.py`](launch/two_turtles.launch.py) | `arkaplan_r` → `background_r`, `kaplumbaga1/2` → `turtle1/2` |
| 18 | `hucre_bringup.launch.py` | [`launch/cell_bringup.launch.py`](launch/cell_bringup.launch.py) | `my_robot_description` → `ch04_ros2_communication`; Gazebo'suz deneme için `use_gazebo` argümanı eklendi |
| 19 | `mini_arm.urdf.xacro` | [`urdf/mini_arm.urdf.xacro`](urdf/mini_arm.urdf.xacro) | `gri/mavi` → `grey/blue`, `taban_boy` → `base_length`, `kol1_boy` → `arm1_length`, `kol2_boy` → `arm2_length`, makro `silindir(ad boy yaricap renk)` → `cylinder_link(name length radius color)` |
| 20 | `mini_arm.launch.py` | [`launch/mini_arm.launch.py`](launch/mini_arm.launch.py) | `model` varsayılanı paketin içindeki tam yol oldu |
| 22 | nokta dönüşümü | [`ch04_ros2_communication/point_transformer.py`](ch04_ros2_communication/point_transformer.py) | `_on_nokta` → `_on_point`, `sonuc` → `result` |
| 23 | `controllers.yaml` | [`config/controllers.yaml`](config/controllers.yaml) | Çalışan hâli Bölüm 5 paketindedir |
| 24 | `alti_eksen_kol.urdf.xacro` | [`urdf/six_axis_arm.urdf.xacro`](urdf/six_axis_arm.urdf.xacro) | robot adı `alti_eksen_manipulator` → `six_axis_manipulator` |
| 25 | `tcp_tf2_hesaplayici.py` | [`ch04_ros2_communication/tcp_tf2_monitor.py`](ch04_ros2_communication/tcp_tf2_monitor.py) | `_hesapla_flans_ve_tcp` → `_compute_flange_and_tcp`, `t_flans` → `t_flange`, `hedef_tcp/hedef_base` → `sample_tcp/sample_base` |
| 26 | `my_robot_description manipulator_rsp.launch.py` | [`launch/manipulator_rsp.launch.py`](launch/manipulator_rsp.launch.py) | Kitapta tanımı olmayan alt başlatma dosyası eklendi |

## Çalıştırma

### 4.1 Servisler

```bash
# 1. terminal
$ ros2 run ch04_ros2_communication add_two_ints_server
[INFO] [add_two_ints_server]: Hazır: add_two_ints bekleniyor

# 2. terminal
$ ros2 run ch04_ros2_communication add_two_ints_client 3 4
3 + 4 = 7
$ ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 7, b: 5}"
```

Hatanın yanıta taşınması (Liste 6):

```bash
$ ros2 run ch04_ros2_communication trigger_server
$ ros2 service call /start std_srvs/srv/Trigger
std_srvs.srv.Trigger_Response(success=True, message='3 öğe işlendi')
$ ros2 param set /trigger_server simulate_failure true
$ ros2 service call /start std_srvs/srv/Trigger
std_srvs.srv.Trigger_Response(success=False, message='başarısız: sensör yanıt vermedi (benzetim)')
```

### 4.2 Eylemler

```bash
# 1. terminal
$ ros2 run turtlesim turtlesim_node

# 2. terminal: hedefi gönder, sonucu bekle
$ ros2 run ch04_ros2_communication rotate_client
[INFO] [rotate_client]: Durum: SUCCEEDED, dönülen açı: -1.55 rad

# hedefi 0.5 s sonra iptal et ve hareketin durmasını bekle (Liste 10)
$ ros2 run ch04_ros2_communication rotate_client --ros-args \
    -p target_theta:=-3.0 -p cancel_after_sec:=0.5
[INFO] [rotate_client]: Durum: CANCELED, dönülen açı: -0.51 rad
```

### 4.3 Parametreler

```bash
# 1. terminal
$ ros2 run turtlesim turtlesim_node

# 2. terminal (paket klasöründen)
$ cd ~/ros2_ws/src/ROS2-Book/ch04_ros2_communication
$ ros2 run ch04_ros2_communication parameter_demo --ros-args --params-file config/params.yaml

# 3. terminal: çalışma anında değiştir
$ ros2 param set /my_node max_speed 0.9
Set parameter successful
$ ros2 param set /my_node max_speed 5.0
Setting parameter failed: Parameter max_speed out of range Min: 0.0, Max: 1.0, value: 5.0
```

> ⚠️ **YAYGIN ÖĞRENCİ HATASI (COMMON PITFALL):**
> `config/params_wrong_types.yaml` dosyasında `operation_mode: on` tırnaksız yazılmıştır.
> Bu dosyayla başlatılan düğüm şu hatayla durur:
> `InvalidParameterTypeException: Trying to set parameter 'operation_mode' to 'True' of type 'BOOL', expecting type 'STRING'`

### 4.4 Başlatma dosyaları

```bash
$ ros2 launch ch04_ros2_communication two_turtles.launch.py --show-args
$ ros2 launch ch04_ros2_communication two_turtles.launch.py background_r:=200
```

### 4.5 URDF ve Xacro

```bash
$ ros2 launch ch04_ros2_communication mini_arm.launch.py
```

### 4.6 TF2

```bash
# 1. terminal
$ ros2 launch ch04_ros2_communication mini_arm.launch.py

# 2. terminal
$ ros2 run tf2_ros tf2_echo base_link camera_optical_frame
$ ros2 run tf2_tools view_frames

# Liste 22: optik çerçevedeki bir noktayı base_link'e çevir
$ ros2 run ch04_ros2_communication point_transformer
$ ros2 topic pub --once /camera_point geometry_msgs/msg/PointStamped \
    "{header: {frame_id: camera_optical_frame}, point: {x: 0.0, y: 0.0, z: 0.5}}"
```

### 4.8 Uygulamalı alıştırma: flanş (tool0) ve TCP

```bash
# 1. terminal: varsayılan TCP ofseti (0.145 m)
$ ros2 launch ch04_ros2_communication manipulator_rsp.launch.py

# 2. terminal
$ ros2 run ch04_ros2_communication tcp_tf2_monitor
[INFO] [manipulator_tcp_monitor]: Flanş (tool0) [m]: X=0.515, Y=0.000, Z=0.900 | TCP (tcp_link) [m]: X=0.660, Y=0.000, Z=0.900 (Quat w=0.71)
[INFO] [manipulator_tcp_monitor]: Numune Noktası (base_link) [m]: (0.710, 0.000, 0.900)

# 3. terminal: üst başlatma dosyasından tcp_offset_z'yi ezerek yeniden başlat
$ ros2 launch ch04_ros2_communication cell_bringup.launch.py tcp_offset_z:=0.220
# -> TCP (tcp_link) X=0.735 olur
```

Gazebo kurulu değilse: `cell_bringup.launch.py use_gazebo:=false use_sim_time:=false`.
Eklemleri elle oynatmak için: `manipulator_rsp.launch.py gui:=true rviz:=true`.
