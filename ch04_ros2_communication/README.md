# ch04_ros2_communication — Bölüm 4: Gelişmiş ROS 2 İletişim Yapıları ve Araçlar

Bölüm 4'teki servis, eylem, parametre, başlatma dosyası, URDF/Xacro ve TF2
örnekleri. Donanım gerektirmez. Örneklerin çoğu `turtlesim` ile çalışır.

Paketin kurulumu için ana [README](../README.md) dosyasına bakınız. Aşağıdaki
komutlar `source ~/ros2_ws/install/setup.bash` yapılmış bir terminali varsayar.

## Kitaptaki listelerin depodaki karşılıkları

Kitaptaki listeler depodaki dosyaların aynısıdır; birden çok listeye bölünen ya da
kitapta bir kısmı gösterilen dosyalarda (rotate_client.py, parameter_demo.py) her liste dosyadan
kesintisiz bir parçadır.

| Liste | Depodaki dosya | Açıklama |
|---|---|---|
| 2 | [`interfaces/AddTwoInts.srv`](interfaces/AddTwoInts.srv), [`interfaces/Trigger.srv`](interfaces/Trigger.srv) | Hazır servis arayüzleri (`example_interfaces`, `std_srvs`) |
| 3 | [`ch04_ros2_communication/add_two_ints_server.py`](ch04_ros2_communication/add_two_ints_server.py) | İki tam sayıyı toplayan servis sunucusu |
| 4 | [`ch04_ros2_communication/add_two_ints_client.py`](ch04_ros2_communication/add_two_ints_client.py) | Servisi asenkron çağıran istemci |
| 6 | [`ch04_ros2_communication/trigger_server.py`](ch04_ros2_communication/trigger_server.py) | Hatayı yanıta taşıyan Trigger sunucusu |
| 7 | [`interfaces/RotateAbsolute.action`](interfaces/RotateAbsolute.action) | Hazır eylem arayüzü (`turtlesim`) |
| 9, 10 | [`ch04_ros2_communication/rotate_client.py`](ch04_ros2_communication/rotate_client.py) | Eylem istemcisi, sonuç durumları ve güvenli iptal |
| 12, 13, 15 | [`ch04_ros2_communication/parameter_demo.py`](ch04_ros2_communication/parameter_demo.py) | Parametre bildirimi, aralık sınırı ve değişiklik doğrulaması |
| 14 | [`config/params.yaml`](config/params.yaml) | Parametre dosyası |
| 14 (hata) | [`config/params_wrong_types.yaml`](config/params_wrong_types.yaml) | YAML tip tuzağını bilerek tetikleyen dosya |
| 16 | [`launch/two_turtles.launch.py`](launch/two_turtles.launch.py) | Argüman, ad alanı ve iki düğüm |
| 18 | [`launch/cell_bringup.launch.py`](launch/cell_bringup.launch.py) | Alt başlatma dosyasını dâhil edip argümanlarını ezen üst dosya; Gazebo'suz deneme için `use_gazebo` argümanı |
| 19 | [`urdf/mini_arm.urdf.xacro`](urdf/mini_arm.urdf.xacro) | Xacro özellikleri ve makrolarıyla iki eklemli kol |
| 20 | [`launch/mini_arm.launch.py`](launch/mini_arm.launch.py) | Xacro modelini RViz2'de gösterir |
| 22 | [`ch04_ros2_communication/point_transformer.py`](ch04_ros2_communication/point_transformer.py) | Noktayı optik çerçeveden `base_link`'e çevirir |
| 23 | [`config/controllers.yaml`](config/controllers.yaml) | Denetleyici yapılandırması; çalışan hâli Bölüm 5 paketindedir |
| 24 | [`urdf/six_axis_arm.urdf.xacro`](urdf/six_axis_arm.urdf.xacro) | 6-eksenli kol, flanş (`tool0`) ve parametrik TCP |
| 25 | [`ch04_ros2_communication/tcp_tf2_monitor.py`](ch04_ros2_communication/tcp_tf2_monitor.py) | Flanş ve TCP dönüşümlerini izleyen düğüm |
| 26 | [`launch/manipulator_rsp.launch.py`](launch/manipulator_rsp.launch.py) | 6-eksenli kolu `tcp_offset_z` argümanıyla başlatan alt dosya |

## Çalıştırma

### 4.1 Servisler

Liste 2'deki arayüzler hazırdır; [`interfaces/`](interfaces) klasöründeki kopyalar
yalnızca başvuru içindir. Kurulu tanımlar şöyle görüntülenir:

```bash
$ ros2 interface show example_interfaces/srv/AddTwoInts
$ ros2 interface show std_srvs/srv/Trigger
```

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

Liste 7'deki eylem arayüzü `turtlesim` paketiyle gelir
(başvuru kopyası: [`interfaces/RotateAbsolute.action`](interfaces/RotateAbsolute.action)):

```bash
$ ros2 interface show turtlesim/action/RotateAbsolute
```

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
