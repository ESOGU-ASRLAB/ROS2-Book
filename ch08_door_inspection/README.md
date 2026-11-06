# ch08_door_inspection — Bölüm 8: Otomotiv Yüzey Kalite Kontrolü: Doblo Kapısı İnceleme

Bölümün kodu bu klasördeki iki ROS 2 paketindedir:

- [`ch08_door_inspection`](ch08_door_inspection): kapı ağlarının birleştirilmesi,
  planlama ve yürütme başlatmaları, OctoMap kaplama raporu (C++) ve operatör paneliyle
  sürülen algılama + temizleme görevi.
- [`ch08_door_inspection_ui`](ch08_door_inspection_ui): görevi tarayıcıdan yöneten
  operatör paneli (Flask + SocketIO).

Kapı işi kendi planlayıcısını yazmaz; Bölüm 7'nin
[`ch07_multirobot_planner`](../ch07_bus_inspection/ch07_multirobot_planner) düğümlerini
kapı mesh'iyle yeniden kullanır.

> **Tam kurulum için araştırma deposu gerekir.** Paketler bu depoda derlenir; ancak
> çalıştırmak için robot hücresinin MoveIt 2 yapılandırması, denetleyicileri ve kapı
> mesh'leri (`windowdoor.stl`, `windowlessdoor.stl`; araştırma deposunun
> `ur_description` paketinde) gerekir. Bunlar
> [ESOGU-ASRLAB/ESOGU-HILTest-DualRobot](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot)
> deposundadır; eklemek için ana [README](../README.md)'deki
> **Bölüm 7–10: araştırma deposunun eklenmesi** adımlarını izleyin.

## Çalıştırma

```bash
# 1. terminal: robot hücresi (sahte donanım; harmony:=true görev konularını açar)
$ ros2 launch my_robot_cell_control hil_test_whole_unified.launch.py \
    use_fake_hardware:=true use_mock_hardware:=true fake_sensor_commands:=true harmony:=true

# 2. terminal: kapı planı (~/ros2_book_output/ch08/doors_viewpoint_plan.json)
$ ros2 launch ch08_door_inspection doors_planning.launch.py
# 3. terminal: planlamayı tetikleyin
$ ros2 service call /doors_planner_node/plan std_srvs/srv/Trigger

# Yürütme (varsayılan olarak paketle gelen kitap planını kullanır) ve kaplama raporu
$ ros2 launch ch08_door_inspection doors_inspection.launch.py
$ ros2 launch ch08_door_inspection doors_octomap.launch.py

# Operatör paneliyle görev: http://localhost:8082
$ ros2 run ch08_door_inspection_ui doors_dashboard
```

Panel, alt süreçleri başlatmadan önce `~/ros2_ws/install/setup.bash` dosyasını
kaynaklar; çalışma alanınız başka bir yerdeyse `ROS2_WS` ortam değişkeniyle verin
(`$ ROS2_WS=~/baska_ws ros2 run ch08_door_inspection_ui doors_dashboard`).

## Bölümde anlatılanların depodaki karşılıkları

| Bölümde Anlatılan | Depodaki Karşılığı | Ne Yapar |
|---|---|---|
| 8.1.2 İki kolun iş bölümü | [`config/doors_params.yaml`](ch08_door_inspection/config/doors_params.yaml) | Kol bütçelerini ve çalışma uzayı kutusunu tanımlar. |
| 8.2.1 Doğrulanmış hattın yeniden kullanımı | [`launch/doors_planning.launch.py`](ch08_door_inspection/launch/doors_planning.launch.py) | Şase planlayıcısını kapı mesh'iyle başlatır. |
| 8.2.2 Çıktı yollarının ayrılması | [`launch/doors_inspection.launch.py`](ch08_door_inspection/launch/doors_inspection.launch.py), [`paths.py`](ch08_door_inspection/ch08_door_inspection/paths.py) | Yürütmeyi kapıya ait çıktı ağacıyla koşturur. |
| 8.3.2 İki kapı tek mesh modeli | [`merge_door_meshes.py`](ch08_door_inspection/ch08_door_inspection/merge_door_meshes.py) | İki mesh'i tek hedef yüzeyde birleştirir. |
| 8.4.2 Kalite raporu üretimi | [`src/build_doors_octomap.cpp`](ch08_door_inspection/src/build_doors_octomap.cpp), [`door_parts.py`](ch08_door_inspection/ch08_door_inspection/door_parts.py), [`launch/doors_octomap.launch.py`](ch08_door_inspection/launch/doors_octomap.launch.py) | Referans ve işgal haritalarından kaplamayı hesaplar. |
| 8.4.3 Operatör arayüzü entegrasyonu | [`doors_mission_node.py`](ch08_door_inspection/ch08_door_inspection/doors_mission_node.py), [`cleaning_mission_runner.py`](ch08_door_inspection/ch08_door_inspection/cleaning_mission_runner.py), [`launch/doors_mission.launch.py`](ch08_door_inspection/launch/doors_mission.launch.py), [`ch08_door_inspection_ui/app.py`](ch08_door_inspection_ui/ch08_door_inspection_ui/app.py) | Panel komutlarını dinler, tarama ve temizliği yönetir. |
| Kitapta kullanılan plan | [`plans/doors_viewpoint_plan.json`](ch08_door_inspection/plans/doors_viewpoint_plan.json) | Yürütmenin varsayılan kapı planı. |
