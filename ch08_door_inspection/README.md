# ch08_door_inspection — Bölüm 8: Otomotiv Yüzey Kalite Kontrolü: Doblo Kapısı İnceleme

> Bu klasörde derlenecek bir paket yoktur. Bölüm, gerçek bir robot hücresine ve
> onlarca sürücü paketine dayandığı için kodu
> [ESOGU-ASRLAB/ESOGU-HILTest-DualRobot](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master) araştırma deposundadır (master dalı).
> Depoyu çalışma alanınıza eklemek için ana [README](../README.md)'deki
> **Bölüm 7–10: araştırma deposunun eklenmesi** adımlarını izleyin.

Bölümün tamamı [`doors_inspection`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master/doors_inspection) paketindedir; operatör
paneli ayrı bir paket olan [`doors_user_interface`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master/doors_user_interface)
içindedir. Kapı işi kendi planlayıcısını yazmaz, Bölüm 7'nin
[`multirobot_viewpoint_planner`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master/multirobot_viewpoint_planner) düğümlerini
kapı mesh'iyle yeniden kullanır. Çalıştırma adımları paketin
[README](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/doors_inspection/README.md) dosyasındadır.

## Bölümde anlatılanların depodaki karşılıkları

| Bölümde Anlatılan | Depodaki Karşılığı | Ne Yapar |
|---|---|---|
| 8.1.2 İki kolun iş bölümü | [`config/doors_params.yaml`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/doors_inspection/config/doors_params.yaml) | Kol bütçelerini ve çalışma uzayı kutusunu tanımlar. |
| 8.2.1 Doğrulanmış hattın yeniden kullanımı | [`launch/doors_planning.launch.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/doors_inspection/launch/doors_planning.launch.py) | Şase planlayıcısını kapı mesh'iyle başlatır. |
| 8.2.2 Çıktı yollarının ayrılması | [`launch/doors_inspection.launch.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/doors_inspection/launch/doors_inspection.launch.py) | Yürütmeyi kapıya ait çıktı ağacıyla koşturur. |
| 8.3.2 İki kapı tek mesh modeli | [`doors_inspection/merge_door_meshes.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/doors_inspection/doors_inspection/merge_door_meshes.py) | İki mesh'i tek hedef yüzeyde birleştirir. |
| 8.4.2 Kalite raporu üretimi | [`src/build_doors_octomap.cpp`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/doors_inspection/src/build_doors_octomap.cpp), [`launch/doors_octomap.launch.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/doors_inspection/launch/doors_octomap.launch.py) | Referans ve işgal haritalarından kaplamayı hesaplar. |
| 8.4.3 Operatör arayüzü entegrasyonu | [`doors_inspection/doors_mission_node.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/doors_inspection/doors_inspection/doors_mission_node.py), [`launch/doors_mission.launch.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/doors_inspection/launch/doors_mission.launch.py) | Panel komutlarını dinler, tarama ve temizliği yönetir. |
