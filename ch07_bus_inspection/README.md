# ch07_bus_inspection — Bölüm 7: Otobüs Şasesinin Kalite Kontrolü İçin Hareket Planlama

> Bu klasörde derlenecek bir paket yoktur. Bölüm, gerçek bir robot hücresine ve
> onlarca sürücü paketine dayandığı için kodu
> [ESOGU-ASRLAB/ESOGU-HILTest-DualRobot](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master) araştırma deposundadır (master dalı).
> Depoyu çalışma alanınıza eklemek için ana [README](../README.md)'deki
> **Bölüm 7–10: araştırma deposunun eklenmesi** adımlarını izleyin.

Uygulamanın tamamı [`multirobot_viewpoint_planner`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master/multirobot_viewpoint_planner)
paketindedir. Paket, tek kollu temel planlayıcı olan
[`viewpoint_planner`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master/viewpoint_planner) paketinin geometri ve erişilebilirlik
modüllerini kullanır.

## Bölümde anlatılanların depodaki karşılıkları

| Bölümde Anlatılan | Depodaki Karşılığı | Ne Yapar |
|---|---|---|
| 7.1 1. Aşama — Offline planlama | [`launch/multirobot_planning.launch.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/multirobot_viewpoint_planner/launch/multirobot_planning.launch.py), [`multirobot_planner_node.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/multirobot_viewpoint_planner/multirobot_viewpoint_planner/multirobot_planner_node.py) | Mesh'ten bakış noktalarını üretir, iki kola paylaştırır ve planı dosyaya yazar. |
| 7.1 2. Aşama — Online yürütme | [`launch/multirobot_inspection.launch.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/multirobot_viewpoint_planner/launch/multirobot_inspection.launch.py) | İki kolun yürütücülerini ayrı süreçlerde başlatır. |
| 7.1 UR10e yürütücüsü | [`ur_inspection_node.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/multirobot_viewpoint_planner/multirobot_viewpoint_planner/ur_inspection_node.py) | UR10e'yi doğrusal eksenle birlikte bakış noktalarında gezdirir. |
| 7.1 Kawasaki yürütücüsü | [`kawasaki_inspection_node.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/multirobot_viewpoint_planner/multirobot_viewpoint_planner/kawasaki_inspection_node.py) | AGV üzerindeki Kawasaki kolunu kendi bakış noktalarında gezdirir. |
| 7.1 Ortak yürütme altyapısı | [`inspection_base.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/multirobot_viewpoint_planner/multirobot_viewpoint_planner/inspection_base.py) | İki yürütücünün paylaştığı sahne yükleme ve kayıt mantığı. |
| 7.3 Adım 2 — Yüzey normalleri | [`viewpoint_planner/mesh_analyzer.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/viewpoint_planner/viewpoint_planner/mesh_analyzer.py) | Mesh'in yüzey normallerini hesaplar ve hedef noktaları örnekler. |
| 7.3 Adım 4 — Aday bakış noktaları | [`viewpoint_planner/viewpoint_generator.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/viewpoint_planner/viewpoint_planner/viewpoint_generator.py) | Standoff mesafeleri ve eğimlerle aday kamera pozlarını üretir. |
| 7.3 Çarpışma denetimli IK | [`viewpoint_planner/reachability_checker.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/viewpoint_planner/viewpoint_planner/reachability_checker.py) | Adayları MoveIt 2 `/compute_ik` servisiyle sınar. |
| 7.4 Açgözlü set-cover ve iki kola tahsis | [`robot_allocator.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/multirobot_viewpoint_planner/multirobot_viewpoint_planner/robot_allocator.py) | Marjinal kazanç eşiğinin altına düşünce durarak bakış noktalarını seçer. |
| 7.4 Planlama parametreleri | [`config/multirobot_params.yaml`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/multirobot_viewpoint_planner/config/multirobot_params.yaml) | Bütçeler, eşikler (`tail_floor`) ve çalışma uzayı. |
| 7.3 Planın RViz2'de gösterimi | [`visualization.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/multirobot_viewpoint_planner/multirobot_viewpoint_planner/visualization.py) | Planı MarkerArray okları olarak yayımlar. |
