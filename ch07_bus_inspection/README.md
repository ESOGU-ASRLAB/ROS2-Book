# ch07_bus_inspection — Bölüm 7: Otobüs Şasesinin Kalite Kontrolü İçin Hareket Planlama

Bölümün kodu bu klasördeki iki ROS 2 paketindedir:

- [`ch07_viewpoint_planner`](ch07_viewpoint_planner): tek kollu temel planlayıcının
  geometri ve erişilebilirlik modülleri (mesh analizi, aday bakış noktaları, IK sınaması).
- [`ch07_multirobot_planner`](ch07_multirobot_planner): iki kola (UR10e + Kawasaki)
  paylaştırılan bakış noktası planlaması ve muayenenin yürütülmesi.

Hareket komutları ortak [`book_pymoveit2`](../common/book_pymoveit2) kütüphanesi
üzerinden MoveIt 2'ye gider.

> **Tam kurulum için araştırma deposu gerekir.** Paketler bu depoda derlenir; ancak
> çalıştırmak için robot hücresinin MoveIt 2 yapılandırması, denetleyicileri ve
> şase mesh'i (`chassis.stl`, araştırma deposunun `ur_description` paketinde)
> gerekir. Bunlar
> [ESOGU-ASRLAB/ESOGU-HILTest-DualRobot](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot)
> deposundadır; eklemek için ana [README](../README.md)'deki
> **Bölüm 7–10: araştırma deposunun eklenmesi** adımlarını izleyin.

## Çalıştırma

```bash
# 1. terminal: robot hücresi (sahte donanım)
$ ros2 launch my_robot_cell_control hil_test_whole_unified.launch.py \
    use_fake_hardware:=true use_mock_hardware:=true fake_sensor_commands:=true

# 2. terminal: 1. aşama, çevrim dışı planlama (plan ~/ros2_book_output/ch07/ altına yazılır)
$ ros2 launch ch07_multirobot_planner multirobot_planning.launch.py
# 3. terminal: planlamayı tetikleyin
$ ros2 service call /multirobot_planner_node/plan std_srvs/srv/Trigger

# 2. aşama, çevrim içi yürütme (varsayılan olarak paketle gelen kitap planını kullanır)
$ ros2 launch ch07_multirobot_planner multirobot_inspection.launch.py
```

## Bölümde anlatılanların depodaki karşılıkları

| Bölümde Anlatılan | Depodaki Karşılığı | Ne Yapar |
|---|---|---|
| 7.1 1. Aşama — Offline planlama | [`launch/multirobot_planning.launch.py`](ch07_multirobot_planner/launch/multirobot_planning.launch.py), [`multirobot_planner_node.py`](ch07_multirobot_planner/ch07_multirobot_planner/multirobot_planner_node.py) | Mesh'ten bakış noktalarını üretir, iki kola paylaştırır ve planı dosyaya yazar. |
| 7.1 2. Aşama — Online yürütme | [`launch/multirobot_inspection.launch.py`](ch07_multirobot_planner/launch/multirobot_inspection.launch.py) | İki kolun yürütücülerini ayrı süreçlerde başlatır. |
| 7.1 UR10e yürütücüsü | [`ur_inspection_node.py`](ch07_multirobot_planner/ch07_multirobot_planner/ur_inspection_node.py) | UR10e'yi doğrusal eksenle birlikte bakış noktalarında gezdirir. |
| 7.1 Kawasaki yürütücüsü | [`kawasaki_inspection_node.py`](ch07_multirobot_planner/ch07_multirobot_planner/kawasaki_inspection_node.py) | AGV üzerindeki Kawasaki kolunu kendi bakış noktalarında gezdirir. |
| 7.1 Ortak yürütme altyapısı | [`inspection_base.py`](ch07_multirobot_planner/ch07_multirobot_planner/inspection_base.py) | İki yürütücünün paylaştığı sahne yükleme ve kayıt mantığı. |
| 7.3 Adım 2 — Yüzey normalleri | [`mesh_analyzer.py`](ch07_viewpoint_planner/ch07_viewpoint_planner/mesh_analyzer.py) | Mesh'in yüzey normallerini hesaplar ve hedef noktaları örnekler. |
| 7.3 Adım 4 — Aday bakış noktaları | [`viewpoint_generator.py`](ch07_viewpoint_planner/ch07_viewpoint_planner/viewpoint_generator.py) | Standoff mesafeleri ve eğimlerle aday kamera pozlarını üretir. |
| 7.3 Çarpışma denetimli IK | [`reachability_checker.py`](ch07_viewpoint_planner/ch07_viewpoint_planner/reachability_checker.py) | Adayları MoveIt 2 `/compute_ik` servisiyle sınar. |
| 7.4 Açgözlü set-cover ve iki kola tahsis | [`robot_allocator.py`](ch07_multirobot_planner/ch07_multirobot_planner/robot_allocator.py) | Marjinal kazanç eşiğinin (`tail_floor`) altına düşünce durarak bakış noktalarını seçer. |
| 7.4 Planlama parametreleri | [`config/multirobot_params.yaml`](ch07_multirobot_planner/config/multirobot_params.yaml) | Bütçeler, kuyruk kesici eşikleri (`min_new_points`, `min_marginal_coverage`) ve çalışma uzayı. |
| 7.3 Planın RViz2'de gösterimi | [`visualization.py`](ch07_multirobot_planner/ch07_multirobot_planner/visualization.py) | Planı MarkerArray okları olarak yayımlar. |
| Kitapta kullanılan plan | [`plans/multirobot_viewpoint_plan.json`](ch07_multirobot_planner/plans/multirobot_viewpoint_plan.json) | 25 UR10e + 11 Kawasaki bakış noktası; yürütmenin varsayılan planı. |
