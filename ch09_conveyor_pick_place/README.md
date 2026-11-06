# ch09_conveyor_pick_place — Bölüm 9: Yapay Zekâ ve Görsel Algılama Destekli Konveyör Al-Yerleştir Otomasyonu

Bölümün kodu [`ch09_conveyor_pick_place`](ch09_conveyor_pick_place) ROS 2 paketindedir
ve [`launch/gemini_pick_place.launch.py`](ch09_conveyor_pick_place/launch/gemini_pick_place.launch.py)
ile başlatılır. Gemini Robotics ER 2 derinlik görüntüsünden 2D nokta üretir; düğümler
noktayı nokta bulutuyla 3D'ye ve `world` çerçevesine taşır, MoveIt 2 üzerinden
(ortak [`book_pymoveit2`](../common/book_pymoveit2) kütüphanesi) al-bırak yaptırır.

> **Tam kurulum için araştırma deposu gerekir.** Paket bu depoda derlenir; ancak
> çalıştırmak için robot hücresinin MoveIt 2 yapılandırması, Gazebo ikizi (konveyör,
> takım kutusu, vakum için koparılabilir eklem) ve kamera köprüsü gerekir. Bunlar
> [ESOGU-ASRLAB/ESOGU-HILTest-DualRobot](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot)
> deposundadır; eklemek için ana [README](../README.md)'deki
> **Bölüm 7–10: araştırma deposunun eklenmesi** adımlarını izleyin.

## Gemini SDK ve API anahtarı

Gemini Python SDK'sının (`google-genai`) rosdep anahtarı yoktur; `pip` ile kurulur:

```bash
$ pip install --upgrade google-genai
```

Anahtar sırasıyla `GEMINI_API_KEY` / `GOOGLE_API_KEY` ortam değişkenlerinden, sonra
`~/.config/gemini/api_key` ve `/etc/gemini/api_key` dosyalarından okunur:

```bash
$ mkdir -p ~/.config/gemini
$ printf '%s\n' 'ANAHTARINIZ' > ~/.config/gemini/api_key
$ chmod 600 ~/.config/gemini/api_key
```

## Çalıştırma

```bash
# 1. terminal: robot hücresi (sahte donanım + Gazebo'da vakum kavrayıcı)
$ ros2 launch my_robot_cell_control hil_test_whole_unified.launch.py \
    use_fake_hardware:=true use_mock_hardware:=true fake_sensor_commands:=true \
    use_vacuum_gripper:=true

# 2. terminal: algılama + al-bırak düğümleri (mode:=sim varsayılandır)
$ ros2 launch ch09_conveyor_pick_place gemini_pick_place.launch.py

# 3. terminal: doğal dil komutu
$ ros2 topic pub --once /gemini/command std_msgs/String \
    "{data: 'düz kutunun üstünden tut ve konveyöre koy'}"
$ ros2 topic echo /gemini/status
```

Koşu kayıtları `~/ros2_book_output/ch09/recordings/` altına yazılır (kapatmak için
`record:=false`).

Gerçek OnRobot VGC10 sürücüsünün mesaj paketi (`vgc10_msgs`) bu depoya alınmamıştır:
Gazebo'da kavrama koparılabilir eklemle yapılır ve paket yoksa
[`vacuum.py`](ch09_conveyor_pick_place/ch09_conveyor_pick_place/vacuum.py) donanım
konularını atlar. Gerçek hücrede çalışırken paket araştırma deposundan gelir.

## Bölümde anlatılanların depodaki karşılıkları

| Bölümde Anlatılan | Depodaki Karşılığı | Ne Yapar |
|---|---|---|
| 9.1.1 Konveyör ikizi | [`config/gemini_params.yaml`](ch09_conveyor_pick_place/config/gemini_params.yaml), [`config/mode_sim.yaml`](ch09_conveyor_pick_place/config/mode_sim.yaml), [`config/mode_real.yaml`](ch09_conveyor_pick_place/config/mode_real.yaml) | Tarama pozlarını, eşikleri ve kip ayrımını tanımlar. |
| 9.1.2 Vakum sürücüsü | [`vacuum.py`](ch09_conveyor_pick_place/ch09_conveyor_pick_place/vacuum.py) | Komutu yollar, simülasyonda koparılabilir eklemi kurar. |
| 9.3.1 Derinlik render | [`depth_render.py`](ch09_conveyor_pick_place/ch09_conveyor_pick_place/depth_render.py) | Derinliği sabit metrik aralıkta görüntüye çevirir. |
| 9.3.2 Zaman damgalı dönüşüm | [`locator.py`](ch09_conveyor_pick_place/ch09_conveyor_pick_place/locator.py) | Pikseli ölçüm anındaki dönüşümle dünyaya taşır. |
| 9.3.3 Kavrama pozu ve normal | [`grasp.py`](ch09_conveyor_pick_place/ch09_conveyor_pick_place/grasp.py) | Yüzey normalinden kavrama yönelimini üretir. |
| 9.4.1 – 9.4.4 Görev ve parça | [`pick_place_node.py`](ch09_conveyor_pick_place/ch09_conveyor_pick_place/pick_place_node.py), [`payload.py`](ch09_conveyor_pick_place/ch09_conveyor_pick_place/payload.py) | Aşamaları yürütür, parçayı sahneye ekler, koşuyu yazar. |
| ER 2 istemcisi | [`er_client.py`](ch09_conveyor_pick_place/ch09_conveyor_pick_place/er_client.py) | Görüntü + metni ER 2'ye gönderir, 2D noktaları ayrıştırır. |
