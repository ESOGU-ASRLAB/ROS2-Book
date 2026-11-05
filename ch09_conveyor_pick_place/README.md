# ch09_conveyor_pick_place — Bölüm 9: Yapay Zekâ ve Görsel Algılama Destekli Konveyör Al-Yerleştir Otomasyonu

> Bu klasörde derlenecek bir paket yoktur. Bölüm, gerçek bir robot hücresine ve
> onlarca sürücü paketine dayandığı için kodu
> [ESOGU-ASRLAB/ESOGU-HILTest-DualRobot](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master) araştırma deposundadır (master dalı).
> Depoyu çalışma alanınıza eklemek için ana [README](../README.md)'deki
> **Bölüm 7–10: araştırma deposunun eklenmesi** adımlarını izleyin.

Uygulamanın tamamı [`gemini_robotics_ros`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master/gemini_robotics_ros) paketindedir
ve [`launch/gemini_pick_place.launch.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/gemini_robotics_ros/launch/gemini_pick_place.launch.py)
ile başlatılır. Simülasyon ve gerçek robot kipleri ile çalıştırma adımları paketin
[README](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/gemini_robotics_ros/README.md) dosyasındadır.

## Bölümde anlatılanların depodaki karşılıkları

| Bölümde Anlatılan | Depodaki Karşılığı | Ne Yapar |
|---|---|---|
| 9.1.1 Konveyör ikizi | [`config/gemini_params.yaml`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/gemini_robotics_ros/config/gemini_params.yaml) | Tarama pozlarını, eşikleri ve kip ayrımını tanımlar. |
| 9.1.2 Vakum sürücüsü | [`gemini_robotics_ros/vacuum.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/gemini_robotics_ros/gemini_robotics_ros/vacuum.py) | Komutu yollar, simülasyonda koparılabilir eklemi kurar. |
| 9.3.1 Derinlik render | [`gemini_robotics_ros/depth_render.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/gemini_robotics_ros/gemini_robotics_ros/depth_render.py) | Derinliği sabit metrik aralıkta görüntüye çevirir. |
| 9.3.2 Zaman damgalı dönüşüm | [`gemini_robotics_ros/locator.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/gemini_robotics_ros/gemini_robotics_ros/locator.py) | Pikseli ölçüm anındaki dönüşümle dünyaya taşır. |
| 9.3.3 Kavrama pozu ve normal | [`gemini_robotics_ros/grasp.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/gemini_robotics_ros/gemini_robotics_ros/grasp.py) | Yüzey normalinden kavrama yönelimini üretir. |
| 9.4.1 – 9.4.4 Görev ve parça | [`pick_place_node.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/gemini_robotics_ros/gemini_robotics_ros/pick_place_node.py), [`payload.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/gemini_robotics_ros/gemini_robotics_ros/payload.py) | Aşamaları yürütür, parçayı sahneye ekler, koşuyu yazar. |
