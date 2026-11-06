# ch10_screwdriving — Bölüm 10: İnsan–Robot İş Birlikli Hassas Vidalama Otomasyonu

Bölümün kodu bu klasördeki `ch10_screwdriving` ROS 2 paketindedir. Vidalama senaryosu
[`human_robot_collaboration_scenario.py`](ch10_screwdriving/human_robot_collaboration_scenario.py)
dosyasıdır ve
[`launch/human_robot_collaboration_scenario.launch.py`](launch/human_robot_collaboration_scenario.launch.py)
ile başlatılır. Hareket komutları ortak [`book_pymoveit2`](../common/book_pymoveit2)
kütüphanesi üzerinden MoveIt 2'ye gider.

> **Tam kurulum için araştırma deposu gerekir.** Paket bu depoda derlenir; ancak
> çalıştırmak için robot hücresinin MoveIt 2 yapılandırması ve denetleyicileri
> gerekir. Bunlar
> [ESOGU-ASRLAB/ESOGU-HILTest-DualRobot](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot)
> deposundadır; eklemek için ana [README](../README.md)'deki
> **Bölüm 7–10: araştırma deposunun eklenmesi** adımlarını izleyin.

## Çalıştırma

```bash
# 1. terminal: robot hücresi (sahte donanım, OnRobot 2FG7 tutuculu yapılandırma)
$ ros2 launch my_robot_cell_control hil_test_whole_unified.launch.py \
    use_fake_hardware:=true use_mock_hardware:=true fake_sensor_commands:=true \
    use_gripper:=true

# 2. terminal: vidalama senaryosu
$ ros2 launch ch10_screwdriving human_robot_collaboration_scenario.launch.py
```

Sahte donanımda GPIO yoktur; senaryo bunu kendisi algılar (`gpio_mode:=auto`) ve
operatör butonunu beklemeden ilerler.

Senaryo her kol hareketini ilk koşuda planlayıp `~/ros2_book_output/ch10/trajectories/hrc_screwing.json`
dosyasına kaydeder; sonraki koşular planlama yapmadan aynı yolları oynatır. Başka bir klasör
için `trajectory_dir:=<klasör>` verilir; dosya silinirse yollar yeniden kaydedilir.

## Bölümde anlatılanların depodaki karşılıkları

| Bölümde Anlatılan | Depodaki Karşılığı | Ne Yapar |
|---|---|---|
| 10.3.2 Koordinat çerçeveleri ve pozlar | [`human_robot_collaboration_scenario.py`](ch10_screwdriving/human_robot_collaboration_scenario.py) | Öğretilen eklem açılarını yaklaşma–işlem çiftleri olarak tutar. |
| 10.4.1 Aletin bağlı gövdeye dönüşü | aynı dosya, `attach_screwdriver()` ve `detach_screwdriver()` | Aleti silindir olarak ekler ve tutucu bağlantısına iliştirir. |
| 10.4.2 Çarpışma geometrisi ve kablo | aynı dosya, `attach_screwdriver()` içindeki `touch_links` listesi | Tutucu ile alet arasındaki çarpışma denetimini kapatır. |
| 10.6.1 Mevcut turun aşamaları | aynı dosya, `safe_joint_sequence()` | Listeyi sırayla yürütür, sözlük elemanlarını komut sayar. |
