# ch10_screwdriving — Bölüm 10: İnsan–Robot İş Birlikli Hassas Vidalama Otomasyonu

> Bu klasörde derlenecek bir paket yoktur. Bölüm, gerçek bir robot hücresine ve
> onlarca sürücü paketine dayandığı için kodu
> [ESOGU-ASRLAB/ESOGU-HILTest-DualRobot](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master) araştırma deposundadır (master dalı).
> Depoyu çalışma alanınıza eklemek için ana [README](../README.md)'deki
> **Bölüm 7–10: araştırma deposunun eklenmesi** adımlarını izleyin.

Vidalama senaryosu [`pymoveit2_real`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/tree/master/pymoveit2_real) paketinin örnekler
dizinindeki [`examples/human_robot_collaboration_scenario.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/pymoveit2_real/examples/human_robot_collaboration_scenario.py)
dosyasıdır ve [`launch/human_robot_collaboration_scenario.launch.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/pymoveit2_real/launch/human_robot_collaboration_scenario.launch.py)
ile başlatılır. Paket hakkında ayrıntılar [README](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/pymoveit2_real/README.md) dosyasındadır.

## Bölümde anlatılanların depodaki karşılıkları

| Bölümde Anlatılan | Depodaki Karşılığı | Ne Yapar |
|---|---|---|
| 10.3.2 Koordinat çerçeveleri ve pozlar | [`examples/human_robot_collaboration_scenario.py`](https://github.com/ESOGU-ASRLAB/ESOGU-HILTest-DualRobot/blob/master/pymoveit2_real/examples/human_robot_collaboration_scenario.py) | Öğretilen eklem açılarını yaklaşma–işlem çiftleri olarak tutar. |
| 10.4.1 Aletin bağlı gövdeye dönüşü | aynı dosya, iliştirme ve ayırma işlevleri | Aleti silindir olarak ekler ve tutucu bağlantısına iliştirir. |
| 10.4.2 Çarpışma geometrisi ve kablo | aynı dosya, dokunma bağlantıları listesi | Tutucu ile alet arasındaki çarpışma denetimini kapatır. |
| 10.6.1 Mevcut turun aşamaları | aynı dosya, dizi yürütme işlevi | Listeyi sırayla yürütür, sözlük elemanlarını komut sayar. |
