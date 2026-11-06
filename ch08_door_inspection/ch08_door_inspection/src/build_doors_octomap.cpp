/**
 * build_doors_octomap.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * ROS 2 Humble — ARABA KAPISI muayenesi için OctoMap oluşturucu (Bölüm 8).
 *
 * Şase oluşturucusuyla (pcd2octomap_builder/build_octomap) aynı iki aşamalı yöntem:
 *
 *   AŞAMA 1 — İnanç haritası (belief map)
 *     <parts_dir> altındaki .pcd dosyaları (doors_parts_prep'in ürettiği kapı
 *     yamaları) yüklenir.
 *     Her yamaya farklı renk atanır ve ColorOcTree'ye updateNode(false) ile BOŞ
 *     (free) olarak eklenir: "var olan ve görülmesi gereken yüzey" referansı budur.
 *     → beliefMap_doors_<mode>.ot
 *
 *   AŞAMA 2 — Doluluk haritası (occupancy map)
 *     İki kolun yakaladığı bulutlar inanç haritasının üstüne damgalanır. insertRay
 *     KULLANILMAZ (ışın boyunca boş hücre açmak inanç haritasını bozar); yalnızca
 *     uç nokta dolu yapılır ve inanç rengi korunur.
 *     → occupancyMap_doors_<mode>.ot
 *
 * Kapı verisi şaseden ÜÇ noktada ayrılır; bu dosyanın var olma sebebi budur:
 *
 *   1. BİRİM. Şase parçaları CAD'den mm geldiği için şase oluşturucusu noktaları
 *      sabit olarak 1000'e böler. Kapı STL'leri zaten METRE'dir (URDF onları
 *      scale="1 1 1" ile yükler). Burada bölme --parts_scale ile parametredir ve
 *      kapılar için varsayılanı 1.0'dır. Şase oluşturucusunu kapılara vermek
 *      kapıları 1.2 mm'lik bir zerreye indirirdi.
 *   2. VERİ KÖKÜ. Kapı yakalamaları ~/ros2_book_output/ch08/pcds/{sim,real}_pcds
 *      altındadır; şase köküne dokunulmaz.
 *   3. RAPOR. Kapılar 2 STL olduğu için parça başına rapor iki satır olurdu.
 *      doors_parts_prep her kapıyı yamalara böler, bu oluşturucu da yamaları
 *      "__" öncesindeki ada göre gruplayıp hem yama hem kapı toplamını yazar.
 *
 * Kullanım (hepsi isteğe bağlı; argümansız çalışır):
 *   ros2 run ch08_door_inspection build_doors_octomap \
 *     [--mode sim|real] [--data_dir <pcds/sim_pcds>] \
 *     [--parts_dir <~/.cache/ch08_door_inspection/door_parts>] [--parts_scale 1.0] \
 *     [--belief_out ...] [--occ_out ...] \
 *     [--res 0.02] [--z_min -10.0] [--use_ur 1] [--use_kawasaki 1] \
 *     [--snap 0] [--snap_radius 1]
 *
 * GERÇEK VERİ İÇİN NOT (6 Ağu 2026 koşusundan sonra eklendi):
 *   Bu oluşturucu bir noktayı ancak TAM olarak bir inanç vokseline düştüğünde sayar.
 *   Benzetimde yakalanan bulutun modele uzaklığı medyan 3 mm olduğu için bu sorunsuz
 *   geçer; gerçekte 38 mm olduğu için nokta 2 cm'lik hücreyi ıskalıyor ve "hiç
 *   taranmamış" gibi görünüyor. Kapının ALTINDA hata ~8 cm'e çıkıyor, üstte ~1-2 cm
 *   kalıyor; yani kapı modeldekinden yatık duruyor. İki kol bunu birbirinden
 *   BAĞIMSIZ olarak aynı işaret ve büyüklükte ölçüyor (UR -81.4 mm, Kawasaki
 *   -81.5 mm); bu yüzden kaynak robot kalibrasyonu değil kapının kendisi.
 *   --snap_radius bunu RAPORDA telafi eder, gerçekte düzeltmez.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// Octomap
#include <octomap/ColorOcTree.h>
#include <octomap/octomap.h>

// Eigen
#include <Eigen/Geometry>

namespace fs = std::filesystem;
using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

// ─────────────────────────────────────────────────────────────────────────────
// Voksel anahtarı — OctoMap çözünürlüğüne göre tam sayı ızgara
// ─────────────────────────────────────────────────────────────────────────────
struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        size_t h = std::hash<int>()(k.x);
        h ^= std::hash<int>()(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
using VoxelPartMap = std::unordered_map<VoxelKey, int, VoxelKeyHash>;

static VoxelKey to_voxel_key(const octomap::point3d& pt, double res) {
    return { static_cast<int>(std::floor(pt.x() / res)),
             static_cast<int>(std::floor(pt.y() / res)),
             static_cast<int>(std::floor(pt.z() / res)) };
}

static octomap::point3d voxel_center(const VoxelKey& k, double res) {
    return octomap::point3d(static_cast<float>((k.x + 0.5) * res),
                            static_cast<float>((k.y + 0.5) * res),
                            static_cast<float>((k.z + 0.5) * res));
}

// ─────────────────────────────────────────────────────────────────────────────
// Renk paleti — pcd2octomap_builder ile AYNI sıra; böylece iki işin octovis
// görüntüleri yan yana karşılaştırılabilir kalır.
// ─────────────────────────────────────────────────────────────────────────────
static const uint8_t COLOR_PALETTE[][3] = {
    { 85,  26, 139}, {144, 238, 144}, {  0,   0, 139}, {190, 170,   8},
    {150,  10,   0}, {135, 206, 250}, {255, 165,   0}, {255, 182, 193},
    {  0, 100,   0}, {255,  20, 147}, {139,  71,  38}, {  0, 245, 255},
    {238, 149, 114}, {  0, 128, 128}, { 67, 205, 128}, {128,   0,   0},
    {220,  20,  60}, {255, 127,  80}, {205,  92,  92}, {255, 215,   0},
    {184, 134,  11}, {189, 183, 107}, {128, 128,   0}, { 85, 107,  47},
    {127, 255,   0}, { 34, 139,  34}, {143, 188, 143}, {  0, 250, 154},
    {102, 205, 170}, { 70, 130, 180}, { 25,  25, 112}, { 65, 105, 225},
    {138,  43, 226}, { 72,  61, 139}, {106,  90, 205}, {147, 112, 219},
    {139,   0, 139}, {153,  50, 204}, {221, 160, 221}, {218, 112, 214},
    {255,  20, 147}, {255, 182, 193}, {245, 222, 179}, {139,  69,  19},
    {210, 105,  30}, {200, 105,  30}, {195, 105,  30}, {190, 105,  30},
    {185, 105,  30}, {180, 105,  30},
};
static const int PALETTE_SIZE =
    static_cast<int>(sizeof(COLOR_PALETTE) / sizeof(COLOR_PALETTE[0]));

// ─────────────────────────────────────────────────────────────────────────────
// Yapılandırma
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    std::string mode = "sim";       // sim | real  — yalnızca varsayılan yolları seçer
    std::string parts_dir;
    std::string data_dir;           // <data_dir>/{ur_data,kawasaki_data}/{pcds,poses}
    std::string belief_out;
    std::string occ_out;
    double      res         = 0.02;
    float       z_min       = -10.0f;
    // Kapı yamaları METRE cinsinden yazılır; şase oluşturucusundaki /1000 burada YOK.
    double      parts_scale = 1.0;
    bool        use_ur       = true;
    bool        use_kawasaki = true;
    // 0 = şase davranışının birebir aynısı (nokta tam olarak bir inanç vokseline
    // düşmelidir). 1 = düşmezse komşulara bakılır; gerçek veride kalibrasyon
    // kayması yüzünden yüzeyin yanına düşen noktaları kurtarır, ama kaplamayı
    // iyimser gösterir; bu yüzden varsayılan kapalıdır.
    bool        snap = false;
    // Snap arama yarıçapı VOKSEL cinsinden; metre karşılığı snap_radius * res.
    // 1 = eski 26-komşu davranışı (res 0.02'de ±2 cm).
    //
    // 6 Ağu 2026 gerçek koşusunda ölçülen kapı hizalama hatası kapının altında
    // ~8 cm, ortasında ~1-2 cm idi; yani +-2 cm alt tarafı kurtarmıyordu.
    // Aynı veri üzerinde ölçülen kaplama (res 0.02):
    //     yarıçap 0 -> gerçek %38.1   benzetim %86.2
    //     yarıçap 1 -> gerçek %48.1   benzetim %87.4
    //     yarıçap 2 -> gerçek %55.0   benzetim %87.4     <- kazanç burada doyuyor
    //     yarıçap 4 -> gerçek %58.9   benzetim %87.5
    // Benzetimin neredeyse hiç değişmemesi önemli: benzetimin hatası zaten 3 mm olduğu
    // için snap ona bir şey katmıyor, yalnızca gerçekteki kaymayı telafi ediyor.
    // Yani bu ayar iki kipi birbirine yaklaştırırken karşılaştırılabilirliği
    // bozmuyor. Yine de bu bir RAPORLAMA telafisidir: hizalama hatasının kendisi
    // duruyor, yalnızca artık voksel ıskası olarak sayılmıyor.
    int         snap_radius = 1;

    // ── Bulut hizalama düzeltmesi (--cloud_fix "roll,pitch,yaw,x,y,z") ─────────
    // Yakalanan buluta voksel anahtarı hesaplanmadan ÖNCE uygulanan katı dönüşüm:
    //     X' = Rz(yaw) * Ry(pitch) * Rx(roll) * X + t
    // rpy DERECE, xyz METRE. URDF ile aynı ZYX sırası; böylece buradaki sayılar
    // ileride kapının URDF pozuna doğrudan taşınabilir.
    //
    // NEDEN BULUTA, MODELE DEĞİL: yanlış olan model, ama inanç haritası hem voksel
    // ızgarasını hem de kaplama paydasını tanımlıyor. Modeli oynatmak paydayı ve
    // yama sınırlarını her koşuda değiştirirdi; buluta uygulamak matematiksel
    // olarak tersidir ve payda sabit kaldığı için koşular karşılaştırılabilir kalır.
    //
    // 7 Ağu 2026'da noktadan düzleme ICP ile ölçüldü (%80 kırpma, 183 740 nokta):
    //     rpy = -0.740, -5.128, +0.347 derece    xyz = +0.109, +0.002, +0.061 m
    // Baskın terim Y ekseni etrafında -5.1 derece: kapı öne yatık duruyor. İki
    // kapı AYRI AYRI uydurulunca da aynı açıyı veriyor (-5.04 ve -4.80) ve ayrı
    // uydurma medyan hatayı genel uydurmadan iyileştirmiyor (18.5/17.9 ile 18.6
    // mm); bu yüzden tek genel dönüşüm kullanılıyor.
    // Medyan model mesafesi: 41.5 mm -> 18.6 mm.
    //
    // YALNIZCA GERÇEK VERİ İÇİN. Benzetim zaten hizalı; aynı düzeltmeyi benzetime uygulamak
    // kaplamayı %86.2'den %28.1'e düşürüyor (ölçüldü). Varsayılan kapalı.
    bool        cloud_fix = false;
    double      fix_rpy[3] = {0.0, 0.0, 0.0};   // derece
    double      fix_xyz[3] = {0.0, 0.0, 0.0};   // metre
};

static std::string expand_tilde(const std::string& path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

static bool parse_bool(const std::string& v) {
    return !(v == "0" || v == "false" || v == "False" || v == "no");
}

// "r,p,y,x,y,z" → 6 sayı. Boşlukla ayırmak tek argümana sığmadığı için virgül.
static bool parse_six(const std::string& v, double* rpy, double* xyz) {
    std::vector<double> vals;
    std::string tok;
    for (size_t i = 0; i <= v.size(); ++i) {
        if (i == v.size() || v[i] == ',') {
            if (tok.empty()) return false;
            try { vals.push_back(std::stod(tok)); } catch (...) { return false; }
            tok.clear();
        } else if (!std::isspace(static_cast<unsigned char>(v[i]))) {
            tok += v[i];
        }
    }
    if (vals.size() != 6) return false;
    for (int i = 0; i < 3; ++i) { rpy[i] = vals[i]; xyz[i] = vals[i + 3]; }
    return true;
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc - 1; ++i) {
        std::string k = argv[i], v = argv[i + 1];
        if      (k == "--mode")         cfg.mode         = v;
        else if (k == "--parts_dir")    cfg.parts_dir    = expand_tilde(v);
        else if (k == "--data_dir")     cfg.data_dir     = expand_tilde(v);
        else if (k == "--belief_out")   cfg.belief_out   = expand_tilde(v);
        else if (k == "--occ_out")      cfg.occ_out      = expand_tilde(v);
        else if (k == "--res")          cfg.res          = std::stod(v);
        else if (k == "--z_min")        cfg.z_min        = std::stof(v);
        else if (k == "--parts_scale")  cfg.parts_scale  = std::stod(v);
        else if (k == "--use_ur")       cfg.use_ur       = parse_bool(v);
        else if (k == "--use_kawasaki") cfg.use_kawasaki = parse_bool(v);
        else if (k == "--snap")         cfg.snap         = parse_bool(v);
        else if (k == "--snap_radius")  cfg.snap_radius  = std::stoi(v);
        else if (k == "--cloud_fix") {
            if (!parse_six(v, cfg.fix_rpy, cfg.fix_xyz)) {
                std::cerr << "❌ --cloud_fix formati: \"roll,pitch,yaw,x,y,z\" "
                             "(rpy derece, xyz metre). Verilen: " << v << "\n";
                std::exit(1);
            }
            // Tam sıfır verilmesi "kapalı" demektir; kimse birim dönüşüm için
            // hesaplama yaptırmak istemez.
            cfg.cloud_fix = !(cfg.fix_rpy[0] == 0 && cfg.fix_rpy[1] == 0 &&
                              cfg.fix_rpy[2] == 0 && cfg.fix_xyz[0] == 0 &&
                              cfg.fix_xyz[1] == 0 && cfg.fix_xyz[2] == 0);
        }
    }

    if (cfg.snap_radius < 1) {
        std::cerr << "❌ --snap_radius en az 1 olmali (verilen: " << cfg.snap_radius
                  << "). Snap'i kapatmak icin --snap 0 kullan.\n";
        std::exit(1);
    }

    if (cfg.mode != "sim" && cfg.mode != "real") {
        std::cerr << "❌ --mode yalnizca 'sim' veya 'real' olabilir (verilen: "
                  << cfg.mode << ")\n";
        std::exit(1);
    }

    const std::string home = expand_tilde("~");
    if (cfg.parts_dir.empty())
        cfg.parts_dir = home + "/.cache/ch08_door_inspection/door_parts";
    if (cfg.data_dir.empty())
        cfg.data_dir = home + "/ros2_book_output/ch08/pcds/" + cfg.mode + "_pcds";
    if (cfg.belief_out.empty())
        cfg.belief_out = cfg.data_dir + "/beliefMap_doors_" + cfg.mode + ".ot";
    if (cfg.occ_out.empty())
        cfg.occ_out = cfg.data_dir + "/occupancyMap_doors_" + cfg.mode + ".ot";

    if (!cfg.use_ur && !cfg.use_kawasaki) {
        std::cerr << "❌ Hem --use_ur hem --use_kawasaki kapali; islenecek veri yok.\n";
        std::exit(1);
    }

    // Benzetim bulutu modelden zaten medyan 3 mm sapıyor; ona gerçek veri düzeltmesi
    // uygulamak hizalamayı BOZAR. Ölçüldü: kaplama %86.2 -> %28.1. Durdurmuyoruz
    // (birinin bilerek denemesi meşrudur), ama görülmemesini imkânsız hâle getiriyoruz.
    if (cfg.cloud_fix && cfg.mode == "sim") {
        std::cerr << "\n⚠️  ⚠️  --cloud_fix SIM modunda aciik!\n"
                  << "    Duzeltme gercek verideki kapi kaymasi icin olculdu. Sim\n"
                  << "    zaten hizali oldugu icin bu onu BOZAR (olculdu: %86.2 ->\n"
                  << "    %28.1). Sim'i duzeltmesiz calistir.\n\n";
    }
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parça PCD'leri. Ad düzeni "<kapı>__y<j>z<j>.pcd"; rapor gruplaması "__"
// öncesindeki kapı adına göre yapılır.
// ─────────────────────────────────────────────────────────────────────────────
static std::string door_of(const fs::path& p) {
    std::string stem = p.stem().string();
    size_t pos = stem.find("__");
    return (pos == std::string::npos) ? stem : stem.substr(0, pos);
}

static std::vector<fs::path> collect_parts(const std::string& dir) {
    std::vector<fs::path> paths;
    if (!fs::is_directory(dir)) return paths;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".pcd") paths.push_back(e.path());
    // Alfabetik sıra yama ızgarasını kapı kapı, sütun sütun sıralar.
    std::sort(paths.begin(), paths.end());
    return paths;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tarama (PCD + poz) eşleştirme — iki kolun dosya adları çakışır (ikisinde de
// 1.pcd, 2.pcd ...); bu yüzden dosya kökü kol etiketiyle benzersizleştirilir.
// ─────────────────────────────────────────────────────────────────────────────
struct ScanPair {
    fs::path    pcd_path;
    fs::path    pose_path;
    std::string stem;
};

static std::vector<ScanPair> match_scans_for_robot(const std::string& robot_dir,
                                                   const std::string& robot_tag) {
    std::vector<ScanPair> pairs;
    fs::path pcd_dir  = fs::path(robot_dir) / "pcds";
    fs::path pose_dir = fs::path(robot_dir) / "poses";

    if (!fs::is_directory(pcd_dir) || !fs::is_directory(pose_dir)) {
        std::cerr << "⚠️  " << robot_tag << ": klasor yok (" << pcd_dir.string()
                  << ") — atlaniyor\n";
        return pairs;
    }

    std::map<std::string, fs::path> pose_map;
    for (const auto& e : fs::directory_iterator(pose_dir))
        if (e.path().extension() == ".txt")
            pose_map[e.path().stem().string()] = e.path();

    for (const auto& e : fs::directory_iterator(pcd_dir)) {
        if (e.path().extension() != ".pcd") continue;
        std::string stem = e.path().stem().string();
        auto it = pose_map.find(stem);
        if (it != pose_map.end())
            pairs.push_back({e.path(), it->second, robot_tag + ":" + stem});
        else
            std::cerr << "⚠️  " << robot_tag << " pose bulunamadi: " << stem
                      << " — atlaniyor\n";
    }

    std::sort(pairs.begin(), pairs.end(), [](const ScanPair& a, const ScanPair& b) {
        auto num = [](const std::string& s) -> int {
            size_t pos = s.find_last_of(':');
            try { return std::stoi(pos == std::string::npos ? s : s.substr(pos + 1)); }
            catch (...) { return 0; }
        };
        return num(a.stem) < num(b.stem);
    });
    return pairs;
}

static std::vector<ScanPair> collect_scans(const Config& cfg) {
    std::vector<ScanPair> scans;
    if (cfg.use_ur) {
        auto ur = match_scans_for_robot(cfg.data_dir + "/ur_data", "ur");
        std::cout << "  UR taramasi       : " << ur.size() << "\n";
        scans.insert(scans.end(), ur.begin(), ur.end());
    }
    if (cfg.use_kawasaki) {
        auto kw = match_scans_for_robot(cfg.data_dir + "/kawasaki_data", "kawa");
        std::cout << "  Kawasaki taramasi : " << kw.size() << "\n";
        scans.insert(scans.end(), kw.begin(), kw.end());
    }
    return scans;
}

// ─────────────────────────────────────────────────────────────────────────────
// Poz TXT → algılayıcı konumu.  Biçim: tx ty tz qx qy qz qw
// PCD'ler zaten world çerçevesinde kaydedilir; poz yalnızca sensorOrigin içindir.
// ─────────────────────────────────────────────────────────────────────────────
struct SensorPose { Eigen::Vector3d t; Eigen::Quaterniond q; };

static SensorPose load_pose(const fs::path& p) {
    std::ifstream f(p);
    if (!f.is_open()) throw std::runtime_error("TXT acilamadi: " + p.string());
    double tx, ty, tz, qx, qy, qz, qw;
    if (!(f >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
        throw std::runtime_error("TXT parse hatasi: " + p.string());
    SensorPose pose;
    pose.t = {tx, ty, tz};
    pose.q = Eigen::Quaterniond(qw, qx, qy, qz);   // Eigen: (w, x, y, z)
    pose.q.normalize();
    return pose;
}

// ─────────────────────────────────────────────────────────────────────────────
// AŞAMA 1 — İnanç haritası
// ─────────────────────────────────────────────────────────────────────────────
static void build_belief_map(octomap::ColorOcTree* tree,
                             const std::vector<fs::path>& parts,
                             double parts_scale,
                             std::vector<int>& belief_counts,
                             VoxelPartMap& voxel_part_map) {
    std::cout << "\n━━━  AŞAMA 1: Belief Map  ━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    belief_counts.assign(parts.size(), 0);
    const double res = tree->getResolution();

    for (size_t i = 0; i < parts.size(); ++i) {
        PointCloud cloud;
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(parts[i].string(), cloud) == -1) {
            std::cerr << "  ⚠️  Yuklenemedi: " << parts[i].filename() << "\n";
            continue;
        }

        const int idx = static_cast<int>(i) % PALETTE_SIZE;
        const uint8_t R = COLOR_PALETTE[idx][0];
        const uint8_t G = COLOR_PALETTE[idx][1];
        const uint8_t B = COLOR_PALETTE[idx][2];

        std::unordered_set<size_t> part_voxels;
        VoxelKeyHash hasher;

        for (const auto& p : cloud.points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                continue;
            // parts_scale: kapılar için 1.0 (metre). Şase parçaları mm olduğu için
            // orada 0.001 verilir.
            octomap::point3d pt(static_cast<float>(p.x * parts_scale),
                                static_cast<float>(p.y * parts_scale),
                                static_cast<float>(p.z * parts_scale));
            // false = BOŞ  →  octovis'te gri; "görülmesi gereken ama henüz
            // görülmemiş yüzey" anlamı buradan gelir.
            if (octomap::ColorOcTreeNode* node = tree->updateNode(pt, false))
                node->setColor(R, G, B);

            VoxelKey vk = to_voxel_key(pt, res);
            voxel_part_map[vk] = static_cast<int>(i);
            part_voxels.insert(hasher(vk));
        }

        belief_counts[i] = std::max<int>(1, static_cast<int>(part_voxels.size()));
        std::cout << "  " << parts[i].filename().string() << "  →  " << cloud.size()
                  << " nokta  " << part_voxels.size() << " voxel\n";
    }

    int total = 0;
    for (int c : belief_counts) total += c;
    std::cout << "  ─────────────────────────────────────────────────\n";
    std::cout << "  Belief map toplam voxel : " << total << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// AŞAMA 2 — Doluluk haritası
// ─────────────────────────────────────────────────────────────────────────────
static void build_occupancy_map(octomap::ColorOcTree* tree,
                                const std::vector<ScanPair>& scans,
                                const VoxelPartMap& voxel_part_map,
                                std::vector<int>& occ_counts,
                                float z_min, bool snap, int snap_radius,
                                bool cloud_fix, const double* fix_rpy,
                                const double* fix_xyz) {
    std::cout << "\n━━━  AŞAMA 2: Occupancy Map  ━━━━━━━━━━━━━━━━━━━━━━\n";

    const double res = tree->getResolution();

    // Hizalama düzeltmesi. URDF ile aynı ZYX sırası: Rz(yaw)*Ry(pitch)*Rx(roll).
    Eigen::Matrix3d fixR = Eigen::Matrix3d::Identity();
    Eigen::Vector3d fixT = Eigen::Vector3d::Zero();
    if (cloud_fix) {
        const double d2r = M_PI / 180.0;
        fixR = (Eigen::AngleAxisd(fix_rpy[2] * d2r, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(fix_rpy[1] * d2r, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(fix_rpy[0] * d2r, Eigen::Vector3d::UnitX()))
                   .toRotationMatrix();
        fixT = Eigen::Vector3d(fix_xyz[0], fix_xyz[1], fix_xyz[2]);
        std::cout << "  Bulut duzeltmesi UYGULANIYOR: rpy(deg) " << fix_rpy[0] << " "
                  << fix_rpy[1] << " " << fix_rpy[2] << "  xyz(m) " << fix_xyz[0]
                  << " " << fix_xyz[1] << " " << fix_xyz[2] << "\n"
                  << "    (yakalanan buluta uygulanir; belief map'e DOKUNULMAZ, "
                     "boylece payda sabit kalir)\n";
    }

    // Snap ofsetleri BİR KEZ, mesafeye göre sıralı üretilir. Sıra kritik: eski kod
    // dx,dy,dz'yi -1..1 tarayıp İLK bulduğu komşuya çekiyordu, bu da noktaları
    // sistematik olarak -x,-y,-z yönüne yanlı kaydırıyordu. Yarıçap 1'de etkisi
    // küçüktü; 2 ve üstünde nokta gerçekten en yakın olmayan bir yamaya
    // yazılabilirdi ve yama bazlı rapor yanlış çıkardı. Mesafeye göre sıralayıp
    // ilk eşleşmede durmak EN YAKIN belief voxelini garanti eder.
    std::vector<VoxelKey> snap_offsets;
    if (snap) {
        for (int dx = -snap_radius; dx <= snap_radius; ++dx)
        for (int dy = -snap_radius; dy <= snap_radius; ++dy)
        for (int dz = -snap_radius; dz <= snap_radius; ++dz) {
            if (!dx && !dy && !dz) continue;      // merkez zaten denendi
            snap_offsets.push_back({dx, dy, dz});
        }
        std::sort(snap_offsets.begin(), snap_offsets.end(),
                  [](const VoxelKey& a, const VoxelKey& b) {
                      return (a.x*a.x + a.y*a.y + a.z*a.z) <
                             (b.x*b.x + b.y*b.y + b.z*b.z);
                  });
        std::cout << "  Snap: yaricap " << snap_radius << " voxel = "
                  << (snap_radius * res * 100.0) << " cm, "
                  << snap_offsets.size() << " komsu, en yakin once\n";
    }

    long total_ins = 0, total_skip = 0, total_snap = 0;
    double total_snap_dist = 0.0;   // kaplamanın ne kadar "ödünç" alındığını gösterir
    std::vector<std::unordered_set<size_t>> occ_voxel_sets(occ_counts.size());
    VoxelKeyHash vk_hasher;

    for (size_t i = 0; i < scans.size(); ++i) {
        const auto& scan = scans[i];
        std::cout << "  [" << (i + 1) << "/" << scans.size() << "]  " << scan.stem << "\n";

        PointCloud::Ptr cs = std::make_shared<PointCloud>();
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(scan.pcd_path.string(), *cs) == -1) {
            std::cerr << "    ⚠️  PCD yuklenemedi — atlaniyor\n";
            continue;
        }

        // Poz yalnızca kaydın bütünlüğünü doğrulamak için okunur: bulut zaten world
        // çerçevesinde yazılmıştır, dönüşüm UYGULANMAZ (şase işiyle aynı sözleşme).
        try { (void)load_pose(scan.pose_path); }
        catch (const std::exception& e) {
            std::cerr << "    ⚠️  " << e.what() << " — atlaniyor\n";
            continue;
        }

        long ins = 0, skip = 0, snapped = 0;
        double snap_dist = 0.0;
        for (const auto& p : cs->points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                { ++skip; continue; }

            double px = p.x, py = p.y, pz = p.z;
            if (cloud_fix) {
                Eigen::Vector3d q = fixR * Eigen::Vector3d(px, py, pz) + fixT;
                px = q.x(); py = q.y(); pz = q.z();
            }
            // z_min düzeltmeden SONRA uygulanır: eşik dünya zemininde tanımlıdır ve
            // düzeltme bir noktayı eşiğin altına/üstüne taşıyabilir.
            if (pz < z_min) { ++skip; continue; }

            octomap::point3d pt(static_cast<float>(px), static_cast<float>(py),
                                static_cast<float>(pz));
            VoxelKey vk = to_voxel_key(pt, res);

            auto vit = voxel_part_map.find(vk);
            if (vit == voxel_part_map.end() && snap) {
                // Gerçek veride hem el-göz kalibrasyonu hem de kapının modeldekinden
                // farklı duruşu noktayı yüzeyin yanına düşürür. En yakın belief
                // voxeli arama yarıçapı içindeyse nokta oraya çekilir.
                for (const VoxelKey& off : snap_offsets) {
                    VoxelKey cand = {vk.x + off.x, vk.y + off.y, vk.z + off.z};
                    auto nit = voxel_part_map.find(cand);
                    if (nit != voxel_part_map.end()) {
                        vit = nit;
                        vk  = cand;
                        pt  = voxel_center(vk, res);
                        ++snapped;
                        snap_dist += std::sqrt(static_cast<double>(
                            off.x*off.x + off.y*off.y + off.z*off.z)) * res;
                        break;      // ofsetler sıralı, ilk eşleşme en yakını
                    }
                }
            }

            if (vit == voxel_part_map.end()) {
                // İnanç haritası dışı nokta — kapı değil (zemin, şase, AGV ...), atla.
                ++skip;
                continue;
            }

            octomap::ColorOcTreeNode* existing = tree->search(pt);
            octomap::ColorOcTreeNode::Color color =
                existing ? existing->getColor() : octomap::ColorOcTreeNode::Color(255, 255, 255);

            const int pid = vit->second;
            if (occ_voxel_sets[pid].insert(vk_hasher(vk)).second)
                occ_counts[pid]++;

            // insertRay KULLANILMIYOR: başlangıç→uç arasını boş yapmak inanç
            // haritasını bozar. Yalnızca uç nokta dolu; renk inanç haritasından korunur.
            if (octomap::ColorOcTreeNode* node = tree->updateNode(pt, true))
                node->setColor(color.r, color.g, color.b);
            ++ins;
        }

        std::cout << "    " << cs->size() << " nokta  |  " << ins << " kapi ici eklendi  |  "
                  << skip << " kapi disi/NaN atlandi";
        if (snap) {
            std::cout << "  |  " << snapped << " komsu voxele cekildi";
            if (snapped) std::cout << " (ort. " << (snap_dist / snapped * 1000.0) << " mm)";
        }
        std::cout << "\n";
        total_ins  += ins;
        total_skip += skip;
        total_snap += snapped;
        total_snap_dist += snap_dist;
    }

    std::cout << "  ─────────────────────────────────────────────────\n";
    std::cout << "  Toplam eklenen : " << total_ins  << "\n";
    std::cout << "  Toplam atlanan : " << total_skip << "\n";
    if (snap) {
        std::cout << "  Toplam cekilen : " << total_snap;
        if (total_ins > 0)
            std::cout << "  (eklenenlerin %" << (100.0 * total_snap / total_ins) << "'i)";
        std::cout << "\n";
        if (total_snap > 0) {
            std::cout << "  Ort. cekme mesafesi : "
                      << (total_snap_dist / total_snap * 1000.0) << " mm\n";
            // Bu iki satır raporun ne kadarının telafiyle kazanıldığını gösterir.
            // Büyük bir çekme oranı + büyük bir mesafe = hizalama sorunu duruyor,
            // yalnızca voksel ıskası olarak görünmüyor demektir.
            std::cout << "  ⚠️  Bu noktalar modeldeki yuzeye TAM oturmadi; kaplama\n"
                      << "      sayisi bu kadar telafiyle elde edildi.\n";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kaplama raporu — yama satırları + kapı toplamı + genel toplam
// ─────────────────────────────────────────────────────────────────────────────
static void print_coverage(const std::vector<fs::path>& parts,
                           const std::vector<int>& belief_counts,
                           const std::vector<int>& occ_counts) {
    std::cout << "\n━━━  COVERAGE RAPORU  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    // parts alfabetik sıralı olduğu için aynı kapının yamaları bitişiktir.
    long g_belief = 0, g_occ = 0;
    size_t i = 0;
    while (i < parts.size()) {
        const std::string door = door_of(parts[i]);
        long d_belief = 0, d_occ = 0;
        int n_miss = 0;

        std::cout << "\n  ▸ " << door << "\n";
        while (i < parts.size() && door_of(parts[i]) == door) {
            const int b = belief_counts[i], o = occ_counts[i];
            const double pct = b > 0 ? 100.0 * o / b : 0.0;
            std::cout << (o > 0 ? "    ✅ " : "    ❌ ")
                      << parts[i].stem().string() << "  %" << static_cast<int>(pct)
                      << "  (" << o << " / " << b << ")"
                      << (o > 0 ? "" : "  MISSING") << "\n";
            if (o == 0) ++n_miss;
            d_belief += b;
            d_occ    += o;
            ++i;
        }
        const double d_pct = d_belief > 0 ? 100.0 * d_occ / d_belief : 0.0;
        std::cout << "    ── " << door << " toplam: %" << d_pct
                  << "  (" << d_occ << " / " << d_belief << " voxel)"
                  << (n_miss ? ("  |  " + std::to_string(n_miss) + " yama hic gorulmedi")
                             : "")
                  << "\n";
        g_belief += d_belief;
        g_occ    += d_occ;
    }

    const double g_pct = g_belief > 0 ? 100.0 * g_occ / g_belief : 0.0;
    std::cout << "\n  ═════════════════════════════════════════════════\n";
    std::cout << "  KAPILAR TOPLAM : %" << g_pct << "  (" << g_occ << " / " << g_belief
              << " voxel)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::cout << std::unitbuf;   // boruya yazarken satır satır aksın
    std::cerr << std::unitbuf;

    Config cfg = parse_args(argc, argv);

    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << "  Doors Octomap Builder — İki Aşamalı\n";
    std::cout << "  Mode        : " << cfg.mode        << "\n";
    std::cout << "  Parts dir   : " << cfg.parts_dir   << "\n";
    std::cout << "  Parts scale : " << cfg.parts_scale << "  (kapilar METRE)\n";
    std::cout << "  Data dir    : " << cfg.data_dir    << "\n";
    std::cout << "  Kollar      : " << (cfg.use_ur ? "UR " : "")
                                    << (cfg.use_kawasaki ? "Kawasaki" : "") << "\n";
    std::cout << "  Belief out  : " << cfg.belief_out  << "\n";
    std::cout << "  Occ out     : " << cfg.occ_out     << "\n";
    std::cout << "  Resolution  : " << cfg.res         << " m\n";
    std::cout << "  Snap        : ";
    if (cfg.snap) std::cout << "acik, yaricap " << cfg.snap_radius << " voxel = "
                            << (cfg.snap_radius * cfg.res * 100.0) << " cm\n";
    else          std::cout << "kapali\n";
    std::cout << "  Cloud fix   : ";
    if (cfg.cloud_fix)
        std::cout << "rpy(deg) " << cfg.fix_rpy[0] << "," << cfg.fix_rpy[1] << ","
                  << cfg.fix_rpy[2] << "  xyz(m) " << cfg.fix_xyz[0] << ","
                  << cfg.fix_xyz[1] << "," << cfg.fix_xyz[2] << "\n";
    else
        std::cout << "kapali\n";
    std::cout << "══════════════════════════════════════════════════════\n";

    auto parts = collect_parts(cfg.parts_dir);
    if (parts.empty()) {
        std::cerr << "❌ parts_dir'de .pcd yok: " << cfg.parts_dir
                  << "\n   Once uret:  ros2 run ch08_door_inspection doors_parts_prep\n";
        return 1;
    }

    auto scans = collect_scans(cfg);
    if (scans.empty()) {
        std::cerr << "❌ Eslesen PCD+pose cifti yok: " << cfg.data_dir << "\n";
        return 1;
    }

    std::cout << "\n" << parts.size() << " kapi yamasi  |  " << scans.size()
              << " sensor taramasi\n";

    // Ağacın sahibi unique_ptr; her çıkış yolunda kendiliğinden serbest bırakılır.
    // Yardımcı fonksiyonlar sahiplik almadan ham işaretçiyle (tree.get()) çalışır.
    auto tree = std::make_unique<octomap::ColorOcTree>(cfg.res);

    std::vector<int> belief_counts;
    VoxelPartMap voxel_part_map;
    build_belief_map(tree.get(), parts, cfg.parts_scale, belief_counts, voxel_part_map);

    std::cout << "\n💾 Belief map: " << cfg.belief_out << " ... ";
    if (!tree->write(cfg.belief_out)) {
        std::cerr << "HATA! Dosya yazilamadi.\n";
        return 1;
    }
    std::cout << "OK\n";

    std::vector<int> occ_counts(parts.size(), 0);
    build_occupancy_map(tree.get(), scans, voxel_part_map, occ_counts, cfg.z_min, cfg.snap,
                        cfg.snap_radius, cfg.cloud_fix, cfg.fix_rpy, cfg.fix_xyz);

    std::cout << "\n💾 Occupancy map: " << cfg.occ_out << " ... ";
    if (!tree->write(cfg.occ_out)) {
        std::cerr << "HATA! Dosya yazilamadi.\n";
        return 1;
    }
    std::cout << "OK\n";

    print_coverage(parts, belief_counts, occ_counts);

    std::cout << "\n✅ Tamamlandi.\n";
    return 0;
}
