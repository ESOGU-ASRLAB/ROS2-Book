import logging
from typing import Any, Dict, List, Optional, Tuple

from ch07_viewpoint_planner.viewpoint_generator import Candidate
import numpy as np
from sensor_msgs.msg import JointState


class RobotAllocator:
    """
    Çok robotlu bakış noktası tahsisi (açgözlü yöntem).

    TEK bir ortak aday bakış noktası havuzu ve onun kaplama matrisi ile robot başına
    bir erişilebilirlik (IK) fonksiyonu verildiğinde, bakış noktalarını açgözlü
    biçimde seçer ve her birini tam olarak bir kola atar. İki hedefi birden izler:

      * Tek koldan DAHA YÜKSEK toplam kaplama. Yalnızca bir kolun ulaşabildiği bir
        hedef (örneğin Kawasaki tarafındaki, tek Y-raylı UR'nin asla ulaşamayacağı
        bir iç/uzak yüzey) yine kaplanır, çünkü tahsis edici onu görebilen kolu
        seçer. Erişilebilir aday kümesi iki kolun BİRLEŞİMİDİR; bu yüzden kaplama
        tavanı tek kolunkinin üstüne çıkar.

      * DAHA HIZLI muayene. İki kol aynı anda hareket ettiği için toplam süre
        ~max(süre_ur, süre_kawasaki) olur. Atanan SAYIYI dengelemek (balance=True)
        bu süreyi yalnızca kollar yaklaşık eşit hızdayken kısaltır. Oysa eşit hızda
        değiller: kapı işinde ölçüldüğünde UR 8 durağının hepsini bitirdiğinde
        Kawasaki daha ikincisindeydi; yani eşit paylaşım UR'yi boşta bırakır ve
        koşunun hızını tamamen Kawasaki belirler. Bu durumun iki ayarı, kol başına
        üst sınır (max_per_robot_by_name) ve balance=False'tur; ikisi birlikte
        ortak işi hızlı kola iter ve yavaş kolun birkaç durağını yalnızca yerine
        başkası konamayacak yerlere harcar.

    Açgözlü çekirdek, tek kollu SetCoverOptimizer'ınkiyle aynı en çok kaplama
    çekirdeğidir (sıradaki olarak hâlâ kaplanmamış yüzeyin en çoğunu kaplayan adayı
    seç); buna bir kol atama adımı eklenmiştir: seçilen adaya gerçekten
    ulaşabilen kollar arasından o anda DAHA AZ bakış noktası olana verilir (yük
    dengeleme), balance=False ise listedeki İLK kola verilir. Hiçbir kolun
    ulaşamadığı ya da yalnızca üst sınırını doldurmuş bir kolun ulaşabildiği aday
    atılır.
    """

    def __init__(
            self, coverage_threshold: float = 0.98, max_per_robot: Optional[int] = None,
            balance: bool = True, min_new_points: int = 1, min_marginal_coverage: float = 0.0,
            max_per_robot_by_name: Optional[Dict[str, int]] = None,
            logger: Optional[Any] = None) -> None:
        self.coverage_threshold = coverage_threshold
        # None ya da <=0 -> kol başına bütçe yok (eşiğe kadar kapla). Aksi hâlde
        # her kol en fazla bu kadar bakış noktası alır; böylece bir koşu "her kol en
        # fazla N durak yapar" ile sınırlanabilir.
        self.max_per_robot = max_per_robot if (max_per_robot and max_per_robot > 0) else None
        # Yukarıdaki bütçenin KOL BAŞINA geçersiz kılınması, {robot_adı: üst_sınır}.
        # İki kol birbirinin yerine geçemez: paralel ama çok farklı hızlarda
        # çalışırlar; bu yüzden koşunun süresini toplam değil, YAVAŞ kolun durak
        # sayısı belirler. Koşuyu kısaltan yalnızca o kolu sınırlamaktır; ortak bir
        # max_per_robot bunu ifade edemez, çünkü onu düşürmek ikisini de kısar.
        # Üst sınırı <= 0 olan girdiler yok sayılır (o kol max_per_robot'a düşer).
        self.max_per_robot_by_name = {
            str(n): int(v) for n, v in (max_per_robot_by_name or {}).items() if int(v) > 0
        }
        # True iken iki kolun da ulaşabildiği bir aday, o anda daha az atanmış
        # kola gider (daha kısa paralel koşu). False iken kollar sabit bir öncelik
        # sırasını korur ve yalnızca dolaylı olarak dengelenir.
        self.balance = balance
        # Kuyruk kesici (MUTLAK): kalan en iyi bakış noktası bu sayıdan AZ yeni hedef
        # noktası ekleyecekse bakış noktası eklemeyi durdur. Kaplama eğrisinin keskin
        # bir dirseği vardır: ilk bakış noktaları yüzlerce nokta ekler, ardından uzun
        # bir kuyruk her biri yalnızca 1-3 nokta ekler. Bunu artırmak biraz kaplamadan
        # vazgeçip çok daha kısa bir plan (daha az ara nokta) verir. 1 = her işe yarar bakış
        # noktasını tut.
        self.min_new_points = max(1, int(min_new_points))
        # Kuyruk kesici (ORANSAL, tek kollu SetCoverOptimizer'ın aynası): kalan en iyi
        # bakış noktası yeni kaplama olarak bütün hedef noktalarının bu ORANINDAN
        # azını ekleyecekse dur. Ağ örnekleme yoğunluğuyla ölçeklenir (0.005 = %0.5;
        # 5000 hedefli bir ağda ~25 yeni nokta). Kullanılan etkin eşik bununla
        # min_new_points'in BÜYÜĞÜDÜR; ikisinden biri kesmeyi sıkılaştırabilir. 0 kapatır.
        self.min_marginal_coverage = max(0.0, float(min_marginal_coverage))
        self.logger = logger or logging.getLogger(__name__)

    def allocate(
            self, candidates: List[Candidate], coverage_matrix: np.ndarray,
            robots: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Candidate]], float]:
        """
        Bakış noktalarını kollar arasında paylaştırır.

        candidates: bakış noktası sözlüklerinin listesi (her birinde 'position', 'rotation').
        coverage_matrix: [num_candidates, num_targets] boyutlu mantıksal dizi.
        robots: sıralı sözlük listesi, her biri:
            {
              'name': str,                      # örneğin 'ur' / 'kawasaki'
              'reachability_fn': callable | None,
                  # aday -> (erişilebilir: bool, eklem_çözümü). None ise kolun her
                  # adaya ulaşabildiği varsayılır (DOĞRULANMAMIŞ; bunu yüksek sesle
                  # günlüğe yazmak çağıranın sorumluluğundadır).
            }

        (assignments, final_coverage) döndürür; assignments, seçilme sırasıyla
        {robot_adı: [seçilen aday sözlükleri]} biçimindedir. Tutulan her adaya 'robot',
        'joint_solution', 'rank', 'new_points_covered', 'total_points_visible' ve
        'cumulative_coverage' alanları eklenir.
        """
        num_c, num_t = coverage_matrix.shape
        names = [r['name'] for r in robots]
        budgets = ", ".join(f"{n}={self._budget(n)}" for n in names)
        self.logger.info(
            f"Multi-robot allocation: {num_c} candidates, {num_t} targets, arms={names}, "
            f"coverage_threshold={self.coverage_threshold * 100:.1f}%, "
            f"budgets [{budgets}] (shared max_per_robot={self.max_per_robot}), "
            f"balance={self.balance}."
        )
        if not self.balance:
            self.logger.info(
                f"balance=False: a candidate both arms can reach always goes to '{names[0]}'. "
                f"The other arms only receive candidates '{names[0]}' cannot reach, which is "
                "what makes a small per-arm cap on a slow arm spend its budget where it is "
                "actually needed instead of on shared, high-gain viewpoints."
            )

        assignments = {r['name']: [] for r in robots}
        if num_c == 0 or num_t == 0:
            self.logger.error("No candidates or no target points; cannot allocate.")
            return assignments, 0.0

        uncovered = np.ones(num_t, dtype=bool)
        covered = np.zeros(num_t, dtype=bool)
        available = np.ones(num_c, dtype=bool)  # aday hâlâ seçilebilir mi?
        # Önbelleğe alınmış IK sonuçları: (aday_indeksi, robot_adı) -> (bool, joint_state).
        ik_cache = {}
        ik_queries = 0
        excluded_unreachable = 0
        excluded_at_budget = 0
        order = 0  # genel seçim sırası (birikimli kaplama için)

        target_covered = int(num_t * self.coverage_threshold)

        # Etkin kuyruk eşiği = mutlak (min_new_points) ile oransal (min_marginal_coverage * num_t)
        # kesicilerin büyüğü; böylece ikisinden biri sıkılaştırabilir.
        frac_floor = int(np.ceil(self.min_marginal_coverage * num_t)) \
            if self.min_marginal_coverage > 0.0 else 0
        tail_floor = max(self.min_new_points, frac_floor)
        if tail_floor > 1:
            self.logger.info(
                f"Tail cut floor = {tail_floor} new points "
                f"(min_new_points={self.min_new_points}, "
                f"min_marginal_coverage={self.min_marginal_coverage * 100:.2f}% "
                f"= {frac_floor}); stops once the best next viewpoint adds fewer."
            )

        def reach(idx: int, robot: Dict[str, Any]) -> Tuple[bool, Optional[JointState]]:
            nonlocal ik_queries
            key = (idx, robot['name'])
            if key not in ik_cache:
                fn = robot['reachability_fn']
                if fn is None:
                    ik_cache[key] = (True, None)
                else:
                    ik_queries += 1
                    ok, sol = fn(candidates[idx])
                    ik_cache[key] = (bool(ok), sol)
            return ik_cache[key]

        while np.sum(covered) < target_covered:
            if all(self._at_budget(n, assignments[n]) for n in assignments):
                self.logger.info("Every arm reached its per-robot budget; stopping allocation.")
                break

            # Hâlâ seçilebilir her adayın marjinal yeni kaplaması, tek seferde.
            gains = np.count_nonzero(coverage_matrix & uncovered, axis=1)
            gains[~available] = -1
            best_idx = int(np.argmax(gains))
            if gains[best_idx] <= 0:
                remaining_pct = 100 * np.sum(uncovered) / num_t
                self.logger.warning(
                    "Allocation exhausted: no remaining candidate adds new coverage. Stopping "
                    "with "
                    f"{remaining_pct:.2f}% of the surface still uncovered (occlusion / camera "
                    "range / "
                    "neither arm can reach it)."
                )
                break
            if gains[best_idx] < tail_floor:
                self.logger.info(
                    f"Tail cut: best remaining viewpoint adds only {int(gains[best_idx])} new "
                    "points "
                    f"(< floor {tail_floor}). Stopping to keep the plan short "
                    f"(coverage {np.sum(covered) / num_t * 100:.2f}%)."
                )
                break

            # Hangi kollar bu adaya ulaşabiliyor VE hâlâ bütçesi var? En az yüklü
            # uygun kol kazansın diye sırala (yük dengeleme).
            eligible = []
            for r in robots:
                if self._at_budget(r['name'], assignments[r['name']]):
                    continue
                ok, sol = reach(best_idx, r)
                if ok:
                    eligible.append((len(assignments[r['name']]), r['name'], sol))

            if not eligible:
                # Hiçbir kol (erişilebilir biçimde, bütçe içinde) en iyi adayı alamıyor.
                # Açgözlü yöntem bir sonraki en iyi adaya geçsin diye onu emekliye ayır.
                available[best_idx] = False
                # Bütçesi dolmamış her kol için erişilemezdi; yalnızca bütün kollarda
                # gerçekten IK'dan geçemediyse say (yalnızca bütçe engeline takıldıysa değil).
                if all(reach(best_idx, r)[0] is False for r in robots):
                    excluded_unreachable += 1
                else:
                    # Bir kol ulaşABİLİRDİ ama üst sınırına ulaşmış. Bu, kol başına bütçenin
                    # doğrudan kaplama maliyetidir; bu yüzden ayrı sayılır, yoksa bir üst
                    # sınır erişilemez bir ağ gibi görünürdü.
                    excluded_at_budget += 1
                continue

            if self.balance:
                eligible.sort(key=lambda e: e[0])  # önce en az ataması olan
            count, name, sol = eligible[0]

            # Atamayı kesinleştir.
            mask = coverage_matrix[best_idx]
            new_pts = int(np.count_nonzero(mask & ~covered))
            covered |= mask
            uncovered &= ~mask
            available[best_idx] = False

            vp = candidates[best_idx]
            vp['robot'] = name
            vp['joint_solution'] = sol
            vp['new_points_covered'] = new_pts
            vp['total_points_visible'] = int(mask.sum())
            vp['cumulative_coverage'] = float(np.sum(covered) / num_t)
            vp['global_order'] = order
            order += 1
            assignments[name].append(vp)

            self.logger.info(
                f"[{len(assignments[name]):>2}] {name} <- candidate {best_idx} "
                f"(+{new_pts} new pts). Total coverage {vp['cumulative_coverage'] * 100:.2f}% "
                f"(ur={len(assignments.get('ur', []))}, "
                f"kawa={len(assignments.get('kawasaki', []))})."
            )

        final_coverage = float(np.sum(covered) / num_t)

        # Kol başına sıra (o kolun kendi ziyaret sırasındaki yeri).
        for name in assignments:
            for rank, vp in enumerate(assignments[name]):
                vp['rank'] = rank

        totals = ", ".join(f"{n}={len(assignments[n])}" for n in assignments)
        self.logger.info(
            f"Allocation done: coverage {final_coverage * 100:.2f}% "
            f"(threshold {self.coverage_threshold * 100:.1f}%), assigned [{totals}], "
            f"{ik_queries} IK queries, {excluded_unreachable} candidates unreachable by any arm."
        )
        if excluded_at_budget:
            capped = [n for n in assignments if self._budget(n) is not None]
            self.logger.warning(
                f"{excluded_at_budget} candidate(s) were dropped ONLY because the arm that "
                f"could reach them had already spent its budget (capped arms: {capped}). "
                "That is the coverage price of the cap; raise the cap to buy it back."
            )
        if final_coverage + 1e-9 < self.coverage_threshold:
            self.logger.warning(
                f"Coverage target NOT met: {final_coverage * 100:.2f}% < "
                f"{self.coverage_threshold * 100:.1f}%. This multi-robot ceiling is still higher "
                "than a "
                "single arm's, but interior/occluded faces neither camera can see remain "
                "uncovered."
            )
        return assignments, final_coverage

    def _budget(self, name: str) -> Optional[int]:
        """Bir kolun bakış noktası üst sınırı: kendi değeri, yoksa ortak değer, yoksa None."""
        return self.max_per_robot_by_name.get(name, self.max_per_robot)

    def _at_budget(self, name: str, selected_list: List[Candidate]) -> bool:
        cap = self._budget(name)
        return cap is not None and len(selected_list) >= cap
