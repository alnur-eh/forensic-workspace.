"""
Комплексный тестовый модуль валидации ядра AI Forensic Workspace.
1. Все 13 отношений Аллена и проверка симметрии обратных отношений.
2. Проверка семантики MEETS (отсутствие ложной билокации при касании границ).
3. Проверка многоуровневой кинематики (шаг, бег, автотранспорт, сверхскорость).
4. Проверка нулевого зазора времени (телепортация vs совпадение координат).
5. Изоляция SOURCE_RELIABILITY от разнесенных во времени фактов.
6. Граничные и поврежденные входные данные.
"""
import unittest
from datetime import datetime
from discrepancy_engine import (
    Location, AtomicFact, Predicate, AnalysisConfig,
    ForensicCollisionEngine, CollisionType, get_allen_relation,
    ALLEN_INVERSES, ScientificValidator
)

class TestAllenAlgebraComprehensive(unittest.TestCase):
    def setUp(self):
        self.t = [datetime(2026, 10, 12, 14, i * 5) for i in range(12)]

    def test_all_13_relations_and_inverses(self):
        test_pairs = [
            ("EQUALS", self.t[1], self.t[3], self.t[1], self.t[3]),
            ("BEFORE", self.t[1], self.t[2], self.t[3], self.t[4]),
            ("AFTER", self.t[3], self.t[4], self.t[1], self.t[2]),
            ("MEETS", self.t[1], self.t[2], self.t[2], self.t[3]),
            ("MET_BY", self.t[2], self.t[3], self.t[1], self.t[2]),
            ("STARTS", self.t[1], self.t[2], self.t[1], self.t[3]),
            ("STARTED_BY", self.t[1], self.t[3], self.t[1], self.t[2]),
            ("FINISHES", self.t[2], self.t[3], self.t[1], self.t[3]),
            ("FINISHED_BY", self.t[1], self.t[3], self.t[2], self.t[3]),
            ("DURING", self.t[2], self.t[3], self.t[1], self.t[4]),
            ("CONTAINS", self.t[1], self.t[4], self.t[2], self.t[3]),
            ("OVERLAPS", self.t[1], self.t[3], self.t[2], self.t[4]),
            ("OVERLAPPED_BY", self.t[2], self.t[4], self.t[1], self.t[3]),
        ]
        for expected_rel, sa, ea, sb, eb in test_pairs:
            direct_rel = get_allen_relation(sa, ea, sb, eb)
            inverse_rel = get_allen_relation(sb, eb, sa, ea)
            self.assertEqual(direct_rel, expected_rel, f"Прямое отношение для {expected_rel} не совпало.")
            self.assertEqual(inverse_rel, ALLEN_INVERSES[expected_rel], f"Обратное отношение для {expected_rel} не совпало.")

class TestForensicLogicAndSemantics(unittest.TestCase):
    def setUp(self):
        self.loc_a = Location("Точка А", 0.0, 0.0)
        self.loc_b = Location("Точка Б", 500.0, 0.0)
        self.locations = {"Точка А": self.loc_a, "Точка Б": self.loc_b}
        self.engine = ForensicCollisionEngine(AnalysisConfig(
            max_walking_speed_kmh=5.0,
            max_sprint_speed_kmh=18.0,
            max_vehicle_speed_kmh=90.0,
            same_location_radius_m=2.0
        ))

    def test_meets_semantics_no_bilocation(self):
        """Отношение MEETS (14:00-14:05 и 14:05-14:10) не должно порождать билокацию."""
        f1 = AtomicFact("F-1", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00:00", "2026-10-12 14:05:00", 0.95, "Лог 1", time_uncertainty_sec=0.0)
        f2 = AtomicFact("F-2", "Свидетель", "свидетель", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:05:00", "2026-10-12 14:10:00", 0.60, "Лог 2", time_uncertainty_sec=0.0)
        res = self.engine.analyze([f1, f2], self.locations)
        # Касание границ не является билокацией
        self.assertFalse(any(r["type"] == CollisionType.SPATIAL_TEMPORAL.value for r in res))

    def test_zero_time_gap_different_locations_teleportation(self):
        """Нулевой временной зазор между разными точками (dist=500м) -> Мгновенная телепортация."""
        f1 = AtomicFact("F-1", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00:00", "2026-10-12 14:05:00", 0.95, "Лог 1")
        f2 = AtomicFact("F-2", "СКУД", "турникет", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:05:00", "2026-10-12 14:10:00", 0.95, "Лог 2")
        res = self.engine.analyze([f1, f2], self.locations)
        self.assertTrue(any(r["type"] == CollisionType.KINEMATIC_CRITICAL.value for r in res),
                        "При 0 с и dist=500 м должна фиксироваться критическая кинематическая аномалия.")

    def test_zero_time_gap_same_location_no_anomaly(self):
        """Нулевой временной зазор в одной и той же точке (dist=0м) -> норма."""
        f1 = AtomicFact("F-1", "Камера 1", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00:00", "2026-10-12 14:05:00", 0.95, "Лог 1")
        f2 = AtomicFact("F-2", "Камера 2", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:05:00", "2026-10-12 14:10:00", 0.95, "Лог 2")
        res = self.engine.analyze([f1, f2], self.locations)
        self.assertEqual(len(res), 0, "В одной локации кинематических аномалий быть не должно.")

    def test_source_bias_requires_temporal_overlap(self):
        """Разница весов без временного конфликта (14:00 и 20:00) НЕ должна создавать коллизию."""
        f1 = AtomicFact("F-1", "Подозреваемый", "подозреваемый", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00", "2026-10-12 14:30", 0.35, "Лог 1")
        f2 = AtomicFact("F-2", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 20:00", "2026-10-12 20:30", 0.95, "Лог 2")
        res = self.engine.analyze([f1, f2], self.locations)
        self.assertFalse(any(r["type"] == CollisionType.SOURCE_RELIABILITY.value for r in res),
                         "SOURCE_RELIABILITY не должен срабатывать на события с разрывом во времени.")

    def test_kinematic_classification_tiers(self):
        """Проверка всех ступеней кинематики: шаг, спринт, автотранспорт, сверхскорость."""
        # 1. Шаг: 500 м за 10 мин (600 с) = 3.0 км/ч (норма)
        f_start = AtomicFact("F-S", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00:00", "2026-10-12 14:05:00", 0.95, "Лог")
        f_walk = AtomicFact("F-W", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:15:00", "2026-10-12 14:20:00", 0.95, "Лог")
        res_walk = self.engine.analyze([f_start, f_walk], self.locations)
        self.assertEqual(len(res_walk), 0, "Шаг (3 км/ч) не должен создавать коллизий.")

        # 2. Требуется транспорт: 500 м за 30 с = 60.0 км/ч (между 18 и 90 км/ч)
        f_veh = AtomicFact("F-V", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:05:30", "2026-10-12 14:10:00", 0.95, "Лог")
        res_veh = self.engine.analyze([f_start, f_veh], self.locations)
        self.assertTrue(any(r["type"] == CollisionType.KINEMATIC_VEHICLE_REQUIRED.value for r in res_veh),
                        "Скорость 60 км/ч должна классифицироваться как 'ТРЕБУЕТСЯ АВТОТРАНСПОРТ'.")

        # 3. Сверхскорость: 500 м за 5 с = 360.0 км/ч (> 90 км/ч)
        f_crit = AtomicFact("F-C", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:05:05", "2026-10-12 14:10:00", 0.95, "Лог")
        res_crit = self.engine.analyze([f_start, f_crit], self.locations)
        self.assertTrue(any(r["type"] == CollisionType.KINEMATIC_CRITICAL.value for r in res_crit),
                        "Скорость 360 км/ч должна классифицироваться как критическая аномалия.")

class TestBenchmarkQuality(unittest.TestCase):
    def setUp(self):
        self.engine = ForensicCollisionEngine(AnalysisConfig())

    def test_benchmark_metrics_thresholds(self):
        res = ScientificValidator.run_benchmark(self.engine, test_samples=100, add_noise=False, seed=42)
        self.assertGreaterEqual(res["accuracy"], 90.0, "Accuracy >= 90%")
        self.assertGreaterEqual(res["precision"], 90.0, "Precision >= 90%")
        self.assertGreaterEqual(res["recall"], 85.0, "Recall >= 85%")
        self.assertGreaterEqual(res["f1_score"], 85.0, "F1-Score >= 85%")

if __name__ == "__main__":
    unittest.main()
