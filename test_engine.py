"""
Комплексный тестовый модуль валидации ядра AI Forensic Workspace.
Тесты:
1. 13 отношений Аллена и проверка симметрии обратных отношений.
2. Тестирование модели неопределенности времени (Time Uncertainty):
   - Тест 1: Разделены на 10 мин, погрешность +-1 мин -> Нет коллизии.
   - Тест 2: Разделены на 30 сек, погрешность +-60 сек -> ВОЗМОЖНАЯ коллизия.
   - Тест 3: Прямой оверлап -> ПОДТВЕРЖДЁННАЯ коллизия.
   - Тест 4: Отсутствие координат -> INSUFFICIENT_SPATIAL_DATA.
3. Проверка прозрачного скоринга Confidence и цепочки аудита (Audit Trail).
4. Проверка независимого бенчмарка на внешнем размеченном датасете.
"""
import unittest
from datetime import datetime
from discrepancy_engine import (
    Location, AtomicFact, Predicate, AnalysisConfig, TimeInterval,
    ForensicCollisionEngine, CollisionType, ConflictStatus, get_allen_relation,
    ALLEN_INVERSES, ScientificValidator
)

class TestAllenAlgebraFull(unittest.TestCase):
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
            self.assertEqual(direct_rel, expected_rel)
            self.assertEqual(inverse_rel, ALLEN_INVERSES[expected_rel])

class TestTimeUncertaintyAndStatus(unittest.TestCase):
    def setUp(self):
        self.loc_a = Location("Точка А", 0.0, 0.0)
        self.loc_b = Location("Точка Б", 500.0, 0.0)
        self.loc_no_coords = Location("Без координат", None, None)
        self.locations = {
            "Точка А": self.loc_a,
            "Точка Б": self.loc_b,
            "Без координат": self.loc_no_coords
        }
        self.engine = ForensicCollisionEngine(AnalysisConfig())

    def test_1_separated_10min_unc_1min_no_collision(self):
        """Тест 1: События разделены на 10 минут, погрешность +-1 мин -> Коллизии нет."""
        f1 = AtomicFact("F-1", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00", "2026-10-12 14:10", 0.95, "Лог", time_uncertainty_sec=60.0)
        f2 = AtomicFact("F-2", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:20", "2026-10-12 14:30", 0.95, "Лог", time_uncertainty_sec=60.0)
        res = self.engine.analyze([f1, f2], self.locations)
        # 500 м за (14:20-60с) - (14:10+60с) = 480 сек = 3.75 км/ч (нормальный шаг)
        self.assertEqual(len(res), 0, "При зазоре 8 минут со скоростью 3.75 км/ч коллизий быть не должно.")

    def test_2_separated_30sec_unc_60sec_possible_collision(self):
        """Тест 2: События разделены на 30 сек, погрешность +-60 сек -> ВОЗМОЖНАЯ коллизия."""
        f1 = AtomicFact("F-1", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00:00", "2026-10-12 14:05:00", 0.95, "Лог", time_uncertainty_sec=60.0)
        f2 = AtomicFact("F-2", "Свидетель", "свидетель", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:05:30", "2026-10-12 14:10:00", 0.60, "Лог", time_uncertainty_sec=60.0)
        res = self.engine.analyze([f1, f2], self.locations)
        possible_biloc = any(r["status"] == ConflictStatus.POSSIBLE.value for r in res)
        self.assertTrue(possible_biloc, "Должна быть обнаружена ВОЗМОЖНАЯ коллизия (POSSIBLE CONFLICT).")

    def test_3_raw_overlap_confirmed_collision(self):
        """Тест 3: Прямой оверлап по исходным меткам -> ПОДТВЕРЖДЁННАЯ коллизия."""
        f1 = AtomicFact("F-1", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00", "2026-10-12 14:30", 0.95, "Лог", time_uncertainty_sec=10.0)
        f2 = AtomicFact("F-2", "Свидетель", "свидетель", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:15", "2026-10-12 14:45", 0.60, "Лог", time_uncertainty_sec=10.0)
        res = self.engine.analyze([f1, f2], self.locations)
        confirmed_biloc = any(r["status"] == ConflictStatus.CONFIRMED.value and r["type"] == CollisionType.SPATIAL_TEMPORAL.value for r in res)
        self.assertTrue(confirmed_biloc, "Должна быть обнаружена ПОДТВЕРЖДЁННАЯ коллизия (CONFIRMED).")

    def test_4_missing_coordinates_insufficient_data(self):
        """Тест 4: Координаты отсутствуют -> INSUFFICIENT_SPATIAL_DATA."""
        f1 = AtomicFact("F-1", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00", "2026-10-12 14:30", 0.95, "Лог")
        f2 = AtomicFact("F-2", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Без координат", "2026-10-12 14:15", "2026-10-12 14:45", 0.95, "Лог")
        res = self.engine.analyze([f1, f2], self.locations)
        insufficient_data = any(r["type"] == CollisionType.INSUFFICIENT_SPATIAL_DATA.value for r in res)
        self.assertTrue(insufficient_data, "При отсутствии координат должен возвращаться статус INSUFFICIENT_SPATIAL_DATA.")

class TestAuditTrailAndConfidence(unittest.TestCase):
    def setUp(self):
        self.loc_a = Location("Точка А", 0.0, 0.0)
        self.loc_b = Location("Точка Б", 500.0, 0.0)
        self.locations = {"Точка А": self.loc_a, "Точка Б": self.loc_b}
        self.engine = ForensicCollisionEngine(AnalysisConfig())

    def test_confidence_and_audit_trail_structure(self):
        f1 = AtomicFact("F-1", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00:00", "2026-10-12 14:05:00", 0.95, "Лог", time_uncertainty_sec=5.0)
        f2 = AtomicFact("F-2", "СКУД", "турникет", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:05:05", "2026-10-12 14:10:00", 0.95, "Лог", time_uncertainty_sec=5.0)
        res = self.engine.analyze([f1, f2], self.locations)
        self.assertTrue(len(res) > 0)
        item = res[0]
        self.assertIn("confidence", item)
        self.assertIn("confidence_factors", item)
        self.assertIn("evidence_chain", item)
        self.assertIn("calculation", item)
        self.assertIn("limitations", item)
        self.assertGreaterEqual(len(item["evidence_chain"]), 3)

class TestExternalLabeledBenchmark(unittest.TestCase):
    def test_external_dataset_evaluation(self):
        locs = {"Точка А": Location("Точка А", 0.0, 0.0), "Точка Б": Location("Точка Б", 400.0, 300.0)}
        engine = ForensicCollisionEngine(AnalysisConfig())
        dataset = [
            {
                "case_id": "TEST-01",
                "facts": [
                    {"fact_id": "1", "source_id": "C1", "source_type": "камера", "subject": "Айбек", "predicate": "находился", "location_name": "Точка А", "t_start": "2026-10-12 14:00", "t_end": "2026-10-12 14:20", "weight": 0.9, "source_excerpt": "лог"},
                    {"fact_id": "2", "source_id": "S1", "source_type": "свидетель", "subject": "Айбек", "predicate": "находился", "location_name": "Точка Б", "t_start": "2026-10-12 14:10", "t_end": "2026-10-12 14:30", "weight": 0.6, "source_excerpt": "лог"}
                ],
                "expected_collisions": [{"type_category": "bilocation", "type_keyword": "БИЛОКАЦИЯ"}]
            }
        ]
        res = ScientificValidator.evaluate_external_dataset(engine, dataset, locs)
        self.assertEqual(res["total_cases"], 1)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["accuracy"], 100.0)

if __name__ == "__main__":
    unittest.main()
