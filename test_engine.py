"""
Комплексный тестовый модуль валидации ядра AI Forensic Workspace.
"""
import unittest
from datetime import datetime
from discrepancy_engine import (
    Location, AtomicFact, Predicate, AnalysisConfig,
    ForensicCollisionEngine, CollisionType, ConflictStatus, get_allen_relation,
    ALLEN_INVERSES, ScientificValidator, SmartFreeTextParser, ConfidenceCalculator
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
        self.locations = {
            "Точка А": Location("Точка А", 0.0, 0.0),
            "Точка Б": Location("Точка Б", 500.0, 0.0),
            "Без координат": Location("Без координат", None, None)
        }
        self.engine = ForensicCollisionEngine(AnalysisConfig())

    def test_separated_no_collision(self):
        f1 = AtomicFact("F-1", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00", "2026-10-12 14:10", 0.95, "Лог", time_uncertainty_sec=60.0)
        f2 = AtomicFact("F-2", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:20", "2026-10-12 14:30", 0.95, "Лог", time_uncertainty_sec=60.0)
        res = self.engine.analyze([f1, f2], self.locations)
        self.assertEqual(len(res), 0)

    def test_possible_collision(self):
        f1 = AtomicFact("F-1", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00:00", "2026-10-12 14:05:00", 0.95, "Лог", time_uncertainty_sec=60.0)
        f2 = AtomicFact("F-2", "Свидетель", "свидетель", "Арман", Predicate.PRESENT.value, "Точка Б", "2026-10-12 14:05:30", "2026-10-12 14:10:00", 0.60, "Лог", time_uncertainty_sec=60.0)
        res = self.engine.analyze([f1, f2], self.locations)
        self.assertTrue(any(r["status"] == ConflictStatus.POSSIBLE.value for r in res))

    def test_missing_coordinates(self):
        f1 = AtomicFact("F-1", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Точка А", "2026-10-12 14:00", "2026-10-12 14:30", 0.95, "Лог")
        f2 = AtomicFact("F-2", "Камера", "камера", "Арман", Predicate.PRESENT.value, "Без координат", "2026-10-12 14:15", "2026-10-12 14:45", 0.95, "Лог")
        res = self.engine.analyze([f1, f2], self.locations)
        self.assertTrue(any(r["type"] == CollisionType.INSUFFICIENT_SPATIAL_DATA.value for r in res))

class TestEdgeCasesAndScoring(unittest.TestCase):
    def test_midnight_rollover_parser(self):
        locs = {"Библиотека": Location("Библиотека", 0.0, 0.0)}
        text = "Свидетель сообщил: видел Армана около 23:50."
        facts, _ = SmartFreeTextParser.parse_documents(text, "2026-10-12", locs)
        self.assertEqual(len(facts), 1)
        self.assertTrue(facts[0].is_valid_interval())
        self.assertIn("00:05:00", facts[0].t_end)

    def test_continuous_confidence_scoring(self):
        f1 = AtomicFact("1", "S1", "камера", "А", Predicate.PRESENT.value, "L1", "2026-10-12 14:00", "2026-10-12 14:05", 0.95, "log")
        f2 = AtomicFact("2", "S2", "камера", "А", Predicate.PRESENT.value, "L2", "2026-10-12 14:05:10", "2026-10-12 14:10", 0.95, "log")
        
        c1, _, _ = ConfidenceCalculator.calculate(f1, f2, ConflictStatus.CONFIRMED, CollisionType.KINEMATIC_CRITICAL, True, kinematic_ratio=1.0)
        c2, _, _ = ConfidenceCalculator.calculate(f1, f2, ConflictStatus.CONFIRMED, CollisionType.KINEMATIC_CRITICAL, True, kinematic_ratio=1.01)
        c3, _, _ = ConfidenceCalculator.calculate(f1, f2, ConflictStatus.CONFIRMED, CollisionType.KINEMATIC_CRITICAL, True, kinematic_ratio=2.0)
        
        self.assertGreaterEqual(c2, c1)
        self.assertGreaterEqual(c3, c2)

if __name__ == "__main__":
    unittest.main()
