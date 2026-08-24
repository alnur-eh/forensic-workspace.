"""
AI Forensic Workspace — Algorithmic Core
"""
from __future__ import annotations
import math
import random
import re
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")

def parse_iso_datetime(dt_str: str) -> Optional[datetime]:
    if not dt_str or not isinstance(dt_str, str):
        return None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            pass
    return None

class Predicate(str, Enum):
    PRESENT = "находился"
    ABSENT = "отсутствовал"
    INTERACTED = "контактировал_с"
    SEEN = "видел_субъекта"

class CollisionType(str, Enum):
    SPATIAL_TEMPORAL = "ПРОСТРАНСТВЕННО-ВРЕМЕННАЯ НЕСОГЛАСОВАННОСТЬ (БИЛОКАЦИЯ)"
    KINEMATIC_VEHICLE_REQUIRED = "КИНЕМАТИКА: ТРЕБУЕТСЯ АВТОТРАНСПОРТ"
    KINEMATIC_CRITICAL = "КИНЕМАТИЧЕСКАЯ АНОМАЛИЯ (ПРЕВЫШЕНИЕ СКОРОСТИ / ТЕЛЕПОРТАЦИЯ)"
    DIRECT_CONTRADICTION = "ПРЯМОЕ ЛОГИЧЕСКОЕ ПРОТИВОРЕЧИЕ УТВЕРЖДЕНИЙ"
    SOURCE_RELIABILITY = "ДИСБАЛАНС ВЕСОВ КОНФЛИКТУЮЩИХ ИСТОЧНИКОВ"

ALLEN_INVERSES: Dict[str, str] = {
    "BEFORE": "AFTER", "AFTER": "BEFORE", "MEETS": "MET_BY", "MET_BY": "MEETS",
    "STARTS": "STARTED_BY", "STARTED_BY": "STARTS", "FINISHES": "FINISHED_BY",
    "FINISHED_BY": "FINISHES", "DURING": "CONTAINS", "CONTAINS": "DURING",
    "OVERLAPS": "OVERLAPPED_BY", "OVERLAPPED_BY": "OVERLAPS", "EQUALS": "EQUALS"
}

@dataclass
class Location:
    name: str
    x: Optional[float] = None
    y: Optional[float] = None
    description: str = ""

    @property
    def has_coordinates(self) -> bool:
        return self.x is not None and self.y is not None

@dataclass
class AtomicFact:
    fact_id: str
    source_id: str
    source_type: str
    subject: str
    predicate: str
    location_name: str
    t_start: str
    t_end: str
    weight: float
    source_excerpt: str
    motive_flag: str = "Нейтральный"
    interest_conflict: float = 0.0
    time_uncertainty_sec: float = 0.0
    object_target: Optional[str] = None

    def parse_start(self) -> Optional[datetime]:
        return parse_iso_datetime(self.t_start)

    def parse_end(self) -> Optional[datetime]:
        return parse_iso_datetime(self.t_end)

    def is_valid_interval(self) -> bool:
        s, e = self.parse_start(), self.parse_end()
        if s is None or e is None:
            return False
        return s <= e

    def get_effective_bounds(self) -> Optional[Tuple[datetime, datetime]]:
        s, e = self.parse_start(), self.parse_end()
        if s is None or e is None or s > e:
            return None
        delta = timedelta(seconds=max(0.0, self.time_uncertainty_sec))
        return (s - delta, e + delta)

    def get_signature(self) -> str:
        return f"{self.source_id}|{self.subject}|{self.predicate}|{self.location_name}|{self.t_start}|{self.t_end}"

@dataclass
class AnalysisConfig:
    max_walking_speed_kmh: float = 5.0
    max_sprint_speed_kmh: float = 18.0
    max_vehicle_speed_kmh: float = 90.0
    same_location_radius_m: float = 2.0
    critical_weight_gap: float = 0.45

def calculate_distance(loc1: Location, loc2: Location) -> Optional[float]:
    if not loc1.has_coordinates or not loc2.has_coordinates:
        return None
    return math.hypot(loc1.x - loc2.x, loc1.y - loc2.y)

def get_allen_relation(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> str:
    if start_a > end_a or start_b > end_b:
        return "INVALID"
    if start_a == start_b and end_a == end_b: return "EQUALS"
    if end_a < start_b: return "BEFORE"
    if end_b < start_a: return "AFTER"
    if end_a == start_b: return "MEETS"
    if end_b == start_a: return "MET_BY"
    if start_a == start_b and end_a < end_b: return "STARTS"
    if start_a == start_b and end_b < end_a: return "STARTED_BY"
    if end_a == end_b and start_b < start_a: return "FINISHES"
    if end_a == end_b and start_a < start_b: return "FINISHED_BY"
    if start_b < start_a and end_a < end_b: return "DURING"
    if start_a < start_b and end_b < end_a: return "CONTAINS"
    if start_a < start_b < end_a < end_b: return "OVERLAPS"
    if start_b < start_a < end_b < end_a: return "OVERLAPPED_BY"
    return "UNKNOWN"

class ForensicCollisionEngine:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

    def analyze(self, facts: List[AtomicFact], locations: Dict[str, Location]) -> List[Dict]:
        collisions: List[Dict] = []
        valid_facts = [f for f in facts if f.is_valid_interval()]
        n = len(valid_facts)
        for i in range(n):
            for j in range(i + 1, n):
                f1, f2 = valid_facts[i], valid_facts[j]
                if f1.subject == f2.subject:
                    collisions.extend(self._evaluate_subject_pair(f1, f2, locations))
        return collisions

    def _evaluate_subject_pair(self, f1: AtomicFact, f2: AtomicFact, locations: Dict[str, Location]) -> List[Dict]:
        results = []
        bounds1 = f1.get_effective_bounds()
        bounds2 = f2.get_effective_bounds()
        if not bounds1 or not bounds2:
            return results

        eff_start1, eff_end1 = bounds1
        eff_start2, eff_end2 = bounds2

        raw_start1, raw_end1 = f1.parse_start(), f1.parse_end()
        raw_start2, raw_end2 = f2.parse_start(), f2.parse_end()
        relation = get_allen_relation(raw_start1, raw_end1, raw_start2, raw_end2)

        loc1 = locations.get(f1.location_name)
        loc2 = locations.get(f2.location_name)

        effective_overlap = (eff_start1 < eff_end2) and (eff_start2 < eff_end1)

        if effective_overlap and loc1 and loc2 and loc1.name == loc2.name:
            if {f1.predicate, f2.predicate} == {Predicate.PRESENT.value, Predicate.ABSENT.value}:
                results.append({
                    "id": f"COL-LOGIC-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.DIRECT_CONTRADICTION.value,
                    "subject": f1.subject,
                    "severity": "ВЫСОКАЯ",
                    "details": f"Взаимоисключающие утверждения о присутствии/отсутствии в локации '{loc1.name}'.",
                    "allen_relation": relation,
                    "facts": [f1, f2],
                    "expert_note": f"Требуется перекрестный допрос. Дельта конфликта интересов: {abs(f1.interest_conflict - f2.interest_conflict):.2f}."
                })

        if effective_overlap and loc1 and loc2 and loc1.has_coordinates and loc2.has_coordinates:
            dist = calculate_distance(loc1, loc2)
            if dist is not None and dist > self.config.same_location_radius_m and f1.predicate == Predicate.PRESENT.value and f2.predicate == Predicate.PRESENT.value:
                results.append({
                    "id": f"COL-ST-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.SPATIAL_TEMPORAL.value,
                    "subject": f1.subject,
                    "severity": "КРИТИЧЕСКАЯ",
                    "details": f"Одновременное присутствие в несовпадающих точках ({dist:.1f} м > допустимой погрешности {self.config.same_location_radius_m:.1f} м).",
                    "allen_relation": relation,
                    "facts": [f1, f2],
                    "expert_note": "Несогласованность временных меток объективного контроля или свидетельских показаний."
                })

        if effective_overlap and f1.location_name != f2.location_name:
            diff = abs(f1.weight - f2.weight)
            if diff >= self.config.critical_weight_gap:
                low_src = f1 if f1.weight < f2.weight else f2
                high_src = f2 if f1.weight < f2.weight else f1
                results.append({
                    "id": f"COL-BIAS-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.SOURCE_RELIABILITY.value,
                    "subject": f1.subject,
                    "severity": "СРЕДНЯЯ",
                    "details": f"При временном конфликте источник низкой надежности '{low_src.source_id}' ({low_src.weight:.2f}) противоречит объективному источнику '{high_src.source_id}' ({high_src.weight:.2f}).",
                    "allen_relation": f"Дельта весов: {diff:.2f}",
                    "facts": [f1, f2],
                    "expert_note": f"Мотивационный интерес источника: {low_src.motive_flag}."
                })

        if loc1 and loc2 and loc1.has_coordinates and loc2.has_coordinates and f1.predicate == Predicate.PRESENT.value and f2.predicate == Predicate.PRESENT.value:
            if eff_end1 <= eff_start2:
                earlier, later, e_end, l_start = f1, f2, eff_end1, eff_start2
            elif eff_end2 <= eff_start1:
                earlier, later, e_end, l_start = f2, f1, eff_end2, eff_start1
            else:
                earlier = None

            if earlier:
                loc_e = locations.get(earlier.location_name)
                loc_l = locations.get(later.location_name)
                if loc_e and loc_l:
                    dist = calculate_distance(loc_e, loc_l)
                    gap_sec = (l_start - e_end).total_seconds()
                    if dist is not None and dist > self.config.same_location_radius_m:
                        if gap_sec == 0:
                            results.append({
                                "id": f"COL-KIN-{f1.fact_id}-{f2.fact_id}",
                                "type": CollisionType.KINEMATIC_CRITICAL.value,
                                "subject": f1.subject,
                                "severity": "КРИТИЧЕСКАЯ",
                                "details": f"Мгновенное перемещение между '{loc_e.name}' и '{loc_l.name}' (дистанция {dist:.1f} м за 0.0 с).",
                                "allen_relation": "Временной зазор: 0.0 с (MEETS)",
                                "facts": [f1, f2],
                                "expert_note": "Физически невозможное перемещение между пространственно разделенными объектами."
                            })
                        elif gap_sec > 0:
                            speed_kmh = (dist / gap_sec) * 3.6
                            if speed_kmh > self.config.max_vehicle_speed_kmh:
                                results.append({
                                    "id": f"COL-KIN-{f1.fact_id}-{f2.fact_id}",
                                    "type": CollisionType.KINEMATIC_CRITICAL.value,
                                    "subject": f1.subject,
                                    "severity": "КРИТИЧЕСКАЯ",
                                    "details": f"Расчетная скорость {speed_kmh:.1f} км/ч превышает порог автотранспорта ({self.config.max_vehicle_speed_kmh:.1f} км/ч). Дистанция: {dist:.1f} м за {gap_sec:.1f} с.",
                                    "allen_relation": f"Временной зазор: {gap_sec:.1f} с",
                                    "facts": [f1, f2],
                                    "expert_note": "Аномальная скорость перемещения."
                                })
                            elif speed_kmh > self.config.max_sprint_speed_kmh:
                                results.append({
                                    "id": f"COL-KIN-{f1.fact_id}-{f2.fact_id}",
                                    "type": CollisionType.KINEMATIC_VEHICLE_REQUIRED.value,
                                    "subject": f1.subject,
                                    "severity": "ВЫСОКАЯ",
                                    "details": f"Расчетная скорость {speed_kmh:.1f} км/ч превышает порог бега ({self.config.max_sprint_speed_kmh:.1f} км/ч), допустима для автотранспорта (дистанция {dist:.1f} м за {gap_sec:.1f} с).",
                                    "allen_relation": f"Временной зазор: {gap_sec:.1f} с",
                                    "facts": [f1, f2],
                                    "expert_note": "Пешее перемещение исключено. Требуется подтверждение использования автотранспорта."
                                })
        return results

class SmartFreeTextParser:
    @staticmethod
    def parse_documents(text: str, default_date: str, current_locs: Dict[str, Location], start_id: int = 1) -> Tuple[List[AtomicFact], Dict[str, Location]]:
        new_facts: List[AtomicFact] = []
        updated_locs = dict(current_locs)
        chunks = [c.strip() for c in re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", text) if len(c.strip()) > 5]
        time_range_re = re.compile(r"(\d{1,2}[:.]\d{2}(?::\d{2})?)\s*(?:-|—|до)\s*(\d{1,2}[:.]\d{2}(?::\d{2})?)")
        single_time_re = re.compile(r"(?:в|около|примерно|после|время[:\s]*)\s*(\d{1,2}[:.]\d{2}(?::\d{2})?)")
        names_pool = ["Арман", "Дамир", "Нурлан", "Алихан", "Охранник", "Курьер", "Директор", "Кайрат", "Айбек", "Ерлан", "Марат"]

        for idx, chunk in enumerate(chunks, start=start_id):
            low = chunk.lower()
            if any(k in low for k in ["камер", "видео", "cam-", "запись"]):
                src_type, src_id, w, conf, mot = "камера", f"Видеокамера #{idx}", 0.95, 0.0, "Объективная видеофиксация"
            elif any(k in low for k in ["биллинг", "телефон", "вышк", "сотов"]):
                src_type, src_id, w, conf, mot = "биллинг", f"Биллинг #{idx}", 0.90, 0.0, "Телеком-след"
            elif any(k in low for k in ["турникет", "скуд", "пропуск"]):
                src_type, src_id, w, conf, mot = "турникет", f"СКУД #{idx}", 0.95, 0.0, "Аппаратный лог"
            elif any(k in low for k in ["подозреваем", "я не был", "не виновен"]):
                src_type, src_id, w, conf, mot = "подозреваемый", "Показания фигуранта", 0.35, 0.85, "Мотив защиты алиби"
            else:
                src_type, src_id, w, conf, mot = "свидетель", f"Свидетель #{idx}", 0.60, 0.20, "Свидетельские показания"

            range_match = time_range_re.findall(chunk)
            if range_match:
                t1_raw, t2_raw = range_match[0]
                t1 = f"{default_date} {t1_raw.replace('.', ':')}"
                t2 = f"{default_date} {t2_raw.replace('.', ':')}"
            else:
                s_match = single_time_re.findall(chunk)
                if s_match:
                    t_val = s_match[0].replace(".", ":")
                    t1 = f"{default_date} {t_val}"
                    parts = list(map(int, t_val.split(":")))
                    h, m = parts[0], parts[1]
                    m_end = (m + 15) % 60
                    h_end = h + (m + 15) // 60
                    sec_str = f":{parts[2]:02d}" if len(parts) == 3 else ""
                    t2 = f"{default_date} {h_end:02d}:{m_end:02d}{sec_str}"
                else:
                    t1, t2 = "", ""

            found_subject = "Неустановленный фигурант"
            for n in names_pool:
                if n.lower() in low:
                    found_subject = f"{n} С."
                    break

            matched_loc_name = None
            for loc_key in updated_locs:
                if loc_key.lower() in low:
                    matched_loc_name = loc_key
                    break

            if not matched_loc_name:
                words = re.findall(r"[А-ЯЁ][а-яё]{3,}", chunk)
                candidate = next((w for w in words if w not in names_pool), None)
                if candidate:
                    matched_loc_name = candidate
                    if candidate not in updated_locs:
                        updated_locs[candidate] = Location(candidate, None, None, "Локация требует калибровки координат")
                else:
                    matched_loc_name = list(updated_locs.keys())[0] if updated_locs else "Неизвестная зона"

            predicate = Predicate.ABSENT.value if any(neg in low for neg in ["не был", "отсутств", "не видел", "не находился"]) else Predicate.PRESENT.value

            new_facts.append(AtomicFact(
                fact_id=f"F-{idx:02d}",
                source_id=src_id, source_type=src_type,
                subject=found_subject, predicate=predicate,
                location_name=matched_loc_name,
                t_start=t1, t_end=t2, weight=w,
                source_excerpt=chunk, motive_flag=mot, interest_conflict=conf
            ))
        return new_facts, updated_locs

class ScientificValidator:
    @staticmethod
    def run_benchmark(engine: ForensicCollisionEngine, test_samples: int = 200, anomaly_rate: float = 0.5, add_noise: bool = True, seed: int = 42) -> Dict:
        random.seed(seed)
        locs = {
            "Точка Альфа": Location("Точка Альфа", 0.0, 0.0),
            "Точка Бета": Location("Точка Бета", 400.0, 300.0)
        }
        tp, fp, tn, fn = 0, 0, 0, 0
        base_time = datetime(2026, 10, 12, 12, 0, 0)
        critical_types = {
            CollisionType.SPATIAL_TEMPORAL.value,
            CollisionType.KINEMATIC_CRITICAL.value,
            CollisionType.KINEMATIC_VEHICLE_REQUIRED.value,
            CollisionType.DIRECT_CONTRADICTION.value
        }

        for i in range(test_samples):
            is_anomaly = random.random() < anomaly_rate
            case_time = base_time + timedelta(minutes=i * 20)
            noise_delta = timedelta(minutes=random.choice([-4, 0, 4])) if add_noise else timedelta(0)

            if is_anomaly:
                col_sub_type = random.choice(["bilocation", "kinematic", "contradiction"])
                if col_sub_type == "bilocation":
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Альфа", case_time.strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "Свидетель", "свидетель", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Бета", (case_time + timedelta(minutes=5) + noise_delta).strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S"), 0.5, "Лог Б")
                elif col_sub_type == "kinematic":
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Альфа", case_time.strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "СКУД Б", "турникет", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Бета", (case_time + timedelta(minutes=5, seconds=10)).strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"), 0.9, "Лог Б")
                else:
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Альфа", case_time.strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "Свидетель", "свидетель", f"Субъект_{i}", Predicate.ABSENT.value, "Точка Альфа", (case_time + noise_delta).strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"), 0.6, "Лог А")
            else:
                f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Альфа", case_time.strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"), 0.95, "Лог А")
                f2 = AtomicFact(f"T-{i}-2", "Камера Б", "камера", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Бета", (case_time + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S"), 0.95, "Лог Б")

            res = engine.analyze([f1, f2], locs)
            detected = any(r["type"] in critical_types for r in res)

            if is_anomaly and detected: tp += 1
            elif not is_anomaly and detected: fp += 1
            elif not is_anomaly and not detected: tn += 1
            elif is_anomaly and not detected: fn += 1

        eps = 1e-7
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        return {
            "total_cases": test_samples, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round(accuracy * 100, 2), "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2), "f1_score": round(f1_score * 100, 2)
        }

class DatabaseManager:
    @staticmethod
    def get_default_dataset() -> Tuple[Dict[str, Location], List[AtomicFact]]:
        locs = {
            "Кабинет 305": Location("Кабинет 305", 120.0, 40.0, "Зона лаборатории"),
            "Библиотека": Location("Библиотека", 300.0, 150.0, "Читальный зал"),
            "Центральный вход": Location("Центральный вход", 0.0, 0.0, "КПП и турникеты"),
            "Столовая": Location("Столовая", -50.0, 80.0, "Общественная зона"),
            "Парковка": Location("Парковка", 250.0, -100.0, "Северная автостоянка")
        }
        facts = [
            AtomicFact("F-01", "Протокол опроса фигуранта", "подозреваемый", "Арман С.", 
                       Predicate.PRESENT.value, "Библиотека", "2026-10-12 14:00", "2026-10-12 14:40", 
                       0.35, "С 14:00 до 14:40 находился в читальном зале библиотеки.", "Формирование алиби", 0.85),
            AtomicFact("F-02", "Камера CAM-305", "камера", "Арман С.", 
                       Predicate.PRESENT.value, "Кабинет 305", "2026-10-12 14:15", "2026-10-12 14:25", 
                       0.95, "Зафиксирован субъект схожей комплекции.", "Объективный видеоконтроль", 0.0),
            AtomicFact("F-03", "Показания Дамира", "свидетель", "Арман С.", 
                       Predicate.PRESENT.value, "Центральный вход", "2026-10-12 14:26", "2026-10-12 14:28", 
                       0.60, "Видел Армана у главного входа.", "Информационный свидетель", 0.15),
            AtomicFact("F-04", "Показания охранника", "свидетель", "Арман С.", 
                       Predicate.ABSENT.value, "Библиотека", "2026-10-12 14:10", "2026-10-12 14:35", 
                       0.75, "В помещении библиотеки посторонних не наблюдалось.", "Служебный контроль", 0.05)
        ]
        return locs, facts

    @staticmethod
    def load_data() -> Tuple[Dict[str, Location], List[AtomicFact]]:
        return DatabaseManager.get_default_dataset()
