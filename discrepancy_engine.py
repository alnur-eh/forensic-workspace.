from __future__ import annotations
import math, random, re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

TIME_FORMAT = "%Y-%m-%d %H:%M"

class Predicate(str, Enum):
    PRESENT = "находился"
    ABSENT = "отсутствовал"
    INTERACTED = "контактировал_с"
    SEEN = "видел_субъекта"

class CollisionType(str, Enum):
    SPATIAL_TEMPORAL = "ПРОСТРАНСТВЕННО-ВРЕМЕННОЙ ПАРАДОКС (БИЛОКАЦИЯ)"
    KINEMATIC = "КИНЕМАТИЧЕСКАЯ НЕВОЗМОЖНОСТЬ ПЕРЕМЕЩЕНИЯ"
    DIRECT_CONTRADICTION = "ПРЯМОЕ ЛОГИЧЕСКОЕ ПРОТИВОРЕЧИЕ ПОКАЗАНИЙ"
    SOURCE_RELIABILITY = "КРИТИЧЕСКИЙ ДИСБАЛАНС ДОСТОВЕРНОСТИ ИСТОЧНИКОВ"

@dataclass(frozen=True)
class Location:
    name: str
    x: float
    y: float
    description: str = ""

@dataclass
class AtomicFact:
    fact_id: str
    source_id: str
    source_type: str
    subject: str
    predicate: str
    object_target: Optional[str]
    location: Optional[Location]
    t_start: str
    t_end: str
    weight: float
    source_excerpt: str
    motive_flag: str = "Нейтральный"
    interest_conflict: float = 0.0

    def parse_start(self) -> datetime:
        return datetime.strptime(self.t_start, TIME_FORMAT)

    def parse_end(self) -> datetime:
        return datetime.strptime(self.t_end, TIME_FORMAT)

@dataclass
class AnalysisConfig:
    max_walking_speed_kmh: float = 5.0
    max_sprint_speed_kmh: float = 18.0
    max_vehicle_speed_kmh: float = 90.0
    same_location_radius_m: float = 2.0
    critical_weight_gap: float = 0.45

def calculate_distance(loc1: Location, loc2: Location) -> float:
    return math.hypot(loc1.x - loc2.x, loc1.y - loc2.y)

def get_allen_relation(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> str:
    if start_a == start_b and end_a == end_b: return "EQUALS"
    if end_a < start_b: return "BEFORE"
    if end_b < start_a: return "AFTER"
    if end_a == start_b: return "MEETS"
    if end_b == start_a: return "MET_BY"
    if start_b <= start_a and end_a <= end_b: return "DURING"
    if start_a <= start_b and end_b <= end_a: return "CONTAINS"
    if start_a < start_b < end_a < end_b: return "OVERLAPS"
    if start_b < start_a < end_b < end_a: return "OVERLAPPED_BY"
    return "UNKNOWN"

def is_overlapping(rel: str) -> bool:
    return rel in {"OVERLAPS", "OVERLAPPED_BY", "DURING", "CONTAINS", "EQUALS"}

class RawTextParser:
    @staticmethod
    def extract_facts_heuristic(text: str, default_date: str = "2026-10-12", locations_dict: Dict[str, Location] = None) -> List[AtomicFact]:
        facts = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        time_pattern = re.compile(r"(\d{1,2}[:.]\d{2})\s*(?:-|—|до)\s*(\d{1,2}[:.]\d{2})")
        single_time_pattern = re.compile(r"(?:в|около|примерно)\s*(\d{1,2}[:.]\d{2})")

        fact_counter = 1
        for line in lines:
            source_type, source_id, weight, conflict, motive = "свидетель", "Протокол опроса", 0.5, 0.2, "Информационный свидетель"
            lower_line = line.lower()
            if "камер" in lower_line or "видео" in lower_line:
                source_type, source_id, weight, conflict, motive = "камера", "Камера видеонаблюдения", 0.95, 0.0, "Объективный видеоконтроль"
            elif "биллинг" in lower_line or "телефон" in lower_line:
                source_type, source_id, weight, conflict, motive = "биллинг", "Телеком-биллинг", 0.90, 0.0, "Цифровой след"
            elif "подозреваем" in lower_line or ("арман" in lower_line and "утвержд" in lower_line):
                source_type, source_id, weight, conflict, motive = "подозреваемый", "Показания фигуранта", 0.35, 0.85, "Мотив алиби"

            times = time_pattern.findall(line)
            if times:
                t1_raw, t2_raw = times[0]
                t1_clean = t1_raw.replace(".", ":").zfill(5)
                t2_clean = t2_raw.replace(".", ":").zfill(5)
                t1 = default_date + " " + t1_clean
                t2 = default_date + " " + t2_clean
            else:
                single_t = single_time_pattern.findall(line)
                if single_t:
                    t1_val = single_t[0].replace(".", ":").zfill(5)
                    t1 = default_date + " " + t1_val
                    h, m = map(int, t1_val.split(":"))
                    m_end = (m + 15) % 60
                    h_end = h + (m + 15) // 60
                    t2 = default_date + " " + f"{h_end:02d}:{m_end:02d}"
                else:
                    t1 = default_date + " 14:00"
                    t2 = default_date + " 14:15"

            subject = "Неустановленный субъект"
            for name in ["Арман", "Дамир", "Нурлан", "Алихан", "Охранник", "Курьер"]:
                if name.lower() in lower_line:
                    subject = name + " С."
                    break

            matched_loc = None
            if locations_dict:
                for loc_name, loc_obj in locations_dict.items():
                    if loc_name.lower() in lower_line:
                        matched_loc = loc_obj
                        break
            if matched_loc is None and locations_dict:
                matched_loc = list(locations_dict.values())[0]

            predicate = Predicate.ABSENT.value if ("не был" in lower_line or "отсутств" in lower_line) else Predicate.PRESENT.value

            facts.append(AtomicFact(
                fact_id="AUTO-F" + f"{fact_counter:02d}",
                source_id=source_id, source_type=source_type, subject=subject,
                predicate=predicate, object_target=None, location=matched_loc,
                t_start=t1, t_end=t2, weight=weight, source_excerpt=line,
                motive_flag=motive, interest_conflict=conflict
            ))
            fact_counter += 1
        return facts

class ForensicCollisionEngine:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

    def analyze(self, facts: List[AtomicFact]) -> List[Dict]:
        collisions: List[Dict] = []
        n = len(facts)
        for i in range(n):
            for j in range(i + 1, n):
                f1, f2 = facts[i], facts[j]
                if f1.subject == f2.subject:
                    collisions.extend(self._evaluate_subject_pair(f1, f2))
                rel_col = self._evaluate_source_bias(f1, f2)
                if rel_col:
                    collisions.append(rel_col)
        return collisions

    def _evaluate_subject_pair(self, f1: AtomicFact, f2: AtomicFact) -> List[Dict]:
        results = []
        try:
            start1, end1 = f1.parse_start(), f1.parse_end()
            start2, end2 = f2.parse_start(), f2.parse_end()
        except Exception:
            return results

        relation = get_allen_relation(start1, end1, start2, end2)

        if is_overlapping(relation) and f1.location and f2.location and f1.location.name == f2.location.name:
            if {f1.predicate, f2.predicate} == {Predicate.PRESENT.value, Predicate.ABSENT.value}:
                results.append({
                    "id": f"COL-LOGIC-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.DIRECT_CONTRADICTION.value,
                    "subject": f1.subject,
                    "severity": "КРИТИЧЕСКАЯ",
                    "details": f"Прямой конфликт показаний в локации '{f1.location.name}'.",
                    "allen_relation": relation,
                    "facts": [f1, f2],
                    "psychological_insight": f"Высокая вероятность дезинформации (Дельта конфликта: {abs(f1.interest_conflict - f2.interest_conflict):.2f})."
                })

        if is_overlapping(relation) and f1.location and f2.location:
            dist = calculate_distance(f1.location, f2.location)
            if dist > self.config.same_location_radius_m and f1.predicate == Predicate.PRESENT.value and f2.predicate == Predicate.PRESENT.value:
                results.append({
                    "id": f"COL-ST-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.SPATIAL_TEMPORAL.value,
                    "subject": f1.subject,
                    "severity": "КРИТИЧЕСКАЯ",
                    "details": f"Одновременное присутствие в разных точках ({dist:.1f} м).",
                    "allen_relation": relation,
                    "facts": [f1, f2],
                    "psychological_insight": f"Ложное алиби в менее надежном источнике. Мотив: {f1.motive_flag if f1.weight < f2.weight else f2.motive_flag}."
                })

        if f1.location and f2.location and f1.predicate == Predicate.PRESENT.value and f2.predicate == Predicate.PRESENT.value:
            if end1 <= start2: earlier, later, e_end, l_start = f1, f2, end1, start2
            elif end2 <= start1: earlier, later, e_end, l_start = f2, f1, end2, start1
            else: earlier = None

            if earlier:
                dist = calculate_distance(earlier.location, later.location)
                gap_sec = max((l_start - e_end).total_seconds(), 1.0)
                speed_kmh = (dist / gap_sec) * 3.6
                if speed_kmh > self.config.max_sprint_speed_kmh:
                    results.append({
                        "id": f"COL-KIN-{f1.fact_id}-{f2.fact_id}",
                        "type": CollisionType.KINEMATIC.value,
                        "subject": f1.subject,
                        "severity": "ВЫСОКАЯ",
                        "details": f"Физически невозможное перемещение: {dist:.1f} м за {int(gap_sec)} с требует {speed_kmh:.1f} км/ч.",
                        "allen_relation": f"Интервал: {int(gap_sec)} с",
                        "facts": [f1, f2],
                        "psychological_insight": "Искажение хронометража свидетелем или сокрытие использования транспорта."
                    })
        return results

    def _evaluate_source_bias(self, f1: AtomicFact, f2: AtomicFact) -> Optional[Dict]:
        if f1.subject == f2.subject and f1.location != f2.location:
            diff = abs(f1.weight - f2.weight)
            if diff >= self.config.critical_weight_gap:
                low_src = f1 if f1.weight < f2.weight else f2
                high_src = f2 if f1.weight < f2.weight else f1
                return {
                    "id": f"COL-BIAS-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.SOURCE_RELIABILITY.value,
                    "subject": f1.subject,
                    "severity": "СРЕДНЯЯ",
                    "details": f"Слабый источник '{low_src.source_id}' противоречит объективному '{high_src.source_id}'.",
                    "allen_relation": f"Дельта веса: {diff:.2f}",
                    "facts": [f1, f2],
                    "psychological_insight": f"Уязвимый мотив: {low_src.motive_flag}."
                }
        return None

class ScientificValidator:
    @staticmethod
    def run_ground_truth_benchmark(engine: ForensicCollisionEngine, test_samples: int = 200, anomaly_rate: float = 0.5) -> Dict:
        loc_a = Location("Точка Альфа", 0.0, 0.0)
        loc_b = Location("Точка Бета", 400.0, 300.0)
        tp, fp, tn, fn = 0, 0, 0, 0
        base_time = datetime(2026, 10, 12, 12, 0)

        for i in range(test_samples):
            is_anomaly = random.random() < anomaly_rate
            case_time = base_time + timedelta(minutes=i * 20)
            if is_anomaly:
                col_sub_type = random.choice(["bilocation", "kinematic", "contradiction"])
                if col_sub_type == "bilocation":
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, None, loc_a, case_time.strftime(TIME_FORMAT), (case_time + timedelta(minutes=15)).strftime(TIME_FORMAT), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "Свидетель", "свидетель", f"Субъект_{i}", Predicate.PRESENT.value, None, loc_b, (case_time + timedelta(minutes=5)).strftime(TIME_FORMAT), (case_time + timedelta(minutes=20)).strftime(TIME_FORMAT), 0.5, "Лог Б")
                elif col_sub_type == "kinematic":
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, None, loc_a, case_time.strftime(TIME_FORMAT), (case_time + timedelta(minutes=5)).strftime(TIME_FORMAT), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "Турникет Б", "турникет", f"Субъект_{i}", Predicate.PRESENT.value, None, loc_b, (case_time + timedelta(minutes=5, seconds=10)).strftime(TIME_FORMAT), (case_time + timedelta(minutes=15)).strftime(TIME_FORMAT), 0.9, "Лог Б")
                else:
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, None, loc_a, case_time.strftime(TIME_FORMAT), (case_time + timedelta(minutes=15)).strftime(TIME_FORMAT), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "Свидетель", "свидетель", f"Субъект_{i}", Predicate.ABSENT.value, None, loc_a, case_time.strftime(TIME_FORMAT), (case_time + timedelta(minutes=15)).strftime(TIME_FORMAT), 0.6, "Лог А")
            else:
                f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, None, loc_a, case_time.strftime(TIME_FORMAT), (case_time + timedelta(minutes=10)).strftime(TIME_FORMAT), 0.95, "Лог А")
                f2 = AtomicFact(f"T-{i}-2", "Камера Б", "камера", f"Субъект_{i}", Predicate.PRESENT.value, None, loc_b, (case_time + timedelta(minutes=30)).strftime(TIME_FORMAT), (case_time + timedelta(minutes=45)).strftime(TIME_FORMAT), 0.95, "Лог Б")

            res = engine.analyze([f1, f2])
            detected = len(res) > 0
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
