from __future__ import annotations
import math, random, re, json, os
from dataclasses import dataclass, asdict
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

@dataclass
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
    location_name: str
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

class DatabaseManager:
    DB_FILE = "forensic_db.json"

    @classmethod
    def load_data(cls) -> tuple[Dict[str, Location], List[AtomicFact]]:
        if os.path.exists(cls.DB_FILE):
            try:
                with open(cls.DB_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    locs = {k: Location(**v) for k, v in data.get("locations", {}).items()}
                    facts = [AtomicFact(**item) for item in data.get("facts", [])]
                    return locs, facts
            except Exception:
                pass
        
        default_locs = {
            "Кабинет 305": Location("Кабинет 305", 120.0, 40.0, "Зона лаборатории"),
            "Библиотека": Location("Библиотека", 300.0, 150.0, "Читальный зал"),
            "Центральный вход": Location("Центральный вход", 0.0, 0.0, "КПП и турникеты"),
            "Столовая": Location("Столовая", -50.0, 80.0, "Общественная зона"),
            "Парковка": Location("Парковка", 250.0, -100.0, "Северная автостоянка")
        }
        default_facts = [
            AtomicFact("F-01", "Протокол подозреваемого", "подозреваемый", "Арман С.", 
                       Predicate.PRESENT.value, None, "Библиотека", 
                       "2026-10-12 14:00", "2026-10-12 14:40", 0.3, "С 14:00 до 14:40 находился в библиотеке.", "Попытка алиби", 0.85),
            AtomicFact("F-02", "Камера CAM-305", "камера", "Арман С.", 
                       Predicate.PRESENT.value, None, "Кабинет 305", 
                       "2026-10-12 14:15", "2026-10-12 14:25", 0.95, "Зафиксирован человек в темной куртке.", "Объективный контроль", 0.0),
            AtomicFact("F-03", "Показания Дамира", "свидетель", "Арман С.", 
                       Predicate.PRESENT.value, None, "Центральный вход", 
                       "2026-10-12 14:26", "2026-10-12 14:28", 0.6, "Видел Армана у главного входа.", "Нейтральный свидетель", 0.2),
            AtomicFact("F-04", "Показания охранника", "свидетель", "Арман С.", 
                       Predicate.ABSENT.value, None, "Библиотека", 
                       "2026-10-12 14:10", "2026-10-12 14:35", 0.75, "В читальном зале никого не было.", "Служебный контроль", 0.1)
        ]
        return default_locs, default_facts

    @classmethod
    def save_data(cls, locations: Dict[str, Location], facts: List[AtomicFact]):
        data = {
            "locations": {k: asdict(v) for k, v in locations.items()},
            "facts": [asdict(f) for f in facts]
        }
        with open(cls.DB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

class SmartFreeTextParser:
    @staticmethod
    def parse_witness_statement(text: str, default_date: str, current_locs: Dict[str, Location]) -> tuple[List[AtomicFact], Dict[str, Location]]:
        new_facts = []
        updated_locs = dict(current_locs)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 5]

        time_range_re = re.compile(r"(\d{1,2}[:.]\d{2})\s*(?:-|—|до)\s*(\d{1,2}[:.]\d{2})")
        single_time_re = re.compile(r"(?:в|около|примерно|после)\s*(\d{1,2}[:.]\d{2})")

        for idx, line in enumerate(lines, 1):
            low = line.lower()
            
            if "камер" in low or "видео" in low:
                src_type, src_id, w, conf, mot = "камера", "Система видеоаналитики", 0.95, 0.0, "Объективная фиксация"
            elif "биллинг" in low or "телефон" in low or "вышк" in low:
                src_type, src_id, w, conf, mot = "биллинг", "Телекоммуникационный лог", 0.90, 0.0, "Цифровой след"
            elif "подозреваем" in low or "я не виновен" in low:
                src_type, src_id, w, conf, mot = "подозреваемый", "Показания допрашиваемого", 0.35, 0.90, "Мотив алиби"
            elif "эксперт" in low or "акт" in low:
                src_type, src_id, w, conf, mot = "экспертиза", "Заключение эксперта", 0.98, 0.0, "Научная экспертиза"
            else:
                src_type, src_id, w, conf, mot = "свидетель", f"Показания очевидца #{idx}", 0.60, 0.25, "Информационное свидетельство"

            range_match = time_range_re.findall(line)
            if range_match:
                t1_raw, t2_raw = range_match[0]
                t1 = default_date + " " + t1_raw.replace(".", ":").zfill(5)
                t2 = default_date + " " + t2_raw.replace(".", ":").zfill(5)
            else:
                s_match = single_time_re.findall(line)
                if s_match:
                    t_val = s_match[0].replace(".", ":").zfill(5)
                    t1 = default_date + " " + t_val
                    h, m = map(int, t_val.split(":"))
                    m_end = (m + 20) % 60
                    h_end = h + (m + 20) // 60
                    t2 = default_date + " " + f"{h_end:02d}:{m_end:02d}"
                else:
                    t1 = default_date + " 14:00"
                    t2 = default_date + " 14:20"

            names_pool = ["Арман", "Дамир", "Нурлан", "Алихан", "Охранник", "Курьер", "Директор", "Кайрат", "Айбек"]
            found_subject = "Неустановленный фигурант"
            for n in names_pool:
                if n.lower() in low:
                    found_subject = n + " С."
                    break

            matched_loc_name = None
            for loc_key in updated_locs:
                if loc_key.lower() in low:
                    matched_loc_name = loc_key
                    break
            
            if not matched_loc_name:
                words = re.findall(r"[А-ЯЁ][а-яё]+", line)
                candidate = None
                for w_item in words:
                    if w_item not in names_pool and len(w_item) > 3:
                        candidate = w_item
                        break
                if candidate:
                    matched_loc_name = candidate
                    if candidate not in updated_locs:
                        rx = round(random.uniform(-100.0, 350.0), 1)
                        ry = round(random.uniform(-100.0, 200.0), 1)
                        updated_locs[candidate] = Location(candidate, rx, ry, "Автоматически добавленная точка")
                else:
                    matched_loc_name = list(updated_locs.keys())[0]

            predicate = Predicate.ABSENT.value if ("не был" in low or "отсутств" in low or "не видел" in low) else Predicate.PRESENT.value

            new_facts.append(AtomicFact(
                fact_id=f"NLP-F{idx:02d}",
                source_id=src_id, source_type=src_type,
                subject=found_subject, predicate=predicate,
                object_target=None, location_name=matched_loc_name,
                t_start=t1, t_end=t2, weight=w,
                source_excerpt=line, motive_flag=mot, interest_conflict=conf
            ))

        return new_facts, updated_locs

class ForensicCollisionEngine:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

    def analyze(self, facts: List[AtomicFact], locations: Dict[str, Location]) -> List[Dict]:
        collisions: List[Dict] = []
        n = len(facts)
        for i in range(n):
            for j in range(i + 1, n):
                f1, f2 = facts[i], facts[j]
                if f1.subject == f2.subject:
                    collisions.extend(self._evaluate_subject_pair(f1, f2, locations))
                rel_col = self._evaluate_source_bias(f1, f2)
                if rel_col:
                    collisions.append(rel_col)
        return collisions

    def _evaluate_subject_pair(self, f1: AtomicFact, f2: AtomicFact, locations: Dict[str, Location]) -> List[Dict]:
        results = []
        try:
            start1, end1 = f1.parse_start(), f1.parse_end()
            start2, end2 = f2.parse_start(), f2.parse_end()
        except Exception:
            return results

        relation = get_allen_relation(start1, end1, start2, end2)
        loc1 = locations.get(f1.location_name)
        loc2 = locations.get(f2.location_name)

        if is_overlapping(relation) and loc1 and loc2 and loc1.name == loc2.name:
            if {f1.predicate, f2.predicate} == {Predicate.PRESENT.value, Predicate.ABSENT.value}:
                results.append({
                    "id": f"COL-LOGIC-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.DIRECT_CONTRADICTION.value,
                    "subject": f1.subject, "severity": "КРИТИЧЕСКАЯ",
                    "details": f"Прямой конфликт показаний в локации '{loc1.name}'.",
                    "allen_relation": relation, "facts": [f1, f2],
                    "psychological_insight": f"Высокая вероятность дезинформации (Дельта конфликта: {abs(f1.interest_conflict - f2.interest_conflict):.2f})."
                })

        if is_overlapping(relation) and loc1 and loc2:
            dist = calculate_distance(loc1, loc2)
            if dist > self.config.same_location_radius_m and f1.predicate == Predicate.PRESENT.value and f2.predicate == Predicate.PRESENT.value:
                results.append({
                    "id": f"COL-ST-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.SPATIAL_TEMPORAL.value,
                    "subject": f1.subject, "severity": "КРИТИЧЕСКАЯ",
                    "details": f"Одновременное присутствие в разных точках ({dist:.1f} м).",
                    "allen_relation": relation, "facts": [f1, f2],
                    "psychological_insight": f"Ложное алиби в менее надежном источнике. Мотив: {f1.motive_flag if f1.weight < f2.weight else f2.motive_flag}."
                })

        if loc1 and loc2 and f1.predicate == Predicate.PRESENT.value and f2.predicate == Predicate.PRESENT.value:
            if end1 <= start2: earlier, later, e_end, l_start = f1, f2, end1, start2
            elif end2 <= start1: earlier, later, e_end, l_start = f2, f1, end2, start1
            else: earlier = None

            if earlier:
                loc_e = locations.get(earlier.location_name)
                loc_l = locations.get(later.location_name)
                if loc_e and loc_l:
                    dist = calculate_distance(loc_e, loc_l)
                    gap_sec = max((l_start - e_end).total_seconds(), 1.0)
                    speed_kmh = (dist / gap_sec) * 3.6
                    if speed_kmh > self.config.max_sprint_speed_kmh:
                        results.append({
                            "id": f"COL-KIN-{f1.fact_id}-{f2.fact_id}",
                            "type": CollisionType.KINEMATIC.value,
                            "subject": f1.subject, "severity": "ВЫСОКАЯ",
                            "details": f"Физически невозможное перемещение: {dist:.1f} м за {int(gap_sec)} с требует {speed_kmh:.1f} км/ч.",
                            "allen_relation": f"Интервал: {int(gap_sec)} с",
                            "facts": [f1, f2],
                            "psychological_insight": "Искажение хронометража очевидцем или сокрытие использования транспорта."
                        })
        return results

    def _evaluate_source_bias(self, f1: AtomicFact, f2: AtomicFact) -> Optional[Dict]:
        if f1.subject == f2.subject and f1.location_name != f2.location_name:
            diff = abs(f1.weight - f2.weight)
            if diff >= self.config.critical_weight_gap:
                low_src = f1 if f1.weight < f2.weight else f2
                high_src = f2 if f1.weight < f2.weight else f1
                return {
                    "id": f"COL-BIAS-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.SOURCE_RELIABILITY.value,
                    "subject": f1.subject, "severity": "СРЕДНЯЯ",
                    "details": f"Слабый источник '{low_src.source_id}' противоречит объективному '{high_src.source_id}'.",
                    "allen_relation": f"Дельта веса: {diff:.2f}",
                    "facts": [f1, f2],
                    "psychological_insight": f"Уязвимый мотив: {low_src.motive_flag}."
                }
        return None

class ScientificValidator:
    @staticmethod
    def run_ground_truth_benchmark(engine: ForensicCollisionEngine, test_samples: int = 200, anomaly_rate: float = 0.5) -> Dict:
        locs = {
            "Точка Альфа": Location("Точка Альфа", 0.0, 0.0),
            "Точка Бета": Location("Точка Бета", 400.0, 300.0)
        }
        tp, fp, tn, fn = 0, 0, 0, 0
        base_time = datetime(2026, 10, 12, 12, 0)

        for i in range(test_samples):
            is_anomaly = random.random() < anomaly_rate
            case_time = base_time + timedelta(minutes=i * 20)
            if is_anomaly:
                col_sub_type = random.choice(["bilocation", "kinematic", "contradiction"])
                if col_sub_type == "bilocation":
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, None, "Точка Альфа", case_time.strftime(TIME_FORMAT), (case_time + timedelta(minutes=15)).strftime(TIME_FORMAT), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "Свидетель", "свидетель", f"Субъект_{i}", Predicate.PRESENT.value, None, "Точка Бета", (case_time + timedelta(minutes=5)).strftime(TIME_FORMAT), (case_time + timedelta(minutes=20)).strftime(TIME_FORMAT), 0.5, "Лог Б")
                elif col_sub_type == "kinematic":
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, None, "Точка Альфа", case_time.strftime(TIME_FORMAT), (case_time + timedelta(minutes=5)).strftime(TIME_FORMAT), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "Турникет Б", "турникет", f"Субъект_{i}", Predicate.PRESENT.value, None, "Точка Бета", (case_time + timedelta(minutes=5, seconds=10)).strftime(TIME_FORMAT), (case_time + timedelta(minutes=15)).strftime(TIME_FORMAT), 0.9, "Лог Б")
                else:
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, None, "Точка Альфа", case_time.strftime(TIME_FORMAT), (case_time + timedelta(minutes=15)).strftime(TIME_FORMAT), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "Свидетель", "свидетель", f"Субъект_{i}", Predicate.ABSENT.value, None, "Точка Альфа", case_time.strftime(TIME_FORMAT), (case_time + timedelta(minutes=15)).strftime(TIME_FORMAT), 0.6, "Лог А")
            else:
                f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, None, "Точка Альфа", case_time.strftime(TIME_FORMAT), (case_time + timedelta(minutes=10)).strftime(TIME_FORMAT), 0.95, "Лог А")
                f2 = AtomicFact(f"T-{i}-2", "Камера Б", "камера", f"Субъект_{i}", Predicate.PRESENT.value, None, "Точка Бета", (case_time + timedelta(minutes=30)).strftime(TIME_FORMAT), (case_time + timedelta(minutes=45)).strftime(TIME_FORMAT), 0.95, "Лог Б")

            res = engine.analyze([f1, f2], locs)
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
