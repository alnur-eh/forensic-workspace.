"""
AI Forensic Workspace — Все в одном файле (Monolithic Standalone Edition)
Математическое ядро Аллена, кинематика, цепочки аудита и Streamlit-интерфейс.
"""
from __future__ import annotations
import math
import random
import re
import json
import os
import time
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pyvis.network import Network
import plotly.express as px

# -------------------------------------------------------------
# 1. МАТЕМАТИЧЕСКИЕ СТРУКТУРЫ И СУДЕБНЫЕ ТИПЫ ДАННЫХ
# -------------------------------------------------------------

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

class ConflictStatus(str, Enum):
    CONFIRMED = "ПОДТВЕРЖДЁННАЯ"
    POSSIBLE = "ВОЗМОЖНАЯ"
    INSUFFICIENT_DATA = "НЕДОСТАТОЧНО ДАННЫХ"

class CollisionType(str, Enum):
    SPATIAL_TEMPORAL = "ПРОСТРАНСТВЕННО-ВРЕМЕННАЯ НЕСОГЛАСОВАННОСТЬ (БИЛОКАЦИЯ)"
    KINEMATIC_VEHICLE_REQUIRED = "КИНЕМАТИКА: ТРЕБУЕТСЯ АВТОТРАНСПОРТ"
    KINEMATIC_CRITICAL = "КИНЕМАТИЧЕСКАЯ АНОМАЛИЯ (ПРЕВЫШЕНИЕ СКОРОСТИ / ТЕЛЕПОРТАЦИЯ)"
    DIRECT_CONTRADICTION = "ПРЯМОЕ ЛОГИЧЕСКОЕ ПРОТИВОРЕЧИЕ УТВЕРЖДЕНИЙ"
    SOURCE_RELIABILITY = "ДИСБАЛАНС ВЕСОВ КОНФЛИКТУЮЩИХ ИСТОЧНИКОВ"
    INSUFFICIENT_SPATIAL_DATA = "НЕДОСТАТОЧНО ПРОСТРАНСТВЕННЫХ ДАННЫХ ДЛЯ ПРОВЕРКИ"

ALLEN_INVERSES: Dict[str, str] = {
    "BEFORE": "AFTER", "AFTER": "BEFORE", "MEETS": "MET_BY", "MET_BY": "MEETS",
    "STARTS": "STARTED_BY", "STARTED_BY": "STARTS", "FINISHES": "FINISHED_BY",
    "FINISHED_BY": "FINISHES", "DURING": "CONTAINS", "CONTAINS": "DURING",
    "OVERLAPS": "OVERLAPPED_BY", "OVERLAPPED_BY": "OVERLAPS", "EQUALS": "EQUALS"
}

@dataclass
class TimeInterval:
    start: datetime
    end: datetime
    uncertainty_sec: float = 0.0

    def get_raw_interval(self) -> Tuple[datetime, datetime]:
        return (self.start, self.end)

    def get_effective_interval(self) -> Tuple[datetime, datetime]:
        delta = timedelta(seconds=max(0.0, self.uncertainty_sec))
        return (self.start - delta, self.end + delta)

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

    def get_time_interval(self) -> Optional[TimeInterval]:
        s, e = self.parse_start(), self.parse_end()
        if s is None or e is None or s > e:
            return None
        return TimeInterval(start=s, end=e, uncertainty_sec=self.time_uncertainty_sec)

    def is_valid_interval(self) -> bool:
        return self.get_time_interval() is not None

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
    if start_a < start_b and end_a < end_b: return "OVERLAPS"
    if start_b < start_a and end_b < end_a: return "OVERLAPPED_BY"
    return "UNKNOWN"

class ConfidenceCalculator:
    @staticmethod
    def calculate(f1: AtomicFact, f2: AtomicFact, status: ConflictStatus, collision_type: CollisionType, has_coords: bool, kinematic_ratio: float = 1.0) -> Tuple[float, str, Dict[str, float]]:
        source_rel = max(0.1, min(1.0, (f1.weight + f2.weight) / 2.0))
        bias_penalty = max(0.0, min(0.3, (f1.interest_conflict + f2.interest_conflict) * 0.15))
        source_factor = max(0.1, source_rel - bias_penalty)
        max_unc = max(f1.time_uncertainty_sec, f2.time_uncertainty_sec)
        temporal_factor = max(0.2, 1.0 - min(0.8, max_unc / 1800.0))
        spatial_factor = 1.0 if has_coords else 0.0
        status_factor = 1.0 if status == ConflictStatus.CONFIRMED else (0.65 if status == ConflictStatus.POSSIBLE else 0.3)
        kin_factor = min(1.0, kinematic_ratio / 2.0) if kinematic_ratio > 1.0 else 1.0

        if collision_type in {CollisionType.KINEMATIC_CRITICAL, CollisionType.KINEMATIC_VEHICLE_REQUIRED, CollisionType.SPATIAL_TEMPORAL}:
            raw_score = (source_factor * 0.35 + temporal_factor * 0.25 + spatial_factor * 0.20 + status_factor * 0.20) * kin_factor
        elif collision_type == CollisionType.INSUFFICIENT_SPATIAL_DATA:
            raw_score = 0.30
        else:
            raw_score = (source_factor * 0.45 + temporal_factor * 0.30 + status_factor * 0.25)

        confidence = round(max(0.1, min(0.99, raw_score)), 2)
        label = "ВЫСОКАЯ" if confidence >= 0.80 else ("СРЕДНЯЯ" if confidence >= 0.50 else "НИЗКАЯ")
        factors = {
            "source_reliability": round(source_factor, 2),
            "temporal_precision": round(temporal_factor, 2),
            "spatial_precision": round(spatial_factor, 2),
            "status_weight": round(status_factor, 2)
        }
        return confidence, label, factors

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
        ti1 = f1.get_time_interval()
        ti2 = f2.get_time_interval()
        if not ti1 or not ti2: return results

        raw_s1, raw_e1 = ti1.get_raw_interval()
        raw_s2, raw_e2 = ti2.get_raw_interval()
        eff_s1, eff_e1 = ti1.get_effective_interval()
        eff_s2, eff_e2 = ti2.get_effective_interval()

        raw_overlap = (raw_s1 < raw_e2) and (raw_s2 < raw_e1)
        effective_overlap = (eff_s1 < eff_e2) and (eff_s2 < eff_e1)
        allen_rel = get_allen_relation(raw_s1, raw_e1, raw_s2, raw_e2)
        loc1 = locations.get(f1.location_name)
        loc2 = locations.get(f2.location_name)

        # 1. Прямое противоречие
        if effective_overlap and loc1 and loc2 and loc1.name == loc2.name:
            if {f1.predicate, f2.predicate} == {Predicate.PRESENT.value, Predicate.ABSENT.value}:
                status = ConflictStatus.CONFIRMED if raw_overlap else ConflictStatus.POSSIBLE
                conf, conf_label, conf_factors = ConfidenceCalculator.calculate(f1, f2, status, CollisionType.DIRECT_CONTRADICTION, has_coords=True)
                results.append({
                    "id": f"COL-LOGIC-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.DIRECT_CONTRADICTION.value,
                    "status": status.value,
                    "subject": f1.subject,
                    "severity": "ВЫСОКАЯ",
                    "confidence": conf, "confidence_label": conf_label, "confidence_factors": conf_factors,
                    "details": f"Взаимоисключающие утверждения о присутствии в '{loc1.name}'.",
                    "allen_relation": allen_rel,
                    "evidence_chain": [
                        f"Факт A ({f1.fact_id}): '{f1.subject}' {f1.predicate} в '{loc1.name}' [{f1.t_start} — {f1.t_end}].",
                        f"Факт B ({f2.fact_id}): '{f2.subject}' {f2.predicate} в '{loc2.name}' [{f2.t_start} — {f2.t_end}].",
                        f"Отношение Аллена: {allen_rel}. Противоположные утверждения в одной точке.",
                        f"Статус: {status.value} коллизия (Confidence {int(conf*100)}%)."
                    ],
                    "calculation": {"location_match": True, "raw_overlap": raw_overlap, "effective_overlap": effective_overlap},
                    "facts": [f1, f2],
                    "expert_note": f"Требуется перекрестный допрос. Дельта конфликта: {abs(f1.interest_conflict - f2.interest_conflict):.2f}.",
                    "limitations": "Вывод базируется на предположении о неизменности смысла показаний."
                })

        # 2. Билокация
        if effective_overlap and loc1 and loc2 and loc1.name != loc2.name:
            if not loc1.has_coordinates or not loc2.has_coordinates:
                conf, conf_label, conf_factors = ConfidenceCalculator.calculate(f1, f2, ConflictStatus.INSUFFICIENT_DATA, CollisionType.INSUFFICIENT_SPATIAL_DATA, has_coords=False)
                results.append({
                    "id": f"COL-NODATA-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.INSUFFICIENT_SPATIAL_DATA.value,
                    "status": ConflictStatus.INSUFFICIENT_DATA.value,
                    "subject": f1.subject, "severity": "ИНФОРМАЦИОННАЯ",
                    "confidence": conf, "confidence_label": conf_label, "confidence_factors": conf_factors,
                    "details": f"Перекрытие во времени между '{loc1.name}' и '{loc2.name}', но координаты не заданы.",
                    "allen_relation": allen_rel,
                    "evidence_chain": ["Пространственная проверка невозможна: отсутствует метрическая калибровка плана."],
                    "calculation": {"coordinates_calibrated": False},
                    "facts": [f1, f2],
                    "expert_note": "Задайте координаты локаций на 2D-карте для выполнения расчета билокации.",
                    "limitations": "Проверка алиби не завершена из-за отсутствия координат."
                })
            else:
                dist = calculate_distance(loc1, loc2)
                if dist is not None and dist > self.config.same_location_radius_m and f1.predicate == Predicate.PRESENT.value and f2.predicate == Predicate.PRESENT.value:
                    status = ConflictStatus.CONFIRMED if raw_overlap else ConflictStatus.POSSIBLE
                    conf, conf_label, conf_factors = ConfidenceCalculator.calculate(f1, f2, status, CollisionType.SPATIAL_TEMPORAL, has_coords=True)
                    results.append({
                        "id": f"COL-ST-{f1.fact_id}-{f2.fact_id}",
                        "type": CollisionType.SPATIAL_TEMPORAL.value,
                        "status": status.value, "subject": f1.subject, "severity": "КРИТИЧЕСКАЯ",
                        "confidence": conf, "confidence_label": conf_label, "confidence_factors": conf_factors,
                        "details": f"Одновременное нахождение в разных точках ({dist:.1f} м > порога {self.config.same_location_radius_m:.1f} м).",
                        "allen_relation": allen_rel,
                        "evidence_chain": [
                            f"Факт A ({f1.fact_id}): '{loc1.name}' [{f1.t_start} — {f1.t_end}].",
                            f"Факт B ({f2.fact_id}): '{loc2.name}' [{f2.t_start} — {f2.t_end}].",
                            f"Дистанция = {dist:.1f} м. Отношение Аллена: {allen_rel}.",
                            f"Статус: {status.value} билокация (Confidence {int(conf*100)}%)."
                        ],
                        "calculation": {"distance_m": round(dist, 1), "raw_overlap": raw_overlap, "effective_overlap": effective_overlap},
                        "facts": [f1, f2],
                        "expert_note": "Несогласованность временных меток объективного контроля или показаний.",
                        "limitations": "Вывод зависит от синхронизации системных часов регистраторов."
                    })

                    diff = abs(f1.weight - f2.weight)
                    if diff >= self.config.critical_weight_gap:
                        low_src = f1 if f1.weight < f2.weight else f2
                        high_src = f2 if f1.weight < f2.weight else f1
                        results.append({
                            "id": f"COL-BIAS-{f1.fact_id}-{f2.fact_id}",
                            "type": CollisionType.SOURCE_RELIABILITY.value,
                            "status": status.value, "subject": f1.subject, "severity": "СРЕДНЯЯ",
                            "confidence": conf, "confidence_label": conf_label, "confidence_factors": conf_factors,
                            "details": f"Источник низкой надежности '{low_src.source_id}' ({low_src.weight:.2f}) противоречит объективному '{high_src.source_id}' ({high_src.weight:.2f}).",
                            "allen_relation": f"Дельта весов: {diff:.2f}",
                            "evidence_chain": [f"Дельта весов = {diff:.2f} превышает порог {self.config.critical_weight_gap:.2f}."],
                            "calculation": {"weight_delta": round(diff, 2)},
                            "facts": [f1, f2],
                            "expert_note": f"Мотивационный профиль источника: {low_src.motive_flag}.",
                            "limitations": "Веса источников назначены на основе криминалистических шкал."
                        })

        # 3. Кинематика
        if loc1 and loc2 and loc1.name != loc2.name and f1.predicate == Predicate.PRESENT.value and f2.predicate == Predicate.PRESENT.value:
            if eff_e1 <= eff_s2: earlier, later, e_end, l_start, raw_e_end, raw_l_start = f1, f2, eff_e1, eff_s2, raw_e1, raw_s2
            elif eff_e2 <= eff_s1: earlier, later, e_end, l_start, raw_e_end, raw_l_start = f2, f1, eff_e2, eff_s1, raw_e2, raw_s1
            else: earlier = None

            if earlier:
                loc_e = locations.get(earlier.location_name)
                loc_l = locations.get(later.location_name)
                if loc_e and loc_l and loc_e.has_coordinates and loc_l.has_coordinates:
                    dist = calculate_distance(loc_e, loc_l)
                    gap_sec = (l_start - e_end).total_seconds()
                    raw_gap_sec = (raw_l_start - raw_e_end).total_seconds()

                    if dist is not None and dist > self.config.same_location_radius_m:
                        if gap_sec == 0:
                            status = ConflictStatus.CONFIRMED if raw_gap_sec == 0 else ConflictStatus.POSSIBLE
                            conf, conf_label, conf_factors = ConfidenceCalculator.calculate(f1, f2, status, CollisionType.KINEMATIC_CRITICAL, has_coords=True, kinematic_ratio=10.0)
                            results.append({
                                "id": f"COL-KIN-ZERO-{f1.fact_id}-{f2.fact_id}",
                                "type": CollisionType.KINEMATIC_CRITICAL.value,
                                "status": status.value, "subject": f1.subject, "severity": "КРИТИЧЕСКАЯ",
                                "confidence": conf, "confidence_label": conf_label, "confidence_factors": conf_factors,
                                "details": f"Мгновенное перемещение между '{loc_e.name}' и '{loc_l.name}' (дистанция {dist:.1f} м за 0.0 с).",
                                "allen_relation": "MEETS (Временной зазор: 0.0 с)",
                                "evidence_chain": [f"Расстояние = {dist:.1f} м. Временной зазор = 0.0 с -> Требуемая скорость = бесконечность."],
                                "calculation": {"distance_m": round(dist, 1), "time_gap_sec": 0.0, "required_speed_kmh": float("inf")},
                                "facts": [f1, f2],
                                "expert_note": "Физически невозможное перемещение между точками.",
                                "limitations": "Вывод основан на синхронности меток времени."
                            })
                        elif gap_sec > 0:
                            speed_kmh = (dist / gap_sec) * 3.6
                            raw_speed_kmh = (dist / raw_gap_sec) * 3.6 if raw_gap_sec > 0 else float("inf")
                            if speed_kmh > self.config.max_vehicle_speed_kmh:
                                status = ConflictStatus.CONFIRMED if raw_speed_kmh > self.config.max_vehicle_speed_kmh else ConflictStatus.POSSIBLE
                                conf, conf_label, conf_factors = ConfidenceCalculator.calculate(f1, f2, status, CollisionType.KINEMATIC_CRITICAL, has_coords=True, kinematic_ratio=speed_kmh/self.config.max_vehicle_speed_kmh)
                                results.append({
                                    "id": f"COL-KIN-CRIT-{f1.fact_id}-{f2.fact_id}",
                                    "type": CollisionType.KINEMATIC_CRITICAL.value,
                                    "status": status.value, "subject": f1.subject, "severity": "КРИТИЧЕСКАЯ",
                                    "confidence": conf, "confidence_label": conf_label, "confidence_factors": conf_factors,
                                    "details": f"Расчетная скорость {speed_kmh:.1f} км/ч превышает транспортный порог ({self.config.max_vehicle_speed_kmh:.1f} км/ч). Дистанция {dist:.1f} м за {gap_sec:.1f} с.",
                                    "allen_relation": f"Зазор: {gap_sec:.1f} с",
                                    "evidence_chain": [f"Дистанция {dist:.1f} м за {gap_sec:.1f} с -> Скорость {speed_kmh:.1f} км/ч."],
                                    "calculation": {"distance_m": round(dist, 1), "time_gap_sec": round(gap_sec, 1), "required_speed_kmh": round(speed_kmh, 1)},
                                    "facts": [f1, f2],
                                    "expert_note": "Аномальная скорость перемещения.",
                                    "limitations": "Расчет по евклидовой метрике (кратчайший путь)."
                                })
                            elif speed_kmh > self.config.max_sprint_speed_kmh:
                                status = ConflictStatus.CONFIRMED if raw_speed_kmh > self.config.max_sprint_speed_kmh else ConflictStatus.POSSIBLE
                                conf, conf_label, conf_factors = ConfidenceCalculator.calculate(f1, f2, status, CollisionType.KINEMATIC_VEHICLE_REQUIRED, has_coords=True, kinematic_ratio=speed_kmh/self.config.max_sprint_speed_kmh)
                                results.append({
                                    "id": f"COL-KIN-VEH-{f1.fact_id}-{f2.fact_id}",
                                    "type": CollisionType.KINEMATIC_VEHICLE_REQUIRED.value,
                                    "status": status.value, "subject": f1.subject, "severity": "ВЫСОКАЯ",
                                    "confidence": conf, "confidence_label": conf_label, "confidence_factors": conf_factors,
                                    "details": f"Расчетная скорость {speed_kmh:.1f} км/ч превышает порог бега ({self.config.max_sprint_speed_kmh:.1f} км/ч), допустима для транспорта (дистанция {dist:.1f} м за {gap_sec:.1f} с).",
                                    "allen_relation": f"Зазор: {gap_sec:.1f} с",
                                    "evidence_chain": [f"Пешком переместиться невозможно ({speed_kmh:.1f} км/ч). Требуется автотранспорт."],
                                    "calculation": {"distance_m": round(dist, 1), "time_gap_sec": round(gap_sec, 1), "required_speed_kmh": round(speed_kmh, 1)},
                                    "facts": [f1, f2],
                                    "expert_note": "Пешее перемещение исключено. Требуется подтверждение автотранспорта.",
                                    "limitations": "Не учитывает задержки на пропускных пунктах."
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
                src_type, src_id, w, conf, mot, unc = "камера", f"Видеокамера #{idx}", 0.95, 0.0, "Объективная видеофиксация", 10.0
            elif any(k in low for k in ["биллинг", "телефон", "вышк", "сотов"]):
                src_type, src_id, w, conf, mot, unc = "биллинг", f"Биллинг #{idx}", 0.90, 0.0, "Телеком-след", 60.0
            elif any(k in low for k in ["турникет", "скуд", "пропуск"]):
                src_type, src_id, w, conf, mot, unc = "турникет", f"СКУД #{idx}", 0.95, 0.0, "Аппаратный лог", 5.0
            elif any(k in low for k in ["подозреваем", "я не был", "не виновен"]):
                src_type, src_id, w, conf, mot, unc = "подозреваемый", "Показания фигуранта", 0.35, 0.85, "Мотив защиты алиби", 180.0
            else:
                src_type, src_id, w, conf, mot, unc = "свидетель", f"Свидетель #{idx}", 0.60, 0.20, "Свидетельские показания", 120.0

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
                source_excerpt=chunk, motive_flag=mot, interest_conflict=conf,
                time_uncertainty_sec=unc
            ))
        return new_facts, updated_locs

class ScientificValidator:
    @staticmethod
    def run_synthetic_benchmark(engine: ForensicCollisionEngine, test_samples: int = 200, anomaly_rate: float = 0.5, add_noise: bool = True, seed: int = 42) -> Dict:
        random.seed(seed)
        locs = {"Точка Альфа": Location("Точка Альфа", 0.0, 0.0), "Точка Бета": Location("Точка Бета", 400.0, 300.0)}
        tp, fp, tn, fn = 0, 0, 0, 0
        base_time = datetime(2026, 10, 12, 12, 0, 0)
        critical_types = {CollisionType.SPATIAL_TEMPORAL.value, CollisionType.KINEMATIC_CRITICAL.value, CollisionType.KINEMATIC_VEHICLE_REQUIRED.value, CollisionType.DIRECT_CONTRADICTION.value}

        for i in range(test_samples):
            is_anomaly = random.random() < anomaly_rate
            case_time = base_time + timedelta(minutes=i * 20)
            noise_delta = timedelta(minutes=random.choice([-4, 0, 4])) if add_noise else timedelta(0)

            if is_anomaly:
                col_sub_type = random.choice(["bilocation", "kinematic", "contradiction"])
                if col_sub_type == "bilocation":
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Альфа", case_time.strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "Свидетель", "свидетель", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Бета", (case_time + timedelta(minutes=5) + noise_delta).strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S"), 0.5, "Лог Б", time_uncertainty_sec=60.0)
                elif col_sub_type == "kinematic":
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Альфа", case_time.strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "СКУД Б", "турникет", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Бета", (case_time + timedelta(minutes=5, seconds=10)).strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"), 0.9, "Лог Б")
                else:
                    f1 = AtomicFact(f"T-{i}-1", "Камера А", "камера", f"Субъект_{i}", Predicate.PRESENT.value, "Точка Альфа", case_time.strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"), 0.95, "Лог А")
                    f2 = AtomicFact(f"T-{i}-2", "Свидетель", "свидетель", f"Субъект_{i}", Predicate.ABSENT.value, "Точка Альфа", (case_time + noise_delta).strftime("%Y-%m-%d %H:%M:%S"), (case_time + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"), 0.6, "Лог А", time_uncertainty_sec=60.0)
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
            "recall": round(recall * 100, 2), "f1_score": round(f1_score * 100, 2),
            "disclaimer": "Ground truth generated synthetically. Results demonstrate algorithmic consistency, not real-world forensic accuracy."
        }

    @staticmethod
    def evaluate_external_dataset(engine: ForensicCollisionEngine, dataset: List[Dict], locations: Dict[str, Location]) -> Dict:
        tp, fp, tn, fn = 0, 0, 0, 0
        type_metrics: Dict[str, Dict[str, int]] = {
            "bilocation": {"tp": 0, "fp": 0, "fn": 0},
            "kinematic": {"tp": 0, "fp": 0, "fn": 0},
            "contradiction": {"tp": 0, "fp": 0, "fn": 0}
        }

        for item in dataset:
            facts_data = item.get("facts", [])
            expected_collisions = item.get("expected_collisions", [])
            facts = [AtomicFact(**fd) for fd in facts_data]

            detected_raw = engine.analyze(facts, locations)
            detected_types = {r["type"] for r in detected_raw}

            is_expected = len(expected_collisions) > 0
            is_detected = len(detected_raw) > 0

            if is_expected and is_detected: tp += 1
            elif not is_expected and is_detected: fp += 1
            elif not is_expected and not is_detected: tn += 1
            elif is_expected and not is_detected: fn += 1

            for exp in expected_collisions:
                exp_type = exp.get("type_category", "bilocation")
                if exp_type in type_metrics:
                    matched = any(exp.get("type_keyword", "") in dt for dt in detected_types)
                    if matched: type_metrics[exp_type]["tp"] += 1
                    else: type_metrics[exp_type]["fn"] += 1

        eps = 1e-7
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            "total_cases": len(dataset), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round(accuracy * 100, 2), "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2), "f1_score": round(f1_score * 100, 2),
            "type_breakdown": type_metrics
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
                       0.35, "С 14:00 до 14:40 находился в читальном зале библиотеки.", "Формирование алиби", 0.85, 120.0),
            AtomicFact("F-02", "Камера CAM-305", "камера", "Арман С.", 
                       Predicate.PRESENT.value, "Кабинет 305", "2026-10-12 14:15", "2026-10-12 14:25", 
                       0.95, "Зафиксирован субъект схожей комплекции.", "Объективный видеоконтроль", 0.0, 10.0),
            AtomicFact("F-03", "Показания Дамира", "свидетель", "Арман С.", 
                       Predicate.PRESENT.value, "Центральный вход", "2026-10-12 14:26", "2026-10-12 14:28", 
                       0.60, "Видел Армана у главного входа.", "Информационный свидетель", 0.15, 60.0),
            AtomicFact("F-04", "Показания охранника", "свидетель", "Арман С.", 
                       Predicate.ABSENT.value, "Библиотека", "2026-10-12 14:10", "2026-10-12 14:35", 
                       0.75, "В помещении библиотеки посторонних не наблюдалось.", "Служебный контроль", 0.05, 30.0)
        ]
        return locs, facts

# -------------------------------------------------------------
# 2. ПОЛЬЗОВАТЕЛЬСКИЙ ИНТЕРФЕЙС STREAMLIT
# -------------------------------------------------------------

st.set_page_config(page_title="AI Forensic Workspace", page_icon="⚖️", layout="wide")

if "locations" not in st.session_state or "facts" not in st.session_state:
    def_locs, def_facts = DatabaseManager.get_default_dataset()
    st.session_state.locations = def_locs
    st.session_state.facts = def_facts

st.sidebar.title("⚙️ Экспертные параметры СППР")
walk_speed = st.sidebar.slider("Норматив скорости шага (км/ч)", 2.0, 8.0, 5.0, 0.5)
sprint_speed = st.sidebar.slider("Физиологический предел бега (км/ч)", 8.0, 30.0, 18.0, 1.0)
veh_speed = st.sidebar.slider("Предел скорости транспорта (км/ч)", 40.0, 150.0, 90.0, 5.0)
radius_same = st.sidebar.slider("Погрешность координат точки (м)", 0.5, 10.0, 2.0, 0.5)
weight_gap = st.sidebar.slider("Порог критического дисбаланса весов", 0.1, 0.9, 0.45, 0.05)

config = AnalysisConfig(
    max_walking_speed_kmh=walk_speed,
    max_sprint_speed_kmh=sprint_speed,
    max_vehicle_speed_kmh=veh_speed,
    same_location_radius_m=radius_same,
    critical_weight_gap=weight_gap
)

st.title("⚖️ AI Forensic Workspace")
st.caption("Система поддержки принятия решений (СППР) для аудита криминалистической доказательной базы")

tab_add, tab_registry, tab_map2d, tab_graph, tab_analysis, tab_benchmark, tab_export = st.tabs([
    "📥 Добавление материалов",
    "📋 Реестр доказательств",
    "🗺️ 2D-Карта (X / Y)",
    "🕸️ Топология связей",
    "🚨 Экспертиза & Аудит",
    "🔬 Научный бенчмарк",
    "📄 Экспорт заключения"
])

with tab_add:
    sub1, sub2, sub3 = st.tabs(["⚡ Поштучный ввод", "✍️ Ручной конструктор", "📁 Импорт файлов (.txt, .json)"])
    
    with sub1:
        st.markdown("**Быстрое внесение показания:**")
        single_input = st.text_input("Текст факта или цитаты:", "Свидетель Айбек сообщил: встретил Армана в Столовая около 14:32.")
        if st.button("🚀 Распознать и добавить"):
            if single_input.strip():
                new_f, updated_locs = SmartFreeTextParser.parse_documents(
                    single_input, default_date="2026-10-12",
                    current_locs=st.session_state.locations,
                    start_id=len(st.session_state.facts) + 1
                )
                existing_sigs = {f.get_signature() for f in st.session_state.facts}
                for f in new_f:
                    if f.get_signature() not in existing_sigs:
                        st.session_state.facts.append(f)
                st.session_state.locations = updated_locs
                st.success("Материал успешно добавлен в базу.")
                st.rerun()

    with sub2:
        st.markdown("**Параметрический ввод факта:**")
        with st.form("manual_entry"):
            c1, c2, c3 = st.columns(3)
            with c1:
                f_s = st.text_input("Субъект (ФИО)", "Арман С.")
                f_p = st.selectbox("Предикат", [p.value for p in Predicate])
                f_l = st.selectbox("Локация", list(st.session_state.locations.keys()))
            with c2:
                f_src = st.text_input("Источник", "Протокол опроса #5")
                f_type = st.selectbox("Тип источника", ["свидетель", "подозреваемый", "камера", "биллинг", "турникет"])
                f_w = st.slider("Вес достоверности", 0.1, 1.0, 0.65, 0.05)
            with c3:
                f_t1 = st.text_input("Начало", "2026-10-12 14:15")
                f_t2 = st.text_input("Окончание", "2026-10-12 14:30")
                f_unc = st.number_input("Погрешность времени (± сек)", value=60.0, step=10.0)
                f_mot = st.text_input("Мотивационный профиль", "Нейтральный свидетель")
                f_conf = st.slider("Конфликт интересов", 0.0, 1.0, 0.2, 0.05)
            f_q = st.text_area("Цитата источника", "Находился в указанном месте.")
            if st.form_submit_button("💾 Сохранить факт"):
                new_atom = AtomicFact(f"F-{len(st.session_state.facts)+1:02d}", f_src, f_type, f_s, f_p, f_l, f_t1, f_t2, f_w, f_q, f_mot, f_conf, f_unc)
                st.session_state.facts.append(new_atom)
                st.success("Факт сохранен.")
                st.rerun()

    with sub3:
        st.markdown("**Пакетная загрузка документов:**")
        up_files = st.file_uploader("Файлы протоколов:", type=["txt", "json"], accept_multiple_files=True)
        mode = st.radio("Режим загрузки:", ["Дописать к текущим", "Перезаписать базу"], horizontal=True)
        if st.button("⚡ Обработать файлы"):
            if up_files:
                combined = ""
                for uf in up_files:
                    if uf.size <= 5 * 1024 * 1024:
                        try: combined += f"\n{uf.getvalue().decode('utf-8')}"
                        except UnicodeDecodeError: combined += f"\n{uf.getvalue().decode('cp1251', errors='replace')}"
                if combined.strip():
                    start_idx = 1 if mode == "Перезаписать базу" else len(st.session_state.facts) + 1
                    b_facts, b_locs = SmartFreeTextParser.parse_documents(combined, default_date="2026-10-12", current_locs=st.session_state.locations, start_id=start_idx)
                    if mode == "Перезаписать базу": st.session_state.facts = b_facts
                    else: st.session_state.facts.extend(b_facts)
                    st.session_state.locations = b_locs
                    st.success(f"Обработано {len(b_facts)} записей.")
                    st.rerun()

with tab_registry:
    st.subheader("Реестр формализованных доказательств")
    if st.session_state.facts:
        f_df = pd.DataFrame([{
            "ID": f.fact_id, "Субъект": f.subject, "Предикат": f.predicate,
            "Локация": f.location_name, "Интервал": f"{f.t_start[-5:] if len(f.t_start)>=5 else ''} — {f.t_end[-5:] if len(f.t_end)>=5 else ''}",
            "Погрешность": f"±{int(f.time_uncertainty_sec)} с", "Источник": f.source_id, "Вес": f.weight, "Конфликт": f.interest_conflict
        } for f in st.session_state.facts])
        st.dataframe(f_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Очистить базу"):
            st.session_state.facts = []
            st.rerun()
    else:
        st.info("Реестр пуст.")

with tab_map2d:
    st.subheader("🗺️ Пространственные координаты локаций")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        with st.form("loc_form"):
            n_name = st.text_input("Локация", "Серверная")
            n_x = st.number_input("Координата X (м)", value=60.0)
            n_y = st.number_input("Координата Y (м)", value=90.0)
            n_desc = st.text_input("Описание", "Служебный сектор")
            if st.form_submit_button("📍 Добавить"):
                st.session_state.locations[n_name] = Location(n_name, n_x, n_y, n_desc)
                st.success(f"Точка '{n_name}' добавлена.")
                st.rerun()

    with col_m2:
        loc_df = pd.DataFrame([{"Локация": l.name, "X": l.x, "Y": l.y, "Описание": l.description} for l in st.session_state.locations.values() if l.has_coordinates])
        if not loc_df.empty:
            fig = px.scatter(loc_df, x="X", y="Y", text="Локация", hover_data=["Описание"], title="План объекта (сетка в метрах)", template="plotly_dark")
            fig.update_traces(marker=dict(size=14, color="#00E676", line=dict(width=2, color="white")), textposition="top center", textfont=dict(size=13, color="white"))
            fig.update_layout(xaxis_title="Ось X (метры)", yaxis_title="Ось Y (метры)", height=430)
            st.plotly_chart(fig, use_container_width=True)

with tab_graph:
    st.subheader("🕸️ Граф темпоральных связей")
    if st.session_state.facts:
        net = Network(height="460px", width="100%", bgcolor="#0E1117", font_color="white")
        net.force_atlas_2based()
        added = set()
        for f in st.session_state.facts:
            if f.subject not in added: net.add_node(f.subject, label=f.subject, color="#1E88E5", size=25); added.add(f.subject)
            if f.location_name not in added: net.add_node(f.location_name, label=f.location_name, color="#43A047", size=22, shape="box"); added.add(f.location_name)
            col = "#E53935" if f.predicate == Predicate.ABSENT.value else "#90CAF9"
            t_lbl = f"[{f.t_start[-5:]}-{f.t_end[-5:]}]" if len(f.t_start)>=5 and len(f.t_end)>=5 else "[Без времени]"
            net.add_edge(f.subject, f.location_name, label=t_lbl, color=col)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
            net.save_graph(tmp.name)
            tmp_path = tmp.name
        with open(tmp_path, "r", encoding="utf-8") as f_html:
            components.html(f_html.read(), height=480)
        os.remove(tmp_path)

with tab_analysis:
    st.subheader("🚨 Экспертный протокол и цепочка аудита коллизий")
    engine = ForensicCollisionEngine(config=config)
    results = engine.analyze(st.session_state.facts, st.session_state.locations)
    
    st.metric("Выявлено экспертных находок", len(results))
    for item in results:
        status_color = "🔴" if item.get("status") == ConflictStatus.CONFIRMED.value else ("🟠" if item.get("status") == ConflictStatus.POSSIBLE.value else "🔵")
        with st.expander(f"{status_color} [{item['id']}] {item['type']} — {item['subject']} | {item.get('status', '')}", expanded=True):
            c_top1, c_top2 = st.columns([2, 1])
            with c_top1:
                st.write(f"**Суть:** {item['details']}")
                st.write(f"**Темпоральное отношение Аллена:** `{item['allen_relation']}`")
            with c_top2:
                st.metric("Достоверность вывода", f"{int(item['confidence']*100)}%", item['confidence_label'])
                with st.popover("Факторы скоринга"):
                    for k, v in item.get("confidence_factors", {}).items(): st.write(f"• **{k}**: {v}")

            st.markdown("---")
            col_d1, col_d2 = st.columns(2)
            f1, f2 = item['facts'][0], item['facts'][1]
            with col_d1:
                st.markdown(f"**Исходные данные: Утверждение А ({f1.fact_id})**")
                st.write(f"• Источник: `{f1.source_id}` ({f1.source_type}, вес {f1.weight})")
                st.write(f"• Действие: *{f1.predicate}* в **{f1.location_name}**")
                st.write(f"• Время: `{f1.t_start}` — `{f1.t_end}` (±{int(f1.time_uncertainty_sec)} с)")
                st.caption(f"«{f1.source_excerpt}»")
            with col_d2:
                st.markdown(f"**Исходные данные: Утверждение Б ({f2.fact_id})**")
                st.write(f"• Источник: `{f2.source_id}` ({f2.source_type}, вес {f2.weight})")
                st.write(f"• Действие: *{f2.predicate}* в **{f2.location_name}**")
                st.write(f"• Время: `{f2.t_start}` — `{f2.t_end}` (±{int(f2.time_uncertainty_sec)} с)")
                st.caption(f"«{f2.source_excerpt}»")

            st.markdown("---")
            c_calc, c_chain = st.columns([1, 1])
            with c_calc:
                st.markdown("**📊 Математический расчет:**")
                for k, v in item.get("calculation", {}).items(): st.write(f"• **{k}**: `{v}`")
            with c_chain:
                st.markdown("**🔗 Цепочка аудита (Audit Trail):**")
                for step in item.get("evidence_chain", []): st.markdown(f"↳ *{step}*")

            st.info(f"📋 **Рекомендация эксперту:** {item['expert_note']}")
            st.warning(f"⚠️ **Ограничения вывода (Forensic Disclaimer):** {item.get('limitations', 'Вывод зависит от исходных данных.')}")

with tab_benchmark:
    st.subheader("🔬 Экспериментальная валидация СППР")
    b_synth, b_ext = st.tabs(["🧪 Синтетическая валидация", "📁 Внешний размеченный датасет"])

    with b_synth:
        col_bn1, col_bn2 = st.columns([1, 2])
        with col_bn1:
            samples = st.selectbox("Размер выборки", [100, 250, 500, 1000], index=1)
            rate = st.slider("Доля аномалий в выборке", 0.1, 0.9, 0.5, 0.1)
            noise = st.checkbox("Учитывать шум свидетельских показаний", value=True)
            run_btn = st.button("🚀 Запустить синтетический тест")
        if run_btn:
            val_res = ScientificValidator.run_synthetic_benchmark(engine, test_samples=samples, anomaly_rate=rate, add_noise=noise)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Точность (Accuracy)", f"{val_res['accuracy']}%")
            m2.metric("Precision", f"{val_res['precision']}%")
            m3.metric("Recall", f"{val_res['recall']}%")
            m4.metric("F1-Score", f"{val_res['f1_score']}%")
            cm_data = {"Факт: Есть коллизия": [f"TP: {val_res['tp']}", f"FN: {val_res['fn']}"], "Факт: Нет коллизии": [f"FP: {val_res['fp']}", f"TN: {val_res['tn']}"]}
            st.table(pd.DataFrame(cm_data, index=["Система нашла коллизию", "Система сочла алиби чистым"]))
            st.info(f"ℹ️ **Дисклеймер:** {val_res['disclaimer']}")

    with b_ext:
        sample_ext = [{"case_id": "CASE-GT-01", "facts": [{"fact_id": "F1", "source_id": "Камера А", "source_type": "камера", "subject": "Арман С.", "predicate": "находился", "location_name": "Кабинет 305", "t_start": "2026-10-12 14:15", "t_end": "2026-10-12 14:25", "weight": 0.95, "source_excerpt": "лог", "time_uncertainty_sec": 10.0}, {"fact_id": "F2", "source_id": "Свидетель", "source_type": "свидетель", "subject": "Арман С.", "predicate": "находился", "location_name": "Библиотека", "t_start": "2026-10-12 14:15", "t_end": "2026-10-12 14:30", "weight": 0.60, "source_excerpt": "лог", "time_uncertainty_sec": 60.0}], "expected_collisions": [{"type_category": "bilocation", "type_keyword": "БИЛОКАЦИЯ"}]}]
        st.download_button("📥 Скачать пример эталонного JSON", data=json.dumps(sample_ext, ensure_ascii=False, indent=2), file_name="sample_ground_truth.json", mime="application/json")
        ext_file = st.file_uploader("Загрузить размеченный файл Ground Truth (.json):", type=["json"])
        if ext_file and st.button("📊 Рассчитать метрики по внешнему датасету"):
            try:
                ds = json.loads(ext_file.getvalue().decode("utf-8"))
                ext_metrics = ScientificValidator.evaluate_external_dataset(engine, ds, st.session_state.locations)
                c_em1, c_em2, c_em3, c_em4 = st.columns(4)
                c_em1.metric("Тестовых дел", ext_metrics["total_cases"])
                c_em2.metric("Precision", f"{ext_metrics['precision']}%")
                c_em3.metric("Recall", f"{ext_metrics['recall']}%")
                c_em4.metric("F1-Score", f"{ext_metrics['f1_score']}%")
                st.dataframe(pd.DataFrame(ext_metrics["type_breakdown"]).T, use_container_width=True)
            except Exception as e: st.error(f"Ошибка парсинга JSON: {e}")

with tab_export:
    st.subheader("📄 Генерация экспертного криминалистического отчета")
    report_data = {
        "case_metadata": {
            "case_id": "EXP-RNKP-2026-08",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "engine_version": "AI-Forensic-Core v2.4 (Standalone)",
            "parameters": {"max_walking_speed_kmh": config.max_walking_speed_kmh, "max_sprint_speed_kmh": config.max_sprint_speed_kmh, "max_vehicle_speed_kmh": config.max_vehicle_speed_kmh}
        },
        "facts_summary": [{"id": f.fact_id, "subject": f.subject, "predicate": f.predicate, "location": f.location_name, "interval": f"{f.t_start} — {f.t_end}", "uncertainty_sec": f.time_uncertainty_sec, "weight": f.weight} for f in st.session_state.facts],
        "findings": [{"id": r["id"], "type": r["type"], "status": r.get("status"), "confidence": r["confidence"], "confidence_label": r["confidence_label"], "confidence_factors": r.get("confidence_factors"), "allen_relation": r["allen_relation"], "evidence_chain": r.get("evidence_chain"), "calculation": r.get("calculation"), "limitations": r.get("limitations")} for r in results]
    }
    c_exp1, c_exp2 = st.columns(2)
    with c_exp1:
        st.download_button("📥 Скачать отчет JSON", data=json.dumps(report_data, ensure_ascii=False, indent=2), file_name=f"Forensic_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
    with c_exp2:
        html_report = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Экспертное заключение</title><style>body {{ font-family: sans-serif; margin: 40px; color: #111; }} h1 {{ border-bottom: 2px solid #222; padding-bottom: 8px; }} table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }} th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }} th {{ background: #f0f4f8; }} .finding {{ border: 1px solid #ddd; background: #fafafa; padding: 15px; margin-bottom: 15px; border-radius: 4px; }} .confirmed {{ border-left: 6px solid #d32f2f; }} .possible {{ border-left: 6px solid #f57c00; }} .chain {{ background: #fff; border: 1px solid #eee; padding: 10px; font-family: monospace; font-size: 12px; }}</style></head><body><h1>ЭКСПЕРТНЫЙ АУДИТ ДОКАЗАТЕЛЬСТВ</h1><p><strong>Дело:</strong> {report_data['case_metadata']['case_id']} | <strong>Дата:</strong> {report_data['case_metadata']['analysis_timestamp']}</p><h2>1. Реестр доказательств</h2><table><tr><th>ID</th><th>Субъект</th><th>Предикат</th><th>Локация</th><th>Интервал</th><th>Погрешность</th><th>Вес</th></tr>{''.join([f"<tr><td>{f['id']}</td><td>{f['subject']}</td><td>{f['predicate']}</td><td>{f['location']}</td><td>{f['interval']}</td><td>&plusmn;{int(f['uncertainty_sec'])} с</td><td>{f['weight']}</td></tr>" for f in report_data['facts_summary']])}</table><h2>2. Выявленные коллизии</h2>{''.join([f"""<div class="finding {'confirmed' if f.get('status')=='ПОДТВЕРЖДЁННАЯ' else 'possible'}"><h3>[{f['id']}] {f['type']} — {f.get('status', '')}</h3><p><strong>Достоверность:</strong> {int(f['confidence']*100)}% ({f['confidence_label']}) | <strong>Аллен:</strong> {f['allen_relation']}</p><div class="chain"><strong>Audit Trail:</strong><br>{'<br>↳ '.join(f.get('evidence_chain', []))}</div><p style="margin-top:10px; font-size:12px; color:#555;"><strong>Ограничения:</strong> {f.get('limitations')}</p></div>""" for f in report_data['findings']])}</body></html>"""
        st.download_button("📥 Скачать официальный отчет HTML", data=html_report, file_name=f"Forensic_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html", mime="text/html")
