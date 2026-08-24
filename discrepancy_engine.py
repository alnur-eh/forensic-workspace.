"""
AI Forensic Workspace — Algorithmic & Decision Support Core
Математическое ядро интервальной логики Аллена, кинематического моделирования,
оценки неопределенности времени, confidence-скоринга и генерации аудиторского следа.
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
from typing import Dict, List, Optional, Tuple, Any

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
    if start_a < start_b < end_a < end_b: return "OVERLAPS"
    if start_b < start_a < end_b < end_a: return "OVERLAPPED_BY"
    return "UNKNOWN"

class ConfidenceCalculator:
    """Прозрачная модель скоринга достоверности вывода на основе детерминированных факторов"""
    @staticmethod
    def calculate(
        f1: AtomicFact,
        f2: AtomicFact,
        status: ConflictStatus,
        collision_type: CollisionType,
        has_coords: bool,
        kinematic_ratio: float = 1.0
    ) -> Tuple[float, str, Dict[str, float]]:
        # 1. Надежность источников (0.1 .. 1.0)
        source_rel = max(0.1, min(1.0, (f1.weight + f2.weight) / 2.0))
        # 2. Штраф за конфликт интересов
        bias_penalty = max(0.0, min(0.3, (f1.interest_conflict + f2.interest_conflict) * 0.15))
        source_factor = max(0.1, source_rel - bias_penalty)

        # 3. Временная точность (штраф за неопределенность)
        max_unc = max(f1.time_uncertainty_sec, f2.time_uncertainty_sec)
        temporal_factor = max(0.2, 1.0 - min(0.8, max_unc / 1800.0))

        # 4. Пространственная точность
        spatial_factor = 1.0 if has_coords else 0.0

        # 5. Вес статуса (Подтвержденная vs Возможная)
        status_factor = 1.0 if status == ConflictStatus.CONFIRMED else (0.65 if status == ConflictStatus.POSSIBLE else 0.3)

        # 6. Запас кинематического превышения
        kin_factor = min(1.0, kinematic_ratio / 2.0) if kinematic_ratio > 1.0 else 1.0

        # Взвешенная сумма
        if collision_type in {CollisionType.KINEMATIC_CRITICAL, CollisionType.KINEMATIC_VEHICLE_REQUIRED, CollisionType.SPATIAL_TEMPORAL}:
            raw_score = (
                source_factor * 0.35 +
                temporal_factor * 0.25 +
                spatial_factor * 0.20 +
                status_factor * 0.20
            ) * kin_factor
        elif collision_type == CollisionType.INSUFFICIENT_SPATIAL_DATA:
            raw_score = 0.30
        else:
            raw_score = (source_factor * 0.45 + temporal_factor * 0.30 + status_factor * 0.25)

        confidence = round(max(0.1, min(0.99, raw_score)), 2)

        if confidence >= 0.80:
            label = "ВЫСОКАЯ"
        elif confidence >= 0.50:
            label = "СРЕДНЯЯ"
        else:
            label = "НИЗКАЯ"

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
        if not ti1 or not ti2:
            return results

        raw_s1, raw_e1 = ti1.get_raw_interval()
        raw_s2, raw_e2 = ti2.get_raw_interval()
        eff_s1, eff_e1 = ti1.get_effective_interval()
        eff_s2, eff_e2 = ti2.get_effective_interval()

        raw_overlap = (raw_s1 < raw_e2) and (raw_s2 < raw_e1)
        effective_overlap = (eff_s1 < eff_e2) and (eff_s2 < eff_e1)

        allen_rel = get_allen_relation(raw_s1, raw_e1, raw_s2, raw_e2)
        loc1 = locations.get(f1.location_name)
        loc2 = locations.get(f2.location_name)

        # 1. Прямое логическое противоречие
        if effective_overlap and loc1 and loc2 and loc1.name == loc2.name:
            if {f1.predicate, f2.predicate} == {Predicate.PRESENT.value, Predicate.ABSENT.value}:
                status = ConflictStatus.CONFIRMED if raw_overlap else ConflictStatus.POSSIBLE
                conf, conf_label, conf_factors = ConfidenceCalculator.calculate(
                    f1, f2, status, CollisionType.DIRECT_CONTRADICTION, has_coords=True
                )
                results.append({
                    "id": f"COL-LOGIC-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.DIRECT_CONTRADICTION.value,
                    "status": status.value,
                    "subject": f1.subject,
                    "severity": "ВЫСОКАЯ",
                    "confidence": conf,
                    "confidence_label": conf_label,
                    "confidence_factors": conf_factors,
                    "details": f"Взаимоисключающие утверждения о присутствии и отсутствии в локации '{loc1.name}'.",
                    "allen_relation": allen_rel,
                    "evidence_chain": [
                        f"Факт A ({f1.fact_id}): Субъект '{f1.subject}' {f1.predicate} в '{loc1.name}' [{f1.t_start} — {f1.t_end}].",
                        f"Факт B ({f2.fact_id}): Субъект '{f2.subject}' {f2.predicate} в '{loc2.name}' [{f2.t_start} — {f2.t_end}].",
                        f"Темпоральный анализ: Отношение Аллена '{allen_rel}'. Эффективное перекрытие = {effective_overlap}.",
                        f"Логический конфликт: Одновременное утверждение противоположных предикатов в одном месте.",
                        f"Статус: {status.value} коллизия (Confidence {int(conf*100)}%)."
                    ],
                    "calculation": {
                        "location_match": True,
                        "raw_overlap": raw_overlap,
                        "effective_overlap": effective_overlap,
                        "weight_delta": round(abs(f1.weight - f2.weight), 2)
                    },
                    "facts": [f1, f2],
                    "expert_note": f"Требуется перекрестный допрос источников. Дельта конфликта интересов: {abs(f1.interest_conflict - f2.interest_conflict):.2f}.",
                    "limitations": "Вывод базируется на предположении о неизменности смыслового значения показаний."
                })

        # 2. Пространственно-временная несогласованность (Билокация)
        if effective_overlap and loc1 and loc2 and loc1.name != loc2.name:
            if not loc1.has_coordinates or not loc2.has_coordinates:
                # Статус: Недостаточно пространственных данных
                conf, conf_label, conf_factors = ConfidenceCalculator.calculate(
                    f1, f2, ConflictStatus.INSUFFICIENT_DATA, CollisionType.INSUFFICIENT_SPATIAL_DATA, has_coords=False
                )
                results.append({
                    "id": f"COL-NODATA-{f1.fact_id}-{f2.fact_id}",
                    "type": CollisionType.INSUFFICIENT_SPATIAL_DATA.value,
                    "status": ConflictStatus.INSUFFICIENT_DATA.value,
                    "subject": f1.subject,
                    "severity": "ИНФОРМАЦИОННАЯ",
                    "confidence": conf,
                    "confidence_label": conf_label,
                    "confidence_factors": conf_factors,
                    "details": f"События перекрываются во времени между '{loc1.name}' и '{loc2.name}', но координаты одной или обеих локаций не откалиброваны.",
                    "allen_relation": allen_rel,
                    "evidence_chain": [
                        f"Факт A ({f1.fact_id}): Локация '{loc1.name}' (координаты: {loc1.x, loc1.y}).",
                        f"Факт B ({f2.fact_id}): Локация '{loc2.name}' (координаты: {loc2.x, loc2.y}).",
                        "Пространственная проверка невозможна: отсутствует метрическая калибровка плана.",
                        "Статус: НЕДОСТАТОЧНО ДАННЫХ ДЛЯ ВЕРИФИКАЦИИ."
                    ],
                    "calculation": {"coordinates_calibrated": False},
                    "facts": [f1, f2],
                    "expert_note": "Необходимо задать координаты локаций на 2D-карте для выполнения расчета билокации.",
                    "limitations": "Проверка алиби не завершена из-за отсутствия координатной привязки."
                })
            else:
                dist = calculate_distance(loc1, loc2)
                if dist is not None and dist > self.config.same_location_radius_m and f1.predicate == Predicate.PRESENT.value and f2.predicate == Predicate.PRESENT.value:
                    status = ConflictStatus.CONFIRMED if raw_overlap else ConflictStatus.POSSIBLE
                    conf, conf_label, conf_factors = ConfidenceCalculator.calculate(
                        f1, f2, status, CollisionType.SPATIAL_TEMPORAL, has_coords=True
                    )
                    results.append({
                        "id": f"COL-ST-{f1.fact_id}-{f2.fact_id}",
                        "type": CollisionType.SPATIAL_TEMPORAL.value,
                        "status": status.value,
                        "subject": f1.subject,
                        "severity": "КРИТИЧЕСКАЯ",
                        "confidence": conf,
                        "confidence_label": conf_label,
                        "confidence_factors": conf_factors,
                        "details": f"Одновременное присутствие в разных точках ({dist:.1f} м > порога {self.config.same_location_radius_m:.1f} м).",
                        "allen_relation": allen_rel,
                        "evidence_chain": [
                            f"Факт A ({f1.fact_id}): Локация '{loc1.name}' [{f1.t_start} — {f1.t_end}].",
                            f"Факт B ({f2.fact_id}): Локация '{loc2.name}' [{f2.t_start} — {f2.t_end}].",
                            f"Евклидово расстояние между точками = {dist:.1f} м (порог одной точки {self.config.same_location_radius_m:.1f} м).",
                            f"Отношение Аллена: {allen_rel}. Перекрытие интервалов подтверждено.",
                            f"Статус: {status.value} билокация (Confidence {int(conf*100)}%)."
                        ],
                        "calculation": {
                            "distance_m": round(dist, 1),
                            "raw_overlap": raw_overlap,
                            "effective_overlap": effective_overlap,
                            "threshold_radius_m": self.config.same_location_radius_m
                        },
                        "facts": [f1, f2],
                        "expert_note": "Несогласованность временных меток объективного контроля или свидетельских показаний.",
                        "limitations": "Вывод зависит от точности синхронизации системных часов регистраторов."
                    })

                    # Проверка квалифицирующего признака надежности источников
                    diff = abs(f1.weight - f2.weight)
                    if diff >= self.config.critical_weight_gap:
                        low_src = f1 if f1.weight < f2.weight else f2
                        high_src = f2 if f1.weight < f2.weight else f1
                        results.append({
                            "id": f"COL-BIAS-{f1.fact_id}-{f2.fact_id}",
                            "type": CollisionType.SOURCE_RELIABILITY.value,
                            "status": status.value,
                            "subject": f1.subject,
                            "severity": "СРЕДНЯЯ",
                            "confidence": conf,
                            "confidence_label": conf_label,
                            "confidence_factors": conf_factors,
                            "details": f"При подтвержденном конфликте источник низкой надежности '{low_src.source_id}' ({low_src.weight:.2f}) противоречит объективному источнику '{high_src.source_id}' ({high_src.weight:.2f}).",
                            "allen_relation": f"Дельта весов: {diff:.2f}",
                            "evidence_chain": [
                                f"Обнаружена пространственно-временная коллизия между фактами {f1.fact_id} и {f2.fact_id}.",
                                f"Анализ достоверности: Источник '{high_src.source_id}' (вес {high_src.weight:.2f}) против '{low_src.source_id}' (вес {low_src.weight:.2f}).",
                                f"Дельта весов = {diff:.2f} превышает порог {self.config.critical_weight_gap:.2f}.",
                                f"Мотивационный профиль низконадежного источника: '{low_src.motive_flag}'."
                            ],
                            "calculation": {
                                "weight_delta": round(diff, 2),
                                "critical_gap_threshold": self.config.critical_weight_gap
                            },
                            "facts": [f1, f2],
                            "expert_note": f"Мотивационный профиль уязвимого источника: {low_src.motive_flag}.",
                            "limitations": "Веса источников назначены на основе нормативных криминалистических шкал."
                        })

        # 3. Кинематический последовательный анализ
        if loc1 and loc2 and loc1.name != loc2.name and f1.predicate == Predicate.PRESENT.value and f2.predicate == Predicate.PRESENT.value:
            if eff_e1 <= eff_s2:
                earlier, later, e_end, l_start, raw_e_end, raw_l_start = f1, f2, eff_e1, eff_s2, raw_e1, raw_s2
            elif eff_e2 <= eff_s1:
                earlier, later, e_end, l_start, raw_e_end, raw_l_start = f2, f1, eff_e2, eff_s1, raw_e2, raw_s1
            else:
                earlier = None

            if earlier:
                loc_e = locations.get(earlier.location_name)
                loc_l = locations.get(later.location_name)
                if loc_e and loc_l:
                    if not loc_e.has_coordinates or not loc_l.has_coordinates:
                        conf, conf_label, conf_factors = ConfidenceCalculator.calculate(
                            f1, f2, ConflictStatus.INSUFFICIENT_DATA, CollisionType.INSUFFICIENT_SPATIAL_DATA, has_coords=False
                        )
                        results.append({
                            "id": f"COL-NODATA-KIN-{f1.fact_id}-{f2.fact_id}",
                            "type": CollisionType.INSUFFICIENT_SPATIAL_DATA.value,
                            "status": ConflictStatus.INSUFFICIENT_DATA.value,
                            "subject": f1.subject,
                            "severity": "ИНФОРМАЦИОННАЯ",
                            "confidence": conf,
                            "confidence_label": conf_label,
                            "confidence_factors": conf_factors,
                            "details": f"Последовательное перемещение между '{loc_e.name}' и '{loc_l.name}' не может быть проверено на скорость (нет координат).",
                            "allen_relation": allen_rel,
                            "evidence_chain": [
                                f"Факт A ({earlier.fact_id}): Окончание в {earlier.t_end}.",
                                f"Факт B ({later.fact_id}): Начало в {later.t_start}.",
                                "Кинематический расчет скорости заблокирован: отсутствуют координаты локаций."
                            ],
                            "calculation": {"coordinates_calibrated": False},
                            "facts": [f1, f2],
                            "expert_note": "Укажите координаты точек для проверки физической возможности перемещения.",
                            "limitations": "Оценка скорости не произведена."
                        })
                    else:
                        dist = calculate_distance(loc_e, loc_l)
                        gap_sec = (l_start - e_end).total_seconds()
                        raw_gap_sec = (raw_l_start - raw_e_end).total_seconds()

                        if dist is not None and dist > self.config.same_location_radius_m:
                            if gap_sec == 0:
                                status = ConflictStatus.CONFIRMED if raw_gap_sec == 0 else ConflictStatus.POSSIBLE
                                conf, conf_label, conf_factors = ConfidenceCalculator.calculate(
                                    f1, f2, status, CollisionType.KINEMATIC_CRITICAL, has_coords=True, kinematic_ratio=10.0
                                )
                                results.append({
                                    "id": f"COL-KIN-ZERO-{f1.fact_id}-{f2.fact_id}",
                                    "type": CollisionType.KINEMATIC_CRITICAL.value,
                                    "status": status.value,
                                    "subject": f1.subject,
                                    "severity": "КРИТИЧЕСКАЯ",
                                    "confidence": conf,
                                    "confidence_label": conf_label,
                                    "confidence_factors": conf_factors,
                                    "details": f"Мгновенное перемещение между '{loc_e.name}' и '{loc_l.name}' (дистанция {dist:.1f} м за 0.0 с).",
                                    "allen_relation": "MEETS (Временной зазор: 0.0 с)",
                                    "evidence_chain": [
                                        f"Исходная точка ({earlier.fact_id}): '{loc_e.name}' [{earlier.t_start} — {earlier.t_end}].",
                                        f"Конечная точка ({later.fact_id}): '{loc_l.name}' [{later.t_start} — {later.t_end}].",
                                        f"Расстояние = {dist:.1f} м. Временной интервал между событиями = 0.0 с.",
                                        "Требуемая скорость = Бесконечность (мгновенная телепортация).",
                                        f"Статус: {status.value} кинематическая аномалия."
                                    ],
                                    "calculation": {
                                        "distance_m": round(dist, 1),
                                        "time_gap_sec": 0.0,
                                        "required_speed_kmh": float("inf"),
                                        "threshold_kmh": self.config.max_vehicle_speed_kmh
                                    },
                                    "facts": [f1, f2],
                                    "expert_note": "Физически невозможное перемещение между пространственно разделенными объектами.",
                                    "limitations": "Вывод основан на строгой синхронности временных меток."
                                })
                            elif gap_sec > 0:
                                speed_kmh = (dist / gap_sec) * 3.6
                                raw_speed_kmh = (dist / raw_gap_sec) * 3.6 if raw_gap_sec > 0 else float("inf")

                                if speed_kmh > self.config.max_vehicle_speed_kmh:
                                    status = ConflictStatus.CONFIRMED if raw_speed_kmh > self.config.max_vehicle_speed_kmh else ConflictStatus.POSSIBLE
                                    conf, conf_label, conf_factors = ConfidenceCalculator.calculate(
                                        f1, f2, status, CollisionType.KINEMATIC_CRITICAL, has_coords=True,
                                        kinematic_ratio=speed_kmh / self.config.max_vehicle_speed_kmh
                                    )
                                    results.append({
                                        "id": f"COL-KIN-CRIT-{f1.fact_id}-{f2.fact_id}",
                                        "type": CollisionType.KINEMATIC_CRITICAL.value,
                                        "status": status.value,
                                        "subject": f1.subject,
                                        "severity": "КРИТИЧЕСКАЯ",
                                        "confidence": conf,
                                        "confidence_label": conf_label,
                                        "confidence_factors": conf_factors,
                                        "details": f"Расчетная скорость {speed_kmh:.1f} км/ч превышает предельный транспортный порог ({self.config.max_vehicle_speed_kmh:.1f} км/ч). Дистанция: {dist:.1f} м за {gap_sec:.1f} с.",
                                        "allen_relation": f"Временной зазор: {gap_sec:.1f} с",
                                        "evidence_chain": [
                                            f"Точка отправления: '{loc_e.name}' в {earlier.t_end}.",
                                            f"Точка прибытия: '{loc_l.name}' в {later.t_start}.",
                                            f"Минимальная дистанция по прямой = {dist:.1f} м.",
                                            f"Эффективный временной зазор = {gap_sec:.1f} с.",
                                            f"Расчетная минимальная скорость = {speed_kmh:.1f} км/ч (Предел транспорта: {self.config.max_vehicle_speed_kmh:.1f} км/ч).",
                                            f"Статус: {status.value} критическая аномалия (Confidence {int(conf*100)}%)."
                                        ],
                                        "calculation": {
                                            "distance_m": round(dist, 1),
                                            "time_gap_sec": round(gap_sec, 1),
                                            "required_speed_kmh": round(speed_kmh, 1),
                                            "threshold_kmh": self.config.max_vehicle_speed_kmh
                                        },
                                        "facts": [f1, f2],
                                        "expert_note": "Аномальная скорость перемещения. Требуется проверка аппаратных логов времени.",
                                        "limitations": "Расчет выполнен по евклидовой метрике (минимально возможное расстояние)."
                                    })
                                elif speed_kmh > self.config.max_sprint_speed_kmh:
                                    status = ConflictStatus.CONFIRMED if raw_speed_kmh > self.config.max_sprint_speed_kmh else ConflictStatus.POSSIBLE
                                    conf, conf_label, conf_factors = ConfidenceCalculator.calculate(
                                        f1, f2, status, CollisionType.KINEMATIC_VEHICLE_REQUIRED, has_coords=True,
                                        kinematic_ratio=speed_kmh / self.config.max_sprint_speed_kmh
                                    )
                                    results.append({
                                        "id": f"COL-KIN-VEH-{f1.fact_id}-{f2.fact_id}",
                                        "type": CollisionType.KINEMATIC_VEHICLE_REQUIRED.value,
                                        "status": status.value,
                                        "subject": f1.subject,
                                        "severity": "ВЫСОКАЯ",
                                        "confidence": conf,
                                        "confidence_label": conf_label,
                                        "confidence_factors": conf_factors,
                                        "details": f"Расчетная скорость {speed_kmh:.1f} км/ч превышает порог бега ({self.config.max_sprint_speed_kmh:.1f} км/ч), но допустима для автотранспорта (дистанция {dist:.1f} м за {gap_sec:.1f} с).",
                                        "allen_relation": f"Временной зазор: {gap_sec:.1f} с",
                                        "evidence_chain": [
                                            f"Точка A: '{loc_e.name}' ({earlier.t_end}) -> Точка B: '{loc_l.name}' ({later.t_start}).",
                                            f"Дистанция = {dist:.1f} м, Зазор = {gap_sec:.1f} с.",
                                            f"Скорость = {speed_kmh:.1f} км/ч (Порог пешком: {self.config.max_sprint_speed_kmh:.1f} км/ч, Транспорт: {self.config.max_vehicle_speed_kmh:.1f} км/ч).",
                                            "Перемещение пешком исключено. Необходим автотранспорт.",
                                            f"Статус: {status.value} (Confidence {int(conf*100)}%)."
                                        ],
                                        "calculation": {
                                            "distance_m": round(dist, 1),
                                            "time_gap_sec": round(gap_sec, 1),
                                            "required_speed_kmh": round(speed_kmh, 1),
                                            "threshold_kmh": self.config.max_sprint_speed_kmh
                                        },
                                        "facts": [f1, f2],
                                        "expert_note": "Пешее перемещение исключено. Требуется подтверждение использования автотранспорта.",
                                        "limitations": "Расчет не учитывает задержки на светофорах, турникетах и пропускных пунктах."
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
        """Оценка точности на внешнем размеченном датасете эксперта"""
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

            if is_expected and is_detected:
                tp += 1
            elif not is_expected and is_detected:
                fp += 1
            elif not is_expected and not is_detected:
                tn += 1
            elif is_expected and not is_detected:
                fn += 1

            for exp in expected_collisions:
                exp_type = exp.get("type_category", "bilocation")
                if exp_type in type_metrics:
                    matched = any(exp.get("type_keyword", "") in dt for dt in detected_types)
                    if matched:
                        type_metrics[exp_type]["tp"] += 1
                    else:
                        type_metrics[exp_type]["fn"] += 1

        eps = 1e-7
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            "total_cases": len(dataset),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1_score * 100, 2),
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

    @staticmethod
    def load_data() -> Tuple[Dict[str, Location], List[AtomicFact]]:
        return DatabaseManager.get_default_dataset()
