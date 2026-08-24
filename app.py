"""
AI Forensic Workspace — Frontend Interface
Судебно-экспертный интерфейс аудита доказательств, 2D-картографии,
детального аудиторского следа, внешнего бенчмаркинга и экспорта заключений.
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import tempfile
import os
import json
from datetime import datetime
from pyvis.network import Network
import plotly.express as px

from discrepancy_engine import (
    Location, AtomicFact, Predicate, AnalysisConfig,
    ForensicCollisionEngine, SmartFreeTextParser, ScientificValidator,
    DatabaseManager, ConflictStatus, CollisionType
)

st.set_page_config(
    page_title="AI Forensic Workspace", 
    page_icon="⚖️", 
    layout="wide"
)

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
            "Локация": f.location_name, "Интервал": f"{f.t_start.split()[-1] if f.t_start else ''} — {f.t_end.split()[-1] if f.t_end else ''}",
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
            t_lbl = f"[{f.t_start.split()[-1]}-{f.t_end.split()[-1]}]" if f.t_start and f.t_end else "[Без времени]"
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
            "engine_version": "AI-Forensic-Core v2.5",
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
