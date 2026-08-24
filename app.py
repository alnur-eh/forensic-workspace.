"""
AI Forensic Workspace — Frontend Interface
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time
import tempfile
import os
from pyvis.network import Network
import plotly.express as px

from discrepancy_engine import (
    Location, AtomicFact, Predicate, AnalysisConfig,
    ForensicCollisionEngine, SmartFreeTextParser, ScientificValidator,
    DatabaseManager, calculate_distance
)

st.set_page_config(
    page_title="AI Forensic Workspace", 
    page_icon="⚖️", 
    layout="wide"
)

# Абсолютно неуязвимая инициализация данных
if "locations" not in st.session_state or "facts" not in st.session_state:
    try:
        def_locs, def_facts = DatabaseManager.get_default_dataset()
    except Exception:
        def_locs = {
            "Кабинет 305": Location("Кабинет 305", 120.0, 40.0, "Зона лаборатории"),
            "Библиотека": Location("Библиотека", 300.0, 150.0, "Читальный зал"),
            "Центральный вход": Location("Центральный вход", 0.0, 0.0, "КПП и турникеты"),
            "Столовая": Location("Столовая", -50.0, 80.0, "Общественная зона"),
            "Парковка": Location("Парковка", 250.0, -100.0, "Северная автостоянка")
        }
        def_facts = [
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
    st.session_state.locations = def_locs
    st.session_state.facts = def_facts

st.sidebar.title("⚙️ Экспертные параметры")
walk_speed = st.sidebar.slider("Порог скорости шага (км/ч)", 2.0, 8.0, 5.0, 0.5)
sprint_speed = st.sidebar.slider("Порог скорости бега (км/ч)", 8.0, 30.0, 18.0, 1.0)
veh_speed = st.sidebar.slider("Макс. скорость транспорта (км/ч)", 40.0, 150.0, 90.0, 5.0)
radius_same = st.sidebar.slider("Погрешность координат точки (м)", 0.5, 10.0, 2.0, 0.5)
weight_gap = st.sidebar.slider("Критический дисбаланс весов", 0.1, 0.9, 0.45, 0.05)

config = AnalysisConfig(
    max_walking_speed_kmh=walk_speed,
    max_sprint_speed_kmh=sprint_speed,
    max_vehicle_speed_kmh=veh_speed,
    same_location_radius_m=radius_same,
    critical_weight_gap=weight_gap
)

st.title("⚖️ AI Forensic Workspace")
st.caption("Аналитический комплекс интеллектуального аудита доказательств и детекции коллизий")

tab_add, tab_registry, tab_map2d, tab_graph, tab_analysis, tab_benchmark = st.tabs([
    "📥 Добавление материалов",
    "📋 Реестр доказательств",
    "🗺️ 2D-Карта (X / Y)",
    "🕸️ Топология связей",
    "🚨 Экспертиза коллизий",
    "🔬 Научный бенчмарк"
])

with tab_add:
    sub1, sub2, sub3 = st.tabs([
        "⚡ Поштучный ввод",
        "✍️ Ручной конструктор",
        "📁 Импорт файлов (.txt, .json)"
    ])
    
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
                f_mot = st.text_input("Мотивационный профиль", "Нейтральный свидетель")
                f_conf = st.slider("Конфликт интересов", 0.0, 1.0, 0.2, 0.05)
            f_q = st.text_area("Цитата источника", "Находился в указанном месте.")
            if st.form_submit_button("💾 Сохранить факт"):
                new_atom = AtomicFact(
                    f"F-{len(st.session_state.facts)+1:02d}", f_src, f_type, f_s, f_p, f_l,
                    f_t1, f_t2, f_w, f_q, f_mot, f_conf
                )
                st.session_state.facts.append(new_atom)
                st.success("Факт сохранен.")
                st.rerun()

    with sub3:
        st.markdown("**Пакетная загрузка документов (лимит 5 МБ):**")
        up_files = st.file_uploader("Файлы протоколов:", type=["txt", "json"], accept_multiple_files=True)
        mode = st.radio("Режим загрузки:", ["Дописать к текущим", "Перезаписать базу"], horizontal=True)
        if st.button("⚡ Обработать файлы"):
            if up_files:
                combined = ""
                for uf in up_files:
                    if uf.size > 5 * 1024 * 1024:
                        st.error(f"Файл {uf.name} превышает 5 МБ.")
                        continue
                    try:
                        content = uf.getvalue().decode("utf-8")
                    except UnicodeDecodeError:
                        content = uf.getvalue().decode("cp1251", errors="replace")
                    combined += f"\n{content}"
                
                if combined.strip():
                    start_idx = 1 if mode == "Перезаписать базу" else len(st.session_state.facts) + 1
                    b_facts, b_locs = SmartFreeTextParser.parse_documents(
                        combined, default_date="2026-10-12",
                        current_locs=st.session_state.locations,
                        start_id=start_idx
                    )
                    if mode == "Перезаписать базу":
                        st.session_state.facts = b_facts
                    else:
                        st.session_state.facts.extend(b_facts)
                    st.session_state.locations = b_locs
                    st.success(f"Обработано {len(b_facts)} записей.")
                    st.rerun()

with tab_registry:
    st.subheader("Реестр формализованных доказательств")
    if st.session_state.facts:
        f_df = pd.DataFrame([{
            "ID": f.fact_id, "Субъект": f.subject, "Предикат": f.predicate,
            "Локация": f.location_name, "Интервал": f"{f.t_start[-5:] if len(f.t_start) >= 5 else ''} — {f.t_end[-5:] if len(f.t_end) >= 5 else ''}",
            "Источник": f.source_id, "Вес": f.weight, "Конфликт": f.interest_conflict
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
            fig = px.scatter(loc_df, x="X", y="Y", text="Локация", hover_data=["Описание"],
                             title="План объекта (сетка в метрах)", template="plotly_dark")
            fig.update_traces(marker=dict(size=14, color="#00E676", line=dict(width=2, color="white")),
                              textposition="top center", textfont=dict(size=13, color="white"))
            fig.update_layout(xaxis_title="Ось X (метры)", yaxis_title="Ось Y (метры)", height=430)
            st.plotly_chart(fig, use_container_width=True)

with tab_graph:
    st.subheader("🕸️ Граф темпоральных связей")
    if st.session_state.facts:
        net = Network(height="460px", width="100%", bgcolor="#0E1117", font_color="white")
        net.force_atlas_2based()
        added = set()
        for f in st.session_state.facts:
            if f.subject not in added:
                net.add_node(f.subject, label=f.subject, color="#1E88E5", size=25)
                added.add(f.subject)
            if f.location_name not in added:
                net.add_node(f.location_name, label=f.location_name, color="#43A047", size=22, shape="box")
                added.add(f.location_name)
            col = "#E53935" if f.predicate == Predicate.ABSENT.value else "#90CAF9"
            t_label = f"[{f.t_start[-5:]}-{f.t_end[-5:]}]" if len(f.t_start) >= 5 and len(f.t_end) >= 5 else "[Без времени]"
            net.add_edge(f.subject, f.location_name, label=t_label, color=col)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
            net.save_graph(tmp.name)
            tmp_path = tmp.name
        with open(tmp_path, "r", encoding="utf-8") as f_html:
            components.html(f_html.read(), height=480)
        os.remove(tmp_path)

with tab_analysis:
    st.subheader("🚨 Экспертный протокол выявленных несогласованностей")
    engine = ForensicCollisionEngine(config=config)
    results = engine.analyze(st.session_state.facts, st.session_state.locations)
    
    st.metric("Выявлено потенциальных коллизий", len(results))
    for item in results:
        with st.expander(f"⚠️ [{item['id']}] {item['type']} — {item['subject']}", expanded=True):
            st.write(f"**Характер несогласованности:** {item['details']}")
            st.write(f"**Темпоральное отношение Аллена:** `{item['allen_relation']}`")
            st.info(f"📋 **Рекомендация эксперту:** {item['expert_note']}")
            c1, c2 = st.columns(2)
            f1, f2 = item['facts'][0], item['facts'][1]
            with c1:
                st.error(f"**Утверждение А ({f1.fact_id})**\n* Источник: `{f1.source_id}` (вес {f1.weight})\n* Предикат: *{f1.predicate}* в **{f1.location_name}**\n* Время: {f1.t_start} — {f1.t_end}\n* Цитата: *«{f1.source_excerpt}»*")
            with c2:
                st.warning(f"**Утверждение Б ({f2.fact_id})**\n* Источник: `{f2.source_id}` (вес {f2.weight})\n* Предикат: *{f2.predicate}* в **{f2.location_name}**\n* Время: {f2.t_start} — {f2.t_end}\n* Цитата: *«{f2.source_excerpt}»*")

with tab_benchmark:
    st.subheader("🔬 Экспериментальная валидация точности и масштабируемости")
    b1, b2 = st.tabs(["🎯 Метрики классификации (Ground Truth)", "⚡ Сложность алгоритма O(N²)"])
    
    with b1:
        col_bn1, col_bn2 = st.columns([1, 2])
        with col_bn1:
            samples = st.selectbox("Размер контрольной выборки", [100, 250, 500, 1000], index=1)
            rate = st.slider("Доля аномалий в датасете", 0.1, 0.9, 0.5, 0.1)
            noise = st.checkbox("Учитывать погрешность очевидцев (±4 мин)", value=True)
            run_btn = st.button("🚀 Провести валидацию")
        
        if run_btn:
            val_res = ScientificValidator.run_benchmark(engine, test_samples=samples, anomaly_rate=rate, add_noise=noise)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Точность (Accuracy)", f"{val_res['accuracy']}%")
            m2.metric("Precision", f"{val_res['precision']}%")
            m3.metric("Recall", f"{val_res['recall']}%")
            m4.metric("F1-Score", f"{val_res['f1_score']}%")
            
            cm_data = {
                "Факт: Есть коллизия": [f"TP: {val_res['tp']}", f"FN: {val_res['fn']}"],
                "Факт: Нет коллизии": [f"FP: {val_res['fp']}", f"TN: {val_res['tn']}"]
            }
            st.table(pd.DataFrame(cm_data, index=["Система нашла коллизию", "Система сочла алиби согласованным"]))

    with b2:
        if st.button("⚡ Замерить скорость O(N²)"):
            counts = [10, 50, 100, 250, 500, 1000]
            times = []
            test_loc = list(st.session_state.locations.values())[0]
            for n in counts:
                synth = [AtomicFact(f"S-{i}", f"Камера #{i%10}", "камера", f"Субъект_{i%4}",
                                   Predicate.PRESENT.value, test_loc.name, 
                                   "2026-10-12 14:00", "2026-10-12 14:30", 1.0, "Лог") for i in range(n)]
                t0 = time.perf_counter()
                engine.analyze(synth, st.session_state.locations)
                times.append((time.perf_counter() - t0) * 1000)
            res_df = pd.DataFrame({"Объем фактов (N)": counts, "Время анализа (мс)": times})
            st.line_chart(res_df.set_index("Объем фактов (N)"))
            st.success(f"1000 фактов обработано за {times[-1]:.2f} мс.")
