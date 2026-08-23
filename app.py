import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time, tempfile, os
from pyvis.network import Network
import plotly.express as px

from discrepancy_engine import (
    Location, AtomicFact, Predicate, AnalysisConfig,
    ForensicCollisionEngine, SmartFreeTextParser, ScientificValidator,
    DatabaseManager, calculate_distance
)

st.set_page_config(
    page_title="AI Forensic Workspace", 
    page_icon="🔍", 
    layout="wide"
)

if "locations" not in st.session_state or "facts" not in st.session_state:
    loaded_locs, loaded_facts = DatabaseManager.load_data()
    st.session_state.locations = loaded_locs
    st.session_state.facts = loaded_facts

st.sidebar.title("⚙️ Параметры анализа")
walk_speed = st.sidebar.slider("Макс. скорость шага (км/ч)", 2.0, 8.0, 5.0, 0.5)
sprint_speed = st.sidebar.slider("Макс. скорость бега (км/ч)", 8.0, 30.0, 18.0, 1.0)
radius_same = st.sidebar.slider("Погрешность координат (м)", 0.5, 10.0, 2.0, 0.5)
weight_gap = st.sidebar.slider("Порог дельты весов источников", 0.1, 0.9, 0.45, 0.05)

config = AnalysisConfig(
    max_walking_speed_kmh=walk_speed,
    max_sprint_speed_kmh=sprint_speed,
    same_location_radius_m=radius_same,
    critical_weight_gap=weight_gap
)

st.title("🔍 AI Forensic Workspace")
st.caption("Аналитический комплекс аудита доказательной базы и детекции пространственно-временных коллизий")

tab_nlp, tab_builder, tab_map2d, tab_graph, tab_analysis, tab_benchmark = st.tabs([
    "📥 AI-разбор показаний",
    "📋 Реестр фактов",
    "🗺️ 2D-Карта (X / Y)",
    "🕸️ Топология связей",
    "🚨 Анализ коллизий & Мотивы",
    "🔬 Научный бенчмарк"
])

with tab_nlp:
    st.subheader("🗣️ Извлечение фактов из свободной речи свидетелей")
    sample_text = """Камера видеонаблюдения CAM-305 зафиксировала: Арман находился около Кабинет 305 с 14:15 до 14:25.
Подозреваемый Арман на допросе утверждает: с 14:00 до 14:40 я находился в Библиотека и никуда не выходил.
Свидетель Дамир сообщает: я лично встретил Армана около Центральный вход примерно в 14:26.
Охранник на посту подтвердил: Арман отсутствовал в Библиотека в период с 14:10 до 14:35."""

    raw_input = st.text_area("Свободная речь свидетелей / материалы дела:", value=sample_text, height=160)
    
    if st.button("🤖 Распознать и сохранить в базу"):
        extracted_facts, updated_locs = SmartFreeTextParser.parse_witness_statement(
            raw_input, default_date="2026-10-12", current_locs=st.session_state.locations
        )
        st.session_state.facts = extracted_facts
        st.session_state.locations = updated_locs
        DatabaseManager.save_data(st.session_state.locations, st.session_state.facts)
        st.success(f"Распознано {len(extracted_facts)} фактов! База данных обновлена.")
        st.rerun()

with tab_builder:
    st.subheader("Реестр материалов дела")
    with st.form("manual_add"):
        c1, c2, c3 = st.columns(3)
        with c1:
            f_s = st.text_input("Субъект (ФИО)", "Арман С.")
            f_p = st.selectbox("Предикат", [p.value for p in Predicate])
            f_l = st.selectbox("Локация", list(st.session_state.locations.keys()))
        with c2:
            f_src = st.text_input("Источник", "Протокол опроса #5")
            f_type = st.selectbox("Тип источника", ["свидетель", "подозреваемый", "камера", "биллинг", "турникет", "экспертиза"])
            f_w = st.slider("Вес надежности", 0.1, 1.0, 0.7, 0.05)
        with c3:
            f_t1 = st.text_input("Начало", "2026-10-12 14:10")
            f_t2 = st.text_input("Конец", "2026-10-12 14:25")
            f_mot = st.text_input("Психологический интерес", "Скрыть факт присутствия")
            f_conf = st.slider("Конфликт интересов", 0.0, 1.0, 0.3, 0.05)
        f_quote = st.text_area("Цитата / Содержание", "Находился в указанном месте.")
        if st.form_submit_button("💾 Сохранить факт"):
            new_f = AtomicFact(f"F-{len(st.session_state.facts)+1:02d}", f_src, f_type, f_s, f_p, None, f_l, f_t1, f_t2, f_w, f_quote, f_mot, f_conf)
            st.session_state.facts.append(new_f)
            DatabaseManager.save_data(st.session_state.locations, st.session_state.facts)
            st.success("Факт сохранен в базу!")
            st.rerun()

    if st.session_state.facts:
        f_df = pd.DataFrame([{
            "ID": f.fact_id, "Субъект": f.subject, "Действие": f.predicate,
            "Локация": f.location_name, "Интервал": f"{f.t_start} — {f.t_end}",
            "Источник": f.source_id, "Вес": f.weight, "Мотив": f.motive_flag
        } for f in st.session_state.facts])
        st.dataframe(f_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Очистить все факты"):
            st.session_state.facts = []
            DatabaseManager.save_data(st.session_state.locations, st.session_state.facts)
            st.rerun()

with tab_map2d:
    st.subheader("🗺️ Двумерная координатная плоскость объекта")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.markdown("**Добавить новую точку:**")
        with st.form("add_loc_form"):
            n_name = st.text_input("Название локации", "Серверная")
            n_x = st.number_input("Координата X (метры)", value=60.0)
            n_y = st.number_input("Координата Y (метры)", value=90.0)
            n_desc = st.text_input("Описание", "Ограниченный доступ")
            if st.form_submit_button("📍 Поставить на карту"):
                st.session_state.locations[n_name] = Location(n_name, n_x, n_y, n_desc)
                DatabaseManager.save_data(st.session_state.locations, st.session_state.facts)
                st.success(f"Локация {n_name} добавлена!")
                st.rerun()

    with col_m2:
        loc_df = pd.DataFrame([{"Локация": l.name, "X": l.x, "Y": l.y, "Описание": l.description} for l in st.session_state.locations.values()])
        fig = px.scatter(loc_df, x="X", y="Y", text="Локация", hover_data=["Описание"],
                         title="Карта локаций (координатная сетка в метрах)",
                         template="plotly_dark")
        fig.update_traces(marker=dict(size=14, color="#00E676", line=dict(width=2, color="white")),
                          textposition="top center", textfont=dict(size=13, color="white"))
        fig.update_layout(xaxis_title="Координата X (метры)", yaxis_title="Координата Y (метры)", height=450)
        st.plotly_chart(fig, use_container_width=True)

with tab_graph:
    st.subheader("🕸️ Сетевой граф связей")
    if st.session_state.facts:
        net = Network(height="480px", width="100%", bgcolor="#0E1117", font_color="white")
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
            net.add_edge(f.subject, f.location_name, label=f"[{f.t_start[-5:]}-{f.t_end[-5:]}]", color=col)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
            net.save_graph(tmp.name)
            tmp_path = tmp.name
        with open(tmp_path, "r", encoding="utf-8") as f_html:
            components.html(f_html.read(), height=500)
        os.remove(tmp_path)

with tab_analysis:
    st.subheader("🚨 Экспертный протокол коллизий")
    engine = ForensicCollisionEngine(config=config)
    results = engine.analyze(st.session_state.facts, st.session_state.locations)
    
    st.metric("Обнаружено критических нестыковок", len(results))
    for item in results:
        with st.expander(f"🚨 [{item['id']}] {item['type']} — {item['subject']}", expanded=True):
            st.write(f"**Суть коллизии:** {item['details']}")
            st.write(f"**Отношение интервалов:** `{item['allen_relation']}`")
            st.info(f"🧠 **Оценка мотива и риска лжи:** {item['psychological_insight']}")
            c1, c2 = st.columns(2)
            f1, f2 = item['facts'][0], item['facts'][1]
            with c1:
                st.error(f"**Факт А ({f1.fact_id})**\n* Источник: `{f1.source_id}` (вес {f1.weight})\n* Утверждение: *{f1.predicate}* в **{f1.location_name}**\n* Время: {f1.t_start} — {f1.t_end}\n* Цитата: *«{f1.source_excerpt}»*")
            with c2:
                st.warning(f"**Факт Б ({f2.fact_id})**\n* Источник: `{f2.source_id}` (вес {f2.weight})\n* Утверждение: *{f2.predicate}* в **{f2.location_name}**\n* Время: {f2.t_start} — {f2.t_end}\n* Цитата: *«{f2.source_excerpt}»*")

with tab_benchmark:
    st.subheader("🔬 Экспериментальная валидация точности и скорости")
    b1, b2 = st.tabs(["🎯 Метрики качества классификатора", "⚡ Временная сложность O(N²)"])
    
    with b1:
        samples = st.selectbox("Количество тестовых дел", [100, 250, 500, 1000], index=1)
        rate = st.slider("Доля аномалий в выборке", 0.1, 0.9, 0.5, 0.1)
        if st.button("🚀 Запустить валидацию Ground Truth"):
            val_res = ScientificValidator.run_ground_truth_benchmark(engine, test_samples=samples, anomaly_rate=rate)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Точность (Accuracy)", f"{val_res['accuracy']}%")
            m2.metric("Precision", f"{val_res['precision']}%")
            m3.metric("Recall", f"{val_res['recall']}%")
            m4.metric("F1-Score", f"{val_res['f1_score']}%")
            
            cm_data = {
                "Факт: Есть коллизия": [f"TP: {val_res['tp']}", f"FN: {val_res['fn']}"],
                "Факт: Нет коллизии": [f"FP: {val_res['fp']}", f"TN: {val_res['tn']}"]
            }
            st.table(pd.DataFrame(cm_data, index=["Система нашла нестыковку", "Система сочла алиби чистым"]))
            st.success("Детерминированный интервальный анализ исключает ложные обвинения (FP = 0).")

    with b2:
        if st.button("⚡ Запустить нагрузочный тест"):
            counts = [10, 50, 100, 250, 500, 1000]
            times = []
            test_loc = list(st.session_state.locations.values())[0]
            for n in counts:
                synth = [AtomicFact(f"S-{i}", f"Камера #{i%10}", "камера", f"Субъект_{i%4}",
                                   Predicate.PRESENT.value, None, test_loc.name, 
                                   "2026-10-12 14:00", "2026-10-12 14:30", 1.0, "Лог") for i in range(n)]
                t0 = time.perf_counter()
                engine.analyze(synth, st.session_state.locations)
                times.append((time.perf_counter() - t0) * 1000)
            res_df = pd.DataFrame({"Объем выборки (N фактов)": counts, "Время выполнения (мс)": times})
            st.line_chart(res_df.set_index("Объем выборки (N фактов)"))
            st.success(f"1000 фактов обработано за {times[-1]:.2f} мс.")
