from __future__ import annotations

import gc
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import simpy
import streamlit as st


# ============================================================
# KONFIGURASI APLIKASI
# ============================================================
st.set_page_config(
    page_title="Dashboard Simulasi Kiosk Bioskop",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SCENARIO_ORDER = [
    "Tanpa Intervensi",
    "Intervensi Reaktif",
    "Intervensi Preventif",
    "Beban Tinggi",
]

SCENARIOS: dict[str, dict[str, Any]] = {
    "Tanpa Intervensi": {
        "scenario_type": "base",
        "num_kiosks": 2,
        "mean_interarrival_time": 3.0,
        "mean_service_time": 4.0,
        "patience_time": 12.0,
    },
    "Intervensi Reaktif": {
        "scenario_type": "reactive",
        "num_kiosks": 2,
        "mean_interarrival_time": 3.0,
        "mean_service_time": 4.0,
        "patience_time": 12.0,
    },
    "Intervensi Preventif": {
        "scenario_type": "preventive",
        "num_kiosks": 3,
        "mean_interarrival_time": 3.0,
        "mean_service_time": 3.6,
        "patience_time": 12.0,
    },
    "Beban Tinggi": {
        "scenario_type": "high_load",
        "num_kiosks": 2,
        "mean_interarrival_time": 2.0,
        "mean_service_time": 4.5,
        "patience_time": 10.0,
    },
}

METRIC_LABELS = {
    "avg_wait_time": "Rata-rata waktu tunggu (menit)",
    "avg_wait_served": "Waktu tunggu pelanggan selesai (menit)",
    "avg_queue_length": "Rata-rata panjang antrean",
    "max_queue_length": "Panjang antrean maksimum",
    "utilization_percent": "Utilisasi kiosk (%)",
    "throughput_per_hour": "Throughput (pelanggan/jam)",
    "abandonment_percent": "Pelanggan batal (%)",
}

REQUIRED_DATA_FILES = {
    "Ringkasan Monte Carlo": "ringkasan_skenario_1000_iterasi.csv",
    "Hasil Monte Carlo": "hasil_monte_carlo_1000_iterasi.csv",
    "Dataset pelanggan": "dataset_pelanggan_kiosk_bioskop.csv",
    "Jejak antrean": "dataset_jejak_antrean_per_menit.csv",
    "Uji statistik": "hasil_uji_statistik.csv",
    "Optimasi kiosk": "ringkasan_optimasi_kiosk.csv",
    "Sensitivitas": "hasil_sensitivitas_interval_kedatangan.csv",
    "Kondisi ekstrem": "hasil_uji_kondisi_ekstrem.csv",
    "Kamus data": "kamus_data.csv",
    "Parameter skenario": "parameter_skenario.csv",
}


# ============================================================
# TAMPILAN
# ============================================================
st.html(
    """
    <style>
    :root {
        --navy: #0B1F3A;
        --blue: #1E5AA8;
        --gold: #D6A84B;
        --ink: #182230;
        --muted: #64748B;
        --surface: #FFFFFF;
        --soft: #F5F7FB;
        --border: #DCE3EC;
    }
    .stApp { background: #F7F9FC; }
    .main .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }
    section[data-testid="stSidebar"] { background: #F1F5F9; }
    .hero {
        background: linear-gradient(135deg, #0B1F3A 0%, #164B86 68%, #1E5AA8 100%);
        border-radius: 22px;
        padding: 30px 32px;
        color: white;
        box-shadow: 0 16px 38px rgba(11, 31, 58, 0.18);
        margin-bottom: 18px;
    }
    .hero .eyebrow {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #D8E7FA;
        margin-bottom: 8px;
    }
    .hero h1 {
        color: white;
        font-size: clamp(1.7rem, 3vw, 2.65rem);
        line-height: 1.15;
        margin: 0 0 10px 0;
    }
    .hero p {
        color: #E8F0FA;
        max-width: 1050px;
        line-height: 1.55;
        margin: 0;
    }
    .badge {
        display: inline-block;
        margin: 14px 7px 0 0;
        padding: 6px 11px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.25);
        background: rgba(255,255,255,.11);
        color: white;
        font-size: .79rem;
        font-weight: 600;
    }
    .info-card {
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 16px;
        padding: 18px 19px;
        min-height: 145px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, .05);
    }
    .info-card h3 { margin: 0 0 8px 0; color: var(--navy); font-size: 1.05rem; }
    .info-card p { margin: 0; color: #475569; line-height: 1.55; font-size: .92rem; }
    .note-box {
        background: #EEF5FF;
        border: 1px solid #C9DCF7;
        border-left: 5px solid #1E5AA8;
        border-radius: 13px;
        padding: 14px 16px;
        color: #183B67;
        margin: 9px 0 15px 0;
    }
    .warning-box {
        background: #FFF8E8;
        border: 1px solid #F1D58A;
        border-left: 5px solid #D6A84B;
        border-radius: 13px;
        padding: 14px 16px;
        color: #604813;
        margin: 9px 0 15px 0;
    }
    .footer {
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: .82rem;
        margin-top: 30px;
        padding-top: 13px;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 14px 16px;
        box-shadow: 0 5px 14px rgba(15,23,42,.04);
    }
    div[data-testid="stMetricLabel"] { color: #64748B; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] {
        border: 1px solid var(--border);
        background: white;
        border-radius: 999px;
        padding-left: 15px;
        padding-right: 15px;
    }
    @media (max-width: 760px) {
        .hero { padding: 22px 20px; border-radius: 17px; }
        .main .block-container { padding-left: .8rem; padding-right: .8rem; }
    }
    </style>
    """
)


# ============================================================
# UTILITAS DATA
# ============================================================
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_bundled_data() -> dict[str, pd.DataFrame]:
    return {
        key: safe_read_csv(DATA_DIR / filename)
        for key, filename in REQUIRED_DATA_FILES.items()
    }


def ordered_frame(df: pd.DataFrame, scenario_col: str = "scenario") -> pd.DataFrame:
    result = df.copy()
    if scenario_col in result.columns:
        result[scenario_col] = pd.Categorical(
            result[scenario_col], categories=SCENARIO_ORDER, ordered=True
        )
        result = result.sort_values(scenario_col).reset_index(drop=True)
        result[scenario_col] = result[scenario_col].astype("string")
    return result


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def format_number(value: Any, digits: int = 2) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(value_float):
        return "–"
    return f"{value_float:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def make_state_chart() -> go.Figure:
    nodes = {
        "Datang": (0.05, 0.55),
        "Menunggu": (0.30, 0.55),
        "Dilayani": (0.57, 0.55),
        "Selesai": (0.84, 0.55),
        "Batal": (0.30, 0.12),
    }
    fig = go.Figure()
    for name, (x, y) in nodes.items():
        fig.add_shape(
            type="rect", x0=x, y0=y, x1=x + 0.15, y1=y + 0.16,
            line=dict(color="#1E5AA8", width=2), fillcolor="#EEF5FF",
            layer="below",
        )
        fig.add_annotation(x=x + 0.075, y=y + 0.08, text=f"<b>{name}</b>", showarrow=False)

    arrows = [
        ((0.20, 0.63), (0.30, 0.63), "masuk antrean"),
        ((0.45, 0.63), (0.57, 0.63), "kiosk tersedia"),
        ((0.72, 0.63), (0.84, 0.63), "pelayanan selesai"),
        ((0.375, 0.55), (0.375, 0.28), "batas sabar habis"),
    ]
    for start, end, label in arrows:
        fig.add_annotation(
            x=end[0], y=end[1], ax=start[0], ay=start[1],
            xref="x", yref="y", axref="x", ayref="y",
            text=label, showarrow=True, arrowhead=3, arrowsize=1,
            arrowwidth=1.6, arrowcolor="#64748B", font=dict(size=11),
        )
    fig.update_xaxes(visible=False, range=[0, 1.03])
    fig.update_yaxes(visible=False, range=[0, 0.9])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white")
    return fig


# ============================================================
# MESIN SIMULASI — SAMA DENGAN NOTEBOOK FINAL
# ============================================================
def generate_service_time(mean_service_time: float, rng: np.random.Generator) -> float:
    sigma = 0.25
    mu = np.log(mean_service_time) - 0.5 * sigma**2
    return max(0.1, float(rng.lognormal(mu, sigma)))


def generate_interarrival_time(mean_interarrival_time: float, rng: np.random.Generator) -> float:
    return max(0.1, float(rng.exponential(mean_interarrival_time)))


def simulate_kiosk(
    seed: int = 42,
    duration: int = 480,
    num_kiosks: int = 2,
    mean_interarrival_time: float = 3.0,
    mean_service_time: float = 4.0,
    patience_time: float = 12.0,
    scenario_name: str = "Simulasi Interaktif",
    scenario_type: str = "base",
    reactive_threshold: int = 4,
    reactive_service_reduction: float = 0.85,
    collect_details: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    if duration <= 0:
        raise ValueError("Durasi harus lebih besar dari nol.")
    if num_kiosks < 1:
        raise ValueError("Jumlah kiosk minimal satu.")
    if min(mean_interarrival_time, mean_service_time, patience_time) <= 0:
        raise ValueError("Seluruh parameter waktu harus lebih besar dari nol.")

    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    kiosk = simpy.Resource(env, capacity=int(num_kiosks))
    customer_records: list[dict[str, Any]] = []
    queue_records: list[dict[str, Any]] = []

    arrivals = served = served_within_horizon = abandoned = 0
    wait_all_sum = wait_served_sum = wait_abandoned_sum = 0.0
    system_time_sum = busy_time_within_horizon = 0.0
    queue_wait_time_within_horizon = 0.0
    max_queue = 0

    def monitor_queue():
        while env.now <= duration:
            queue_records.append(
                {
                    "time": float(env.now),
                    "queue_length": len(kiosk.queue),
                    "in_service": int(kiosk.count),
                    "scenario": scenario_name,
                    "seed": seed,
                }
            )
            yield env.timeout(1)

    def customer_process(customer_id: int):
        nonlocal served, served_within_horizon, abandoned
        nonlocal wait_all_sum, wait_served_sum, wait_abandoned_sum
        nonlocal system_time_sum, busy_time_within_horizon
        nonlocal queue_wait_time_within_horizon, max_queue

        arrival_time = float(env.now)
        record = {
            "customer_id": customer_id,
            "scenario": scenario_name,
            "arrival_time": arrival_time,
            "start_service_time": np.nan,
            "departure_time": np.nan,
            "wait_time": np.nan,
            "service_time": np.nan,
            "system_time": np.nan,
            "state": "menunggu",
            "patience_time": patience_time,
            "num_kiosks": num_kiosks,
            "mean_interarrival_time": mean_interarrival_time,
            "mean_service_time": mean_service_time,
            "duration": duration,
            "seed": seed,
        }

        with kiosk.request() as request:
            max_queue = max(max_queue, len(kiosk.queue))
            outcome = yield request | env.timeout(patience_time)
            event_time = float(env.now)
            queue_wait_time_within_horizon += max(
                0.0, min(event_time, duration) - min(arrival_time, duration)
            )

            if request not in outcome:
                abandoned += 1
                wait_time = event_time - arrival_time
                wait_all_sum += wait_time
                wait_abandoned_sum += wait_time
                system_time_sum += wait_time
                record.update(
                    {
                        "departure_time": event_time,
                        "wait_time": wait_time,
                        "service_time": 0.0,
                        "system_time": wait_time,
                        "state": "batal",
                    }
                )
                if collect_details:
                    customer_records.append(record)
                return

            start_service_time = event_time
            wait_time = start_service_time - arrival_time
            effective_service_time = mean_service_time
            if scenario_type == "reactive" and len(kiosk.queue) >= reactive_threshold:
                effective_service_time *= reactive_service_reduction

            service_time = generate_service_time(effective_service_time, rng)
            planned_departure = start_service_time + service_time
            busy_time_within_horizon += max(
                0.0,
                min(planned_departure, duration) - min(start_service_time, duration),
            )

            yield env.timeout(service_time)
            departure_time = float(env.now)
            served += 1
            if departure_time <= duration:
                served_within_horizon += 1

            wait_all_sum += wait_time
            wait_served_sum += wait_time
            system_time_sum += wait_time + service_time
            record.update(
                {
                    "start_service_time": start_service_time,
                    "departure_time": departure_time,
                    "wait_time": wait_time,
                    "service_time": service_time,
                    "system_time": wait_time + service_time,
                    "state": "selesai",
                }
            )
            if collect_details:
                customer_records.append(record)

    def arrival_source():
        nonlocal arrivals
        while env.now < duration:
            arrivals += 1
            env.process(customer_process(arrivals))
            yield env.timeout(generate_interarrival_time(mean_interarrival_time, rng))

    env.process(arrival_source())
    if collect_details:
        env.process(monitor_queue())
    env.run()

    summary = {
        "scenario": scenario_name,
        "scenario_type": scenario_type,
        "seed": seed,
        "total_arrivals": arrivals,
        "served_customers": served,
        "served_within_horizon": served_within_horizon,
        "abandoned_customers": abandoned,
        "avg_wait_time": wait_all_sum / arrivals if arrivals else 0.0,
        "avg_wait_served": wait_served_sum / served if served else 0.0,
        "avg_wait_abandoned": wait_abandoned_sum / abandoned if abandoned else 0.0,
        "avg_system_time": system_time_sum / arrivals if arrivals else 0.0,
        "avg_queue_length": queue_wait_time_within_horizon / duration,
        "max_queue_length": max_queue,
        "utilization_percent": busy_time_within_horizon / (num_kiosks * duration) * 100,
        "throughput_per_hour": served_within_horizon / (duration / 60),
        "abandonment_percent": abandoned / arrivals * 100 if arrivals else 0.0,
        "num_kiosks": num_kiosks,
        "mean_interarrival_time": mean_interarrival_time,
        "mean_service_time": mean_service_time,
        "patience_time": patience_time,
        "duration": duration,
        "reactive_threshold": reactive_threshold,
    }
    details = pd.DataFrame(customer_records) if collect_details else None
    queue_details = pd.DataFrame(queue_records) if collect_details else None
    return summary, details, queue_details


@st.cache_data(show_spinner=False, max_entries=24)
def run_light_monte_carlo(
    n_iterations: int,
    seed: int,
    duration: int,
    num_kiosks: int,
    mean_interarrival_time: float,
    mean_service_time: float,
    patience_time: float,
    scenario_name: str,
    scenario_type: str,
    reactive_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    first_details = pd.DataFrame()
    first_queue = pd.DataFrame()
    for iteration in range(n_iterations):
        result, details, queue = simulate_kiosk(
            seed=seed + iteration,
            duration=duration,
            num_kiosks=num_kiosks,
            mean_interarrival_time=mean_interarrival_time,
            mean_service_time=mean_service_time,
            patience_time=patience_time,
            scenario_name=scenario_name,
            scenario_type=scenario_type,
            reactive_threshold=reactive_threshold,
            collect_details=(iteration == 0),
        )
        result["iteration"] = iteration + 1
        rows.append(result)
        if iteration == 0:
            first_details = details if details is not None else pd.DataFrame()
            first_queue = queue if queue is not None else pd.DataFrame()
    gc.collect()
    return pd.DataFrame(rows), first_details, first_queue


@st.cache_data(show_spinner=False, max_entries=12)
def run_light_optimization(
    max_kiosks: int,
    n_iterations: int,
    seed: int,
    duration: int,
    mean_interarrival_time: float,
    mean_service_time: float,
    patience_time: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kiosks in range(1, max_kiosks + 1):
        for iteration in range(n_iterations):
            result, _, _ = simulate_kiosk(
                seed=seed + kiosks * 10_000 + iteration,
                duration=duration,
                num_kiosks=kiosks,
                mean_interarrival_time=mean_interarrival_time,
                mean_service_time=mean_service_time,
                patience_time=patience_time,
                scenario_name=f"{kiosks} Kiosk",
                collect_details=False,
            )
            rows.append(result)
    raw = pd.DataFrame(rows)
    return (
        raw.groupby("num_kiosks", as_index=False)[
            [
                "avg_wait_time",
                "avg_queue_length",
                "utilization_percent",
                "throughput_per_hour",
                "abandonment_percent",
            ]
        ]
        .mean()
        .sort_values("num_kiosks")
    )


@st.cache_data(show_spinner=False, max_entries=12)
def run_light_sensitivity(
    values: tuple[float, ...],
    parameter_name: str,
    n_iterations: int,
    seed: int,
    duration: int,
    num_kiosks: int,
    mean_interarrival_time: float,
    mean_service_time: float,
    patience_time: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value_index, value in enumerate(values):
        summaries: list[dict[str, Any]] = []
        for iteration in range(n_iterations):
            params = {
                "mean_interarrival_time": mean_interarrival_time,
                "mean_service_time": mean_service_time,
                "patience_time": patience_time,
            }
            params[parameter_name] = float(value)
            result, _, _ = simulate_kiosk(
                seed=seed + value_index * 10_000 + iteration,
                duration=duration,
                num_kiosks=num_kiosks,
                scenario_name=f"{parameter_name}={value:.2f}",
                collect_details=False,
                **params,
            )
            summaries.append(result)
        temp = pd.DataFrame(summaries)
        rows.append(
            {
                parameter_name: value,
                "avg_wait_time": temp["avg_wait_time"].mean(),
                "avg_queue_length": temp["avg_queue_length"].mean(),
                "utilization_percent": temp["utilization_percent"].mean(),
                "abandonment_percent": temp["abandonment_percent"].mean(),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# UPLOAD DAN STANDARDISASI DATASET
# ============================================================
def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix == ".csv":
        try:
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1", sep=None, engine="python")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Format file tidak didukung.")


def normalize_state(series: pd.Series) -> pd.Series:
    mapping_finished = {
        "selesai", "served", "complete", "completed", "done", "success",
        "dilayani", "selesai dilayani",
    }
    mapping_cancelled = {
        "batal", "abandoned", "cancelled", "canceled", "leave", "left",
        "batal menunggu", "dropout",
    }

    def convert(value: Any) -> str:
        text = str(value).strip().lower()
        if text in mapping_finished:
            return "selesai"
        if text in mapping_cancelled:
            return "batal"
        return text

    return series.map(convert)


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    result = df.copy()
    rename_map = {source: target for target, source in mapping.items() if source}
    result = result.rename(columns=rename_map)

    numeric_columns = [
        "arrival_time", "start_service_time", "departure_time", "wait_time",
        "service_time", "system_time", "patience_time", "num_kiosks",
        "mean_interarrival_time", "mean_service_time", "duration", "seed",
    ]
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if "wait_time" not in result.columns and {"start_service_time", "arrival_time"}.issubset(result.columns):
        result["wait_time"] = result["start_service_time"] - result["arrival_time"]
    if "service_time" not in result.columns and {"departure_time", "start_service_time"}.issubset(result.columns):
        result["service_time"] = result["departure_time"] - result["start_service_time"]
    if "system_time" not in result.columns and {"departure_time", "arrival_time"}.issubset(result.columns):
        result["system_time"] = result["departure_time"] - result["arrival_time"]
    if "scenario" not in result.columns:
        result["scenario"] = "Dataset Upload"
    if "state" in result.columns:
        result["state"] = normalize_state(result["state"])
    return result


def uploaded_metrics(df: pd.DataFrame) -> dict[str, float]:
    total = len(df)
    states = df["state"].astype(str).str.lower() if "state" in df.columns else pd.Series(dtype=str)
    served = int((states == "selesai").sum())
    abandoned = int((states == "batal").sum())
    return {
        "total": total,
        "served": served,
        "abandoned": abandoned,
        "avg_wait": float(df["wait_time"].mean()) if "wait_time" in df.columns else np.nan,
        "avg_service": float(df["service_time"].mean()) if "service_time" in df.columns else np.nan,
        "avg_system": float(df["system_time"].mean()) if "system_time" in df.columns else np.nan,
        "abandonment": abandoned / total * 100 if total else 0.0,
    }


# ============================================================
# DATA BAWAAN DAN SIDEBAR
# ============================================================
data = load_bundled_data()
missing_files = [
    filename for filename in REQUIRED_DATA_FILES.values()
    if not (DATA_DIR / filename).exists()
]

with st.sidebar:
    st.title("Informasi Project")
    st.markdown(
        "**Tsabita Arni Safitri Azzila**  \n"
        "NIM: 202310370311155  \n"
        "Program Studi Informatika  \n"
        "Universitas Muhammadiyah Malang"
    )
    st.divider()
    st.markdown("**Mode dashboard**")
    st.success("Hasil final 1.000 iterasi dibaca dari CSV.")
    st.info("Simulasi web dibatasi maksimal 100 iterasi agar stabil di Streamlit Cloud.")
    if missing_files:
        st.error("Ada file data yang belum tersedia di folder data.")
        st.caption("\n".join(missing_files))
    else:
        st.caption("Seluruh dataset final berhasil ditemukan.")
    st.divider()
    st.caption("Versi pembaruan dashboard final • tanpa ngrok • tanpa Graphviz")

st.html(
    """
    <div class="hero">
      <div class="eyebrow">Tugas Besar Pemodelan dan Simulasi</div>
      <h1>Simulasi Sistem Antrean Self-Service Kiosk Bioskop</h1>
      <p>Dashboard akademik untuk membandingkan skenario antrean, menampilkan hasil Monte Carlo 1.000 iterasi, menjalankan simulasi ringan, mengoptimalkan jumlah kiosk, serta memvalidasi dataset eksternal.</p>
      <span class="badge">Agent-Based Modeling</span>
      <span class="badge">Discrete Event Simulation</span>
      <span class="badge">Monte Carlo</span>
      <span class="badge">What-If Analysis</span>
    </div>
    """
)

summary_final = ordered_frame(data["Ringkasan Monte Carlo"])
mc_final = ordered_frame(data["Hasil Monte Carlo"])

if not summary_final.empty:
    best_wait = summary_final.loc[summary_final["avg_wait_time_mean"].idxmin()]
    highest_abandon = summary_final.loc[summary_final["abandonment_percent_mean"].idxmax()]
    highest_throughput = summary_final.loc[summary_final["throughput_per_hour_mean"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Skenario paling stabil", str(best_wait["scenario"]))
    k2.metric("Waktu tunggu terendah", f"{best_wait['avg_wait_time_mean']:.2f} menit")
    k3.metric("Pembatalan tertinggi", f"{highest_abandon['abandonment_percent_mean']:.2f}%")
    k4.metric("Throughput tertinggi", f"{highest_throughput['throughput_per_hour_mean']:.2f}/jam")

(
    tab_overview,
    tab_results,
    tab_simulation,
    tab_upload,
    tab_optimization,
    tab_sensitivity,
    tab_method,
) = st.tabs(
    [
        "Ringkasan Project",
        "Hasil Monte Carlo Final",
        "Simulasi Interaktif",
        "Upload & Validasi Dataset",
        "Optimasi Kiosk",
        "Analisis Sensitivitas",
        "Metodologi & Ekspor",
    ]
)


# ============================================================
# TAB 1 — RINGKASAN
# ============================================================
with tab_overview:
    st.header("Ringkasan Project")
    c1, c2 = st.columns(2)
    with c1:
        st.html(
            """
            <div class="info-card">
              <h3>Masalah yang dimodelkan</h3>
              <p>Ketika kedatangan pelanggan lebih cepat daripada kemampuan kiosk melayani, antrean, waktu tunggu, utilisasi, dan risiko pelanggan meninggalkan antrean dapat meningkat. Simulasi digunakan untuk menguji strategi sebelum diterapkan pada sistem nyata.</p>
            </div>
            """
        )
    with c2:
        st.html(
            """
            <div class="info-card">
              <h3>Tujuan simulasi</h3>
              <p>Membandingkan kondisi dasar, intervensi reaktif, intervensi preventif, dan beban tinggi; kemudian mencari kapasitas kiosk minimum yang mampu menjaga waktu tunggu, abandonment, dan utilisasi pada tingkat yang terkendali.</p>
            </div>
            """
        )

    st.subheader("State pelanggan")
    st.plotly_chart(make_state_chart(), width="stretch", config={"displayModeBar": False})

    st.subheader("Formulasi indikator")
    f1, f2, f3, f4 = st.columns(4)
    f1.code("Waktu tunggu = mulai dilayani − datang")
    f2.code("Waktu sistem = keluar − datang")
    f3.code("Utilisasi = busy time / kapasitas × 100%")
    f4.code("Abandonment = batal / datang × 100%")

    st.subheader("Parameter empat skenario")
    params = data["Parameter skenario"].rename(
        columns={
            "scenario_name": "Skenario",
            "scenario_type": "Tipe",
            "num_kiosks": "Jumlah kiosk",
            "mean_interarrival_time": "Interval kedatangan (menit)",
            "mean_service_time": "Waktu pelayanan (menit)",
            "patience_time": "Batas sabar (menit)",
        }
    )
    st.dataframe(params, width="stretch", hide_index=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("#### Agent-Based Modeling")
        st.write("Setiap pelanggan diperlakukan sebagai agen individual dengan waktu datang, waktu menunggu, waktu pelayanan, batas kesabaran, dan status akhir.")
    with a2:
        st.markdown("#### Discrete Event Simulation")
        st.write("Perubahan sistem terjadi pada event kedatangan, permintaan kiosk, awal pelayanan, akhir pelayanan, dan pembatalan antrean.")
    with a3:
        st.markdown("#### Monte Carlo")
        st.write("Setiap skenario dijalankan 1.000 kali menggunakan seed berbeda agar kesimpulan tidak bergantung pada satu kejadian acak.")


# ============================================================
# TAB 2 — HASIL FINAL
# ============================================================
with tab_results:
    st.header("Hasil Monte Carlo Final — 1.000 Iterasi per Skenario")
    st.html(
        '<div class="note-box">Bagian ini membaca hasil final dari CSV. Server tidak menghitung ulang 4.000 simulasi, sehingga dashboard lebih ringan dan konsisten dengan notebook serta laporan.</div>'
    )

    if summary_final.empty or mc_final.empty:
        st.error("File hasil Monte Carlo tidak tersedia.")
    else:
        display_columns = [
            "scenario",
            "avg_wait_time_mean",
            "avg_wait_time_ci95_low",
            "avg_wait_time_ci95_high",
            "avg_queue_length_mean",
            "utilization_percent_mean",
            "throughput_per_hour_mean",
            "abandonment_percent_mean",
        ]
        display_summary = summary_final[display_columns].rename(
            columns={
                "scenario": "Skenario",
                "avg_wait_time_mean": "Waktu tunggu",
                "avg_wait_time_ci95_low": "CI 95% bawah",
                "avg_wait_time_ci95_high": "CI 95% atas",
                "avg_queue_length_mean": "Panjang antrean",
                "utilization_percent_mean": "Utilisasi (%)",
                "throughput_per_hour_mean": "Throughput/jam",
                "abandonment_percent_mean": "Abandonment (%)",
            }
        )
        numeric_cols = display_summary.select_dtypes(include="number").columns
        display_summary[numeric_cols] = display_summary[numeric_cols].round(3)
        st.dataframe(display_summary, width="stretch", hide_index=True)

        metric_choice = st.selectbox(
            "Pilih indikator grafik",
            options=list(METRIC_LABELS),
            format_func=lambda key: METRIC_LABELS[key],
            key="final_metric_choice",
        )
        mean_column = f"{metric_choice}_mean"
        chart_data = summary_final[["scenario", mean_column]].rename(columns={mean_column: "value"})
        fig_bar = px.bar(
            chart_data,
            x="scenario",
            y="value",
            text_auto=".2f",
            labels={"scenario": "Skenario", "value": METRIC_LABELS[metric_choice]},
            title=f"Perbandingan {METRIC_LABELS[metric_choice]}",
        )
        fig_bar.update_layout(showlegend=False, margin=dict(t=55, l=10, r=10, b=10))
        st.plotly_chart(fig_bar, width="stretch")

        fig_box = px.box(
            mc_final,
            x="scenario",
            y=metric_choice,
            points=False,
            category_orders={"scenario": SCENARIO_ORDER},
            labels={"scenario": "Skenario", metric_choice: METRIC_LABELS[metric_choice]},
            title=f"Sebaran 1.000 Replikasi — {METRIC_LABELS[metric_choice]}",
        )
        fig_box.update_layout(margin=dict(t=55, l=10, r=10, b=10))
        st.plotly_chart(fig_box, width="stretch")

        st.subheader("Hasil uji statistik")
        stats = data["Uji statistik"].copy()
        if not stats.empty:
            stats["p_value"] = stats["p_value"].map(lambda x: f"{x:.4g}")
            stats["p_adjusted"] = stats["p_adjusted"].map(lambda x: f"{x:.4g}")
            st.dataframe(stats, width="stretch", hide_index=True)
        else:
            st.warning("File uji statistik tidak ditemukan.")

        baseline = summary_final[summary_final["scenario"] == "Tanpa Intervensi"].iloc[0]
        reactive = summary_final[summary_final["scenario"] == "Intervensi Reaktif"].iloc[0]
        preventive = summary_final[summary_final["scenario"] == "Intervensi Preventif"].iloc[0]
        high = summary_final[summary_final["scenario"] == "Beban Tinggi"].iloc[0]
        st.subheader("Interpretasi otomatis")
        st.write(
            f"Intervensi preventif menghasilkan waktu tunggu terendah, yaitu "
            f"{preventive['avg_wait_time_mean']:.2f} menit dengan abandonment "
            f"{preventive['abandonment_percent_mean']:.2f}%. Intervensi reaktif "
            f"menurunkan abandonment dari {baseline['abandonment_percent_mean']:.2f}% "
            f"menjadi {reactive['abandonment_percent_mean']:.2f}%. Pada beban tinggi, "
            f"utilisasi mencapai {high['utilization_percent_mean']:.2f}% dan abandonment "
            f"meningkat menjadi {high['abandonment_percent_mean']:.2f}%, sehingga sistem "
            "mendekati kapasitas penuh dan membutuhkan tambahan kapasitas atau strategi preventif."
        )


# ============================================================
# TAB 3 — SIMULASI INTERAKTIF
# ============================================================
with tab_simulation:
    st.header("Simulasi Interaktif Ringan")
    st.html(
        '<div class="warning-box">Untuk menjaga kestabilan Streamlit Community Cloud, simulasi interaktif dibatasi maksimal 100 iterasi. Hasil penelitian final tetap berasal dari 1.000 iterasi per skenario pada file CSV.</div>'
    )

    with st.form("interactive_form"):
        preset = st.selectbox("Preset skenario", SCENARIO_ORDER + ["Kustom"])
        preset_config = SCENARIOS.get(preset, SCENARIOS["Tanpa Intervensi"])
        c1, c2, c3, c4 = st.columns(4)
        iterations = c1.select_slider("Iterasi", options=[10, 20, 30, 50, 75, 100], value=30)
        seed_value = c2.number_input("Seed awal", min_value=1, max_value=999_999, value=2026, step=1)
        duration_value = c3.number_input("Durasi (menit)", min_value=60, max_value=720, value=480, step=30)
        kiosk_value = c4.number_input("Jumlah kiosk", min_value=1, max_value=8, value=int(preset_config["num_kiosks"]), step=1)

        c5, c6, c7, c8 = st.columns(4)
        arrival_value = c5.number_input(
            "Interval kedatangan", min_value=0.5, max_value=15.0,
            value=float(preset_config["mean_interarrival_time"]), step=0.1,
        )
        service_value = c6.number_input(
            "Waktu pelayanan", min_value=0.5, max_value=15.0,
            value=float(preset_config["mean_service_time"]), step=0.1,
        )
        patience_value = c7.number_input(
            "Batas kesabaran", min_value=1.0, max_value=60.0,
            value=float(preset_config["patience_time"]), step=1.0,
        )
        reactive_value = c8.number_input("Ambang antrean reaktif", min_value=1, max_value=20, value=4, step=1)
        run_simulation = st.form_submit_button("Jalankan simulasi")

    if run_simulation:
        scenario_type_value = preset_config["scenario_type"] if preset != "Kustom" else "base"
        with st.spinner("Menjalankan simulasi ringan..."):
            try:
                sim_mc, sim_customers, sim_queue = run_light_monte_carlo(
                    int(iterations), int(seed_value), int(duration_value), int(kiosk_value),
                    float(arrival_value), float(service_value), float(patience_value),
                    preset, scenario_type_value, int(reactive_value),
                )
                st.session_state["interactive_result"] = (sim_mc, sim_customers, sim_queue)
            except Exception as error:
                st.error(f"Simulasi gagal: {error}")

    if "interactive_result" in st.session_state:
        sim_mc, sim_customers, sim_queue = st.session_state["interactive_result"]
        mean_result = sim_mc.mean(numeric_only=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Waktu tunggu", f"{mean_result['avg_wait_time']:.2f} menit")
        m2.metric("Panjang antrean", f"{mean_result['avg_queue_length']:.2f}")
        m3.metric("Utilisasi", f"{mean_result['utilization_percent']:.2f}%")
        m4.metric("Throughput", f"{mean_result['throughput_per_hour']:.2f}/jam")
        m5.metric("Abandonment", f"{mean_result['abandonment_percent']:.2f}%")

        if mean_result["abandonment_percent"] > 10 or mean_result["avg_wait_time"] > 8:
            st.error("Sistem berada pada kondisi berat. Tambahkan kiosk atau gunakan strategi preventif.")
        elif mean_result["utilization_percent"] > 90:
            st.warning("Utilisasi melebihi 90%. Sistem mendekati kapasitas penuh.")
        else:
            st.success("Sistem relatif terkendali pada parameter yang diuji.")

        left, right = st.columns(2)
        with left:
            if not sim_queue.empty:
                fig_queue = px.line(
                    sim_queue, x="time", y=["queue_length", "in_service"],
                    labels={"time": "Waktu (menit)", "value": "Jumlah pelanggan", "variable": "Kondisi"},
                    title="Dinamika antrean — replikasi pertama",
                )
                st.plotly_chart(fig_queue, width="stretch")
        with right:
            if not sim_customers.empty and "wait_time" in sim_customers.columns:
                fig_hist = px.histogram(
                    sim_customers, x="wait_time", color="state", nbins=30,
                    labels={"wait_time": "Waktu tunggu (menit)", "count": "Pelanggan"},
                    title="Distribusi waktu tunggu — replikasi pertama",
                )
                st.plotly_chart(fig_hist, width="stretch")

        fig_mc = px.box(
            sim_mc, y=["avg_wait_time", "utilization_percent", "abandonment_percent"],
            points="all", title="Sebaran hasil simulasi interaktif",
            labels={"value": "Nilai", "variable": "Indikator"},
        )
        st.plotly_chart(fig_mc, width="stretch")

        st.subheader("Data pelanggan replikasi pertama")
        st.dataframe(sim_customers.head(300), width="stretch", hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download ringkasan simulasi CSV", csv_bytes(sim_mc),
            "hasil_simulasi_interaktif.csv", "text/csv",
        )
        d2.download_button(
            "Download data pelanggan CSV", csv_bytes(sim_customers),
            "data_pelanggan_simulasi_interaktif.csv", "text/csv",
        )


# ============================================================
# TAB 4 — UPLOAD DATASET
# ============================================================
with tab_upload:
    st.header("Upload dan Validasi Dataset Eksternal")
    st.write("Gunakan tab ini untuk membaca dataset antrean lain dalam format CSV atau Excel, memetakan kolom, memeriksa kualitas data, menghitung metrik, dan mengunduh hasil yang sudah distandardisasi.")

    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info("Belum ada dataset yang diunggah.")
    else:
        try:
            uploaded_raw = read_uploaded_table(uploaded_file)
        except Exception as error:
            st.error(f"File tidak dapat dibaca: {error}")
            uploaded_raw = pd.DataFrame()

        if not uploaded_raw.empty:
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Jumlah baris", f"{len(uploaded_raw):,}")
            q2.metric("Jumlah kolom", f"{uploaded_raw.shape[1]}")
            q3.metric("Missing value", f"{int(uploaded_raw.isna().sum().sum()):,}")
            q4.metric("Duplikasi", f"{int(uploaded_raw.duplicated().sum()):,}")
            st.dataframe(uploaded_raw.head(100), width="stretch", hide_index=True)

            st.subheader("Pemetaan kolom")
            options = [None] + uploaded_raw.columns.tolist()
            canonical = [
                "customer_id", "scenario", "arrival_time", "start_service_time",
                "departure_time", "wait_time", "service_time", "system_time",
                "state", "patience_time", "num_kiosks", "mean_interarrival_time",
                "mean_service_time", "duration", "seed",
            ]
            mapping: dict[str, str | None] = {}
            columns = st.columns(3)
            for index, target in enumerate(canonical):
                default_index = options.index(target) if target in options else 0
                mapping[target] = columns[index % 3].selectbox(
                    target, options=options, index=default_index,
                    format_func=lambda value: "(Tidak tersedia)" if value is None else str(value),
                    key=f"map_{target}",
                )

            cleaned = apply_column_mapping(uploaded_raw, mapping)
            negative_columns = [
                col for col in ["wait_time", "service_time", "system_time"]
                if col in cleaned.columns and (cleaned[col] < 0).any()
            ]
            if negative_columns:
                st.warning("Ditemukan nilai negatif pada: " + ", ".join(negative_columns))

            metrics = uploaded_metrics(cleaned)
            u1, u2, u3, u4, u5 = st.columns(5)
            u1.metric("Total data", f"{metrics['total']:,}")
            u2.metric("Selesai", f"{metrics['served']:,}")
            u3.metric("Batal", f"{metrics['abandoned']:,}")
            u4.metric("Waktu tunggu", format_number(metrics["avg_wait"]) + " menit")
            u5.metric("Abandonment", format_number(metrics["abandonment"]) + "%")

            if "wait_time" in cleaned.columns:
                plot_df = cleaned.dropna(subset=["wait_time"]).copy()
                if not plot_df.empty:
                    color_col = "state" if "state" in plot_df.columns else None
                    fig_upload_hist = px.histogram(
                        plot_df, x="wait_time", color=color_col, nbins=35,
                        title="Distribusi waktu tunggu dataset upload",
                    )
                    st.plotly_chart(fig_upload_hist, width="stretch")

            if "scenario" in cleaned.columns and "wait_time" in cleaned.columns:
                scenario_stats = (
                    cleaned.groupby("scenario", dropna=False)["wait_time"]
                    .agg(["count", "mean", "median", "max"])
                    .reset_index()
                )
                st.subheader("Ringkasan per skenario")
                st.dataframe(scenario_stats, width="stretch", hide_index=True)

            st.download_button(
                "Download dataset yang sudah distandardisasi",
                csv_bytes(cleaned),
                "dataset_antrean_terstandardisasi.csv",
                "text/csv",
            )


# ============================================================
# TAB 5 — OPTIMASI
# ============================================================
with tab_optimization:
    st.header("Optimasi Jumlah Kiosk")
    mode_optimization = st.radio(
        "Sumber hasil", ["Hasil final notebook", "Hitung konfigurasi ringan"], horizontal=True
    )

    if mode_optimization == "Hasil final notebook":
        optimization = data["Optimasi kiosk"].copy()
    else:
        with st.form("optimization_form"):
            o1, o2, o3, o4 = st.columns(4)
            max_kiosks = o1.number_input("Kiosk maksimum", min_value=2, max_value=8, value=5, step=1)
            opt_iterations = o2.select_slider("Iterasi per kiosk", options=[10, 20, 30, 40, 50], value=30)
            opt_duration = o3.number_input("Durasi", min_value=60, max_value=720, value=480, step=30)
            opt_seed = o4.number_input("Seed", min_value=1, max_value=999_999, value=3030, step=1)
            o5, o6, o7 = st.columns(3)
            opt_arrival = o5.number_input("Interval kedatangan", min_value=0.5, max_value=15.0, value=3.0, step=0.1)
            opt_service = o6.number_input("Waktu pelayanan", min_value=0.5, max_value=15.0, value=4.0, step=0.1)
            opt_patience = o7.number_input("Batas kesabaran", min_value=1.0, max_value=60.0, value=12.0, step=1.0)
            run_opt = st.form_submit_button("Jalankan optimasi ringan")
        if run_opt:
            with st.spinner("Menghitung optimasi ringan..."):
                st.session_state["opt_result"] = run_light_optimization(
                    int(max_kiosks), int(opt_iterations), int(opt_seed), int(opt_duration),
                    float(opt_arrival), float(opt_service), float(opt_patience),
                )
        optimization = st.session_state.get("opt_result", pd.DataFrame())

    if optimization.empty:
        st.info("Belum ada hasil optimasi.")
    else:
        optimization = optimization.sort_values("num_kiosks")
        st.dataframe(optimization.round(3), width="stretch", hide_index=True)
        targets = optimization[
            (optimization["avg_wait_time"] <= 1.0)
            & (optimization["abandonment_percent"] <= 1.0)
            & (optimization["utilization_percent"] <= 85.0)
        ]
        if not targets.empty:
            recommendation = int(targets.iloc[0]["num_kiosks"])
            st.success(f"Rekomendasi minimum berdasarkan target layanan: {recommendation} kiosk.")
        else:
            st.warning("Belum ada konfigurasi yang memenuhi seluruh target layanan.")

        g1, g2 = st.columns(2)
        with g1:
            fig_opt_wait = px.line(
                optimization, x="num_kiosks", y="avg_wait_time", markers=True,
                title="Jumlah kiosk terhadap waktu tunggu",
                labels={"num_kiosks": "Jumlah kiosk", "avg_wait_time": "Waktu tunggu (menit)"},
            )
            fig_opt_wait.add_hline(y=1.0, line_dash="dash", annotation_text="Target 1 menit")
            st.plotly_chart(fig_opt_wait, width="stretch")
        with g2:
            fig_opt_util = px.line(
                optimization, x="num_kiosks", y="utilization_percent", markers=True,
                title="Jumlah kiosk terhadap utilisasi",
                labels={"num_kiosks": "Jumlah kiosk", "utilization_percent": "Utilisasi (%)"},
            )
            fig_opt_util.add_hline(y=85.0, line_dash="dash", annotation_text="Batas 85%")
            st.plotly_chart(fig_opt_util, width="stretch")

        st.download_button(
            "Download hasil optimasi CSV", csv_bytes(optimization),
            "ringkasan_optimasi_dashboard.csv", "text/csv",
        )


# ============================================================
# TAB 6 — SENSITIVITAS
# ============================================================
with tab_sensitivity:
    st.header("Analisis Sensitivitas")
    sensitivity_mode = st.radio(
        "Sumber hasil sensitivitas", ["Hasil final notebook", "Hitung parameter lain"], horizontal=True
    )

    if sensitivity_mode == "Hasil final notebook":
        sensitivity = data["Sensitivitas"].copy()
        sensitivity_parameter = "mean_interarrival_time"
    else:
        with st.form("sensitivity_form"):
            sensitivity_parameter = st.selectbox(
                "Parameter yang diuji",
                ["mean_interarrival_time", "mean_service_time", "patience_time"],
                format_func=lambda value: {
                    "mean_interarrival_time": "Interval kedatangan",
                    "mean_service_time": "Waktu pelayanan",
                    "patience_time": "Batas kesabaran",
                }[value],
            )
            s1, s2, s3, s4 = st.columns(4)
            sens_min = s1.number_input("Nilai minimum", min_value=0.5, max_value=30.0, value=2.0, step=0.5)
            sens_max = s2.number_input("Nilai maksimum", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
            sens_points = s3.number_input("Jumlah titik", min_value=3, max_value=7, value=4, step=1)
            sens_iterations = s4.select_slider("Iterasi per titik", options=[10, 20, 30, 40, 50], value=30)
            s5, s6, s7, s8 = st.columns(4)
            sens_kiosks = s5.number_input("Jumlah kiosk", min_value=1, max_value=8, value=2, step=1)
            sens_arrival = s6.number_input("Interval dasar", min_value=0.5, max_value=15.0, value=3.0, step=0.1)
            sens_service = s7.number_input("Pelayanan dasar", min_value=0.5, max_value=15.0, value=4.0, step=0.1)
            sens_patience = s8.number_input("Kesabaran dasar", min_value=1.0, max_value=60.0, value=12.0, step=1.0)
            run_sens = st.form_submit_button("Jalankan sensitivitas ringan")
        if run_sens:
            if sens_max <= sens_min:
                st.error("Nilai maksimum harus lebih besar daripada nilai minimum.")
            else:
                values = tuple(np.linspace(float(sens_min), float(sens_max), int(sens_points)).round(4))
                with st.spinner("Menghitung sensitivitas..."):
                    st.session_state["sens_result"] = run_light_sensitivity(
                        values, sensitivity_parameter, int(sens_iterations), 4040, 480,
                        int(sens_kiosks), float(sens_arrival), float(sens_service), float(sens_patience),
                    )
        sensitivity = st.session_state.get("sens_result", pd.DataFrame())

    if sensitivity.empty:
        st.info("Belum ada hasil sensitivitas.")
    else:
        st.dataframe(sensitivity.round(3), width="stretch", hide_index=True)
        long_sens = sensitivity.melt(
            id_vars=[sensitivity_parameter],
            value_vars=["avg_wait_time", "avg_queue_length", "utilization_percent", "abandonment_percent"],
            var_name="metric", value_name="value",
        )
        fig_sens = px.line(
            long_sens, x=sensitivity_parameter, y="value", color="metric", markers=True,
            title="Perubahan kinerja terhadap parameter yang diuji",
            labels={sensitivity_parameter: "Nilai parameter", "value": "Nilai indikator", "metric": "Indikator"},
        )
        st.plotly_chart(fig_sens, width="stretch")
        st.download_button(
            "Download hasil sensitivitas CSV", csv_bytes(sensitivity),
            "hasil_sensitivitas_dashboard.csv", "text/csv",
        )


# ============================================================
# TAB 7 — METODOLOGI DAN EKSPOR
# ============================================================
with tab_method:
    st.header("Metodologi, Asumsi, dan Ekspor")
    methodology = pd.DataFrame(
        {
            "Komponen": [
                "Pendekatan", "Agen", "Resource", "Distribusi kedatangan",
                "Distribusi pelayanan", "Disiplin antrean", "Ketidakpastian",
                "Horizon", "Status akhir",
            ],
            "Penerapan": [
                "Agent-Based Modeling dan Discrete Event Simulation",
                "Pelanggan bioskop",
                "Self-service kiosk",
                "Eksponensial",
                "Lognormal",
                "First Come, First Served",
                "Monte Carlo dengan seed berbeda",
                "480 menit; kedatangan berhenti pada akhir horizon dan sistem dikuras",
                "selesai atau batal",
            ],
        }
    )
    st.dataframe(methodology, width="stretch", hide_index=True)

    st.subheader("Asumsi dan batasan")
    st.markdown(
        """
        - Semua kiosk dianggap memiliki kemampuan pelayanan yang sama.
        - Satu kiosk hanya melayani satu pelanggan pada satu waktu.
        - Tidak ada prioritas pelanggan dan kerusakan kiosk.
        - Parameter bersifat sintetis dan belum dikalibrasi menggunakan data bioskop nyata.
        - Hasil digunakan untuk perbandingan kebijakan, bukan prediksi operasional absolut.
        """
    )

    st.subheader("Download data final")
    available_downloads = {
        label: DATA_DIR / filename
        for label, filename in REQUIRED_DATA_FILES.items()
        if (DATA_DIR / filename).exists()
    }
    selected_download = st.selectbox("Pilih file", list(available_downloads))
    selected_path = available_downloads[selected_download]
    st.download_button(
        "Download file terpilih",
        selected_path.read_bytes(),
        selected_path.name,
        "text/csv",
    )

    narrative = (
        "Dashboard menggunakan Agent-Based Modeling dan Discrete Event Simulation untuk "
        "merepresentasikan pelanggan sebagai agen yang datang, menunggu, dilayani, selesai, "
        "atau meninggalkan antrean. Hasil final diperoleh dari 1.000 iterasi Monte Carlo per "
        "skenario. Intervensi preventif memberikan waktu tunggu dan abandonment terendah, "
        "sedangkan beban tinggi menyebabkan utilisasi mendekati kapasitas penuh. Optimasi "
        "menunjukkan bahwa tiga kiosk merupakan kapasitas minimum yang memenuhi target waktu "
        "tunggu maksimal satu menit, abandonment maksimal satu persen, dan utilisasi maksimal "
        "delapan puluh lima persen pada parameter dasar."
    )
    st.subheader("Narasi ringkas siap laporan")
    st.text_area("Narasi", narrative, height=180)
    st.download_button(
        "Download narasi TXT", narrative.encode("utf-8"),
        "narasi_hasil_dashboard.txt", "text/plain",
    )

st.html(
    """
    <div class="footer">
      Dashboard akademik simulasi antrean kiosk bioskop. Hasil final berasal dari notebook Monte Carlo 1.000 iterasi; simulasi interaktif dibatasi untuk menjaga kestabilan server.
    </div>
    """
)
