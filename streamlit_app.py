from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Dashboard Akademik ABM CBT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE DASHBOARD
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --navy: #102a43;
        --blue: #1f6f8b;
        --teal: #2cb1bc;
        --ink: #243b53;
        --muted: #627d98;
        --line: #d9e2ec;
        --soft: #f5f9fb;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--navy);
        letter-spacing: -0.02em;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #102a43 0%,
            #1f6f8b 62%,
            #2cb1bc 100%
        );
        padding: 28px 32px;
        border-radius: 22px;
        color: white;
        box-shadow: 0 16px 36px rgba(16, 42, 67, 0.18);
        margin-bottom: 16px;
    }

    .hero h1 {
        color: white;
        margin: 0.25rem 0 0.45rem;
        font-size: 2.05rem;
    }

    .hero p {
        margin: 0;
        max-width: 1100px;
        line-height: 1.6;
        color: rgba(255, 255, 255, 0.92);
    }

    .badge {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.25);
        margin: 0 6px 6px 0;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 7px 20px rgba(16, 42, 67, 0.06);
        min-height: 115px;
    }

    .m-label {
        font-size: 0.78rem;
        color: var(--muted);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .m-value {
        font-size: 1.55rem;
        color: var(--navy);
        font-weight: 850;
        margin-top: 5px;
        line-height: 1.15;
    }

    .m-note {
        font-size: 0.84rem;
        color: var(--muted);
        margin-top: 5px;
    }

    .formula {
        background: #f0f7fa;
        border-left: 5px solid var(--teal);
        border-radius: 12px;
        padding: 15px 18px;
        font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        color: var(--navy);
        margin-bottom: 16px;
    }

    .notice {
        background: #fff8e6;
        border: 1px solid #f4cf6a;
        border-radius: 14px;
        padding: 13px 16px;
        color: #5c3d00;
        margin-bottom: 14px;
    }

    .info-box {
        background: #f0f7fa;
        border: 1px solid #bee3ea;
        border-radius: 14px;
        padding: 14px 16px;
        color: #102a43;
        margin-bottom: 14px;
    }

    .foot {
        border-top: 1px solid var(--line);
        padding-top: 12px;
        margin-top: 26px;
        color: var(--muted);
        font-size: 0.84rem;
    }

    section[data-testid="stSidebar"] {
        background: #f7fafc;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding-left: 15px;
        padding-right: 15px;
        background: white;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stDownloadButton"] button {
        border-radius: 10px;
        font-weight: 700;
    }

    div[data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA DAN KONFIGURASI MODEL
# ============================================================

SCENARIOS = [
    "Tanpa Intervensi",
    "Reaktif",
    "Preventif",
    "Distorsi Kognitif Tinggi",
]

SCENARIO_INFO = {
    "Tanpa Intervensi":
        "Skenario baseline. Agen menerima stressor tanpa CBT "
        "atau latihan koping.",

    "Reaktif":
        "CBT diberikan ketika Anxiety Level telah melampaui "
        "ambang kecemasan tinggi.",

    "Preventif":
        "CBT dilakukan secara rutin sejak awal dan secara "
        "bertahap memperkuat resilience agen.",

    "Distorsi Kognitif Tinggi":
        "Agen memiliki nilai Cognitive Distortion lebih tinggi "
        "untuk menguji apakah CBT standar masih memadai.",
}

STATE_TABLE = pd.DataFrame(
    {
        "State": [
            "Tenang",
            "Cemas Ringan",
            "Cemas Tinggi",
            "Panik",
            "Pulih",
        ],
        "Aturan": [
            "A < 0.30",
            "0.30 ≤ A < 0.60",
            "0.60 ≤ A < 0.80",
            "A ≥ 0.80",
            "Pernah panik dan A < 0.30",
        ],
        "Interpretasi": [
            "Agen relatif stabil.",
            "Tekanan mulai muncul tetapi masih terkendali.",
            "Agen membutuhkan dukungan atau strategi koping.",
            "Agen mencapai ambang risiko tinggi pada model.",
            "Agen kembali stabil setelah episode panik.",
        ],
    }
)

VARIABLE_TABLE = pd.DataFrame(
    {
        "Simbol": ["A", "S", "D", "R", "P"],
        "Variabel": [
            "Anxiety Level",
            "Stressor",
            "Cognitive Distortion",
            "Resilience",
            "CBT Protocol",
        ],
        "Rentang": [
            "0.00–1.00",
            "0.00–0.30",
            "0.05–1.50",
            "0.01–0.09",
            "0.00–1.00",
        ],
        "Peran": [
            "Kondisi kecemasan agen pada waktu t.",
            "Tekanan eksternal acak dan terjadwal.",
            "Pengali yang memperbesar dampak stressor.",
            "Kemampuan pemulihan alami agen.",
            "Kekuatan intervensi CBT atau latihan koping.",
        ],
    }
)

CHECKLIST_TABLE = pd.DataFrame(
    {
        "Ketentuan Tugas": [
            "State chart kondisi agen",
            "Variabel numerik A, S, D, R, dan P",
            "Agen bergerak dan berinteraksi",
            "Stressor acak dan terjadwal",
            "Intervensi CBT",
            "Skenario What-If",
            "Monte Carlo 1000 iterasi",
            "Pengolahan hasil simulasi",
            "Dashboard dan ekspor data",
            "Uraian luaran HKI",
        ],
        "Implementasi": [
            "Tersedia lima state agen.",
            "Seluruh variabel dapat diatur melalui sidebar.",
            "Agen bergerak pada grid dan memiliki kontak sosial.",
            "Tersedia peluang stressor dan tick stressor terjadwal.",
            "Tersedia CBT reaktif dan preventif.",
            "Tersedia empat skenario eksperimen.",
            "Iterasi dapat diatur sampai 1000.",
            "Tersedia rata-rata A, panik, waktu pulih, dan ranking.",
            "Tersedia grafik, CSV, snapshot, dan narasi.",
            "Tersedia uraian program komputer untuk persiapan HKI.",
        ],
    }
)


# ============================================================
# FUNGSI TAMPILAN
# ============================================================

def metric_card(
    label: str,
    value: str,
    note: str = "",
) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="m-label">{label}</div>
            <div class="m-value">{value}</div>
            <div class="m-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_ticks(
    text: str,
    max_tick: int,
) -> Tuple[int, ...]:
    values = []

    for item in text.replace(";", ",").split(","):
        try:
            tick = int(item.strip())

            if 0 <= tick < max_tick:
                values.append(tick)

        except ValueError:
            continue

    return tuple(sorted(set(values)))


def state_labels(
    anxiety: np.ndarray,
    had_panic: np.ndarray,
) -> np.ndarray:
    labels = np.where(
        anxiety < 0.30,
        "Tenang",
        np.where(
            anxiety < 0.60,
            "Cemas Ringan",
            np.where(
                anxiety < 0.80,
                "Cemas Tinggi",
                "Panik",
            ),
        ),
    )

    return np.where(
        (anxiety < 0.30) & had_panic,
        "Pulih",
        labels,
    )


def stability_score(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    def inverse_minmax(
        series: pd.Series,
    ) -> pd.Series:
        span = float(series.max() - series.min())

        if span < 1e-12:
            return pd.Series(
                np.ones(len(series)),
                index=series.index,
            )

        return 1.0 - (
            (series - series.min()) / span
        )

    result["Skor Stabilitas"] = 100 * (
        0.38 * inverse_minmax(result["Rata-rata A"])
        + 0.30 * inverse_minmax(result["Peak % Panik"])
        + 0.20 * inverse_minmax(result["Waktu Pulih"])
        + 0.12 * inverse_minmax(result["Final Mean A"])
    )

    return result


# ============================================================
# FUNGSI SIMULASI
# ============================================================

@st.cache_data(show_spinner=False)
def simulate_scenario(
    scenario: str,
    iterations: int,
    agents: int,
    steps: int,
    grid_size: int,
    seed: int,
    stress_prob: float,
    stress_min: float,
    stress_max: float,
    scheduled_ticks: Tuple[int, ...],
    scheduled_boost: float,
    mean_r: float,
    mean_d: float,
    high_d: float,
    cbt_strength: float,
    reactive_threshold: float,
    preventive_base: float,
    preventive_boost: float,
    social_contagion: float,
    social_support: float,
    noise_sd: float,
) -> Tuple[
    Dict[str, float],
    pd.DataFrame,
    pd.DataFrame,
]:
    rng = np.random.default_rng(seed)

    anxiety = np.clip(
        rng.normal(
            0.25,
            0.08,
            (iterations, agents),
        ),
        0.05,
        0.55,
    )

    resilience = np.clip(
        rng.normal(
            mean_r,
            0.008,
            (iterations, agents),
        ),
        0.01,
        0.09,
    )

    distortion_center = (
        high_d
        if scenario == "Distorsi Kognitif Tinggi"
        else mean_d
    )

    distortion = np.clip(
        rng.normal(
            distortion_center,
            0.12,
            (iterations, agents),
        ),
        0.05,
        1.50,
    )

    cbt_response = np.clip(
        rng.normal(
            1.0,
            0.15,
            (iterations, agents),
        ),
        0.70,
        1.30,
    )

    x_position = rng.integers(
        0,
        grid_size,
        (iterations, agents),
    )

    y_position = rng.integers(
        0,
        grid_size,
        (iterations, agents),
    )

    episode_start = np.full(
        (iterations, agents),
        -1,
        dtype=np.int16,
    )

    had_panic = np.zeros(
        (iterations, agents),
        dtype=bool,
    )

    recovery_sum = np.zeros(iterations)
    recovery_count = np.zeros(iterations)
    panic_episode_count = np.zeros(iterations)

    rows = []
    scheduled = set(scheduled_ticks)

    iteration_index = np.arange(
        iterations
    )[:, None, None]

    for tick in range(steps):

        # ----------------------------------------------------
        # GERAKAN AGEN
        # ----------------------------------------------------

        x_position = (
            x_position
            + rng.integers(
                -1,
                2,
                (iterations, agents),
            )
        ) % grid_size

        y_position = (
            y_position
            + rng.integers(
                -1,
                2,
                (iterations, agents),
            )
        ) % grid_size

        # ----------------------------------------------------
        # INTERAKSI SOSIAL
        # ----------------------------------------------------

        contact_index = rng.integers(
            0,
            agents,
            size=(iterations, agents, 3),
        )

        neighbor_anxiety = anxiety[
            iteration_index,
            contact_index,
        ].mean(axis=2)

        contagion_effect = (
            social_contagion
            * np.maximum(
                neighbor_anxiety - 0.65,
                0,
            )
        )

        support_effect = (
            social_support
            * np.maximum(
                0.35 - neighbor_anxiety,
                0,
            )
        )

        # ----------------------------------------------------
        # STRESSOR LINGKUNGAN
        # ----------------------------------------------------

        has_stressor = (
            rng.random(iterations)
            < stress_prob
        )

        stressor = np.where(
            has_stressor,
            rng.uniform(
                stress_min,
                stress_max,
                iterations,
            ),
            0.0,
        )

        if tick in scheduled:
            stressor = (
                stressor
                + scheduled_boost
            )

        # ----------------------------------------------------
        # PROTOKOL CBT
        # ----------------------------------------------------

        if scenario == "Tanpa Intervensi":
            protocol = np.zeros_like(
                anxiety
            )

        elif scenario == "Reaktif":
            protocol = np.where(
                anxiety >= reactive_threshold,
                0.85,
                0.0,
            )

        elif scenario == "Preventif":
            morning_boost = (
                preventive_boost
                if tick % 24 in (0, 1, 2)
                else 0.0
            )

            protocol = np.full_like(
                anxiety,
                preventive_base
                + morning_boost,
            )

            resilience = np.clip(
                resilience + 0.00012,
                0.01,
                0.09,
            )

        else:
            protocol = np.full_like(
                anxiety,
                0.45,
            )

        # ----------------------------------------------------
        # PERSAMAAN TRANSISI
        # ----------------------------------------------------

        stress_component = (
            stressor[:, None]
            * (1 + distortion)
        )

        natural_recovery = (
            resilience
            * (1 + 0.70 * protocol)
        )

        cbt_reduction = (
            cbt_strength
            * protocol
            * cbt_response
        )

        noise = rng.normal(
            0,
            noise_sd,
            (iterations, agents),
        )

        anxiety = np.clip(
            anxiety
            + stress_component
            + contagion_effect
            - support_effect
            - natural_recovery
            - cbt_reduction
            + noise,
            0,
            1,
        )

        # ----------------------------------------------------
        # DETEKSI PANIK DAN PEMULIHAN
        # ----------------------------------------------------

        newly_panic = (
            (anxiety >= 0.80)
            & (episode_start == -1)
        )

        episode_start[newly_panic] = tick
        had_panic |= newly_panic

        panic_episode_count += (
            newly_panic.sum(axis=1)
        )

        recovered = (
            (anxiety < 0.30)
            & (episode_start != -1)
        )

        if recovered.any():
            duration = (
                tick - episode_start
            ).astype(float)

            recovery_sum += (
                duration * recovered
            ).sum(axis=1)

            recovery_count += (
                recovered.sum(axis=1)
            )

            episode_start[recovered] = -1

        # ----------------------------------------------------
        # PENCATATAN HASIL PER TICK
        # ----------------------------------------------------

        mean_per_iteration = (
            anxiety.mean(axis=1)
        )

        panic_per_iteration = (
            (anxiety >= 0.80)
            .mean(axis=1)
            * 100
        )

        rows.append(
            {
                "time": tick,
                "scenario": scenario,

                "mean_A":
                    float(
                        mean_per_iteration.mean()
                    ),

                "q25_A":
                    float(
                        np.quantile(
                            mean_per_iteration,
                            0.25,
                        )
                    ),

                "q75_A":
                    float(
                        np.quantile(
                            mean_per_iteration,
                            0.75,
                        )
                    ),

                "pct_tenang":
                    float(
                        (anxiety < 0.30)
                        .mean()
                        * 100
                    ),

                "pct_cemas_ringan":
                    float(
                        (
                            (anxiety >= 0.30)
                            & (anxiety < 0.60)
                        )
                        .mean()
                        * 100
                    ),

                "pct_cemas_tinggi":
                    float(
                        (
                            (anxiety >= 0.60)
                            & (anxiety < 0.80)
                        )
                        .mean()
                        * 100
                    ),

                "pct_panik":
                    float(
                        panic_per_iteration.mean()
                    ),

                "stressor_S":
                    float(
                        stressor.mean()
                    ),

                "mean_D":
                    float(
                        distortion.mean()
                    ),

                "mean_R":
                    float(
                        resilience.mean()
                    ),

                "mean_P":
                    float(
                        protocol.mean()
                    ),

                "std_A":
                    float(
                        anxiety.std()
                    ),
            }
        )

    # --------------------------------------------------------
    # EPISODE YANG BELUM PULIH
    # --------------------------------------------------------

    unresolved = (
        episode_start != -1
    )

    if unresolved.any():
        duration = (
            steps - episode_start
        ).astype(float)

        recovery_sum += (
            duration * unresolved
        ).sum(axis=1)

        recovery_count += (
            unresolved.sum(axis=1)
        )

    average_recovery = np.divide(
        recovery_sum,
        recovery_count,
        out=np.zeros_like(recovery_sum),
        where=recovery_count > 0,
    )

    history = pd.DataFrame(rows)

    first_risk = history.loc[
        history["pct_panik"] >= 5,
        "time",
    ]

    summary = {
        "Skenario":
            scenario,

        "Final Mean A":
            float(
                history.iloc[-1]["mean_A"]
            ),

        "Rata-rata A":
            float(
                history["mean_A"].mean()
            ),

        "Peak Mean A":
            float(
                history["mean_A"].max()
            ),

        "Final % Panik":
            float(
                history.iloc[-1]["pct_panik"]
            ),

        "Rata-rata % Panik":
            float(
                history["pct_panik"].mean()
            ),

        "Peak % Panik":
            float(
                history["pct_panik"].max()
            ),

        "Waktu Pulih":
            float(
                average_recovery.mean()
            ),

        "Episode Panik":
            float(
                panic_episode_count.mean()
            ),

        "Episode Pulih":
            float(
                recovery_count.mean()
            ),

        "Tick Risiko Awal":
            int(first_risk.iloc[0])
            if len(first_risk)
            else -1,

        "Rata-rata P":
            float(
                history["mean_P"].mean()
            ),

        "Rata-rata D":
            float(
                history["mean_D"].mean()
            ),

        "Rata-rata R":
            float(
                history["mean_R"].mean()
            ),
    }

    sample = pd.DataFrame(
        {
            "x":
                x_position[0].astype(int),

            "y":
                y_position[0].astype(int),

            "A":
                anxiety[0],

            "R":
                resilience[0],

            "D":
                distortion[0],

            "state":
                state_labels(
                    anxiety[0],
                    had_panic[0],
                ),
        }
    )

    return (
        summary,
        history,
        sample,
    )


@st.cache_data(show_spinner=False)
def run_experiment(
    selected: Tuple[str, ...],
    iterations: int,
    params: Dict,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    summaries = []
    histories = []
    samples = []

    for index, scenario in enumerate(
        selected
    ):
        summary, history, sample = (
            simulate_scenario(
                scenario=scenario,
                iterations=iterations,
                seed=(
                    int(params["seed"])
                    + index * 97
                ),
                **{
                    key: value
                    for key, value
                    in params.items()
                    if key != "seed"
                },
            )
        )

        summaries.append(summary)
        histories.append(history)

        sample.insert(
            0,
            "scenario",
            scenario,
        )

        samples.append(sample)

    return (
        pd.DataFrame(summaries),

        pd.concat(
            histories,
            ignore_index=True,
        ),

        pd.concat(
            samples,
            ignore_index=True,
        ),
    )


@st.cache_data(show_spinner=False)
def run_sensitivity(
    selected: Tuple[str, ...],
    parameter: str,
    values: Tuple[float, ...],
    params: Dict,
    iterations: int,
) -> pd.DataFrame:
    rows = []

    mapping = {
        "Kekuatan CBT":
            "cbt_strength",

        "Peluang Stressor":
            "stress_prob",

        "Distorsi Kognitif":
            "mean_d",

        "Resilience":
            "mean_r",
    }

    for value in values:
        varied_params = dict(params)

        varied_params[
            mapping[parameter]
        ] = float(value)

        summary, _, _ = run_experiment(
            selected,
            iterations,
            varied_params,
        )

        for _, row in summary.iterrows():
            rows.append(
                {
                    "Parameter":
                        parameter,

                    "Nilai":
                        float(value),

                    "Skenario":
                        row["Skenario"],

                    "Rata-rata A":
                        row["Rata-rata A"],

                    "Peak % Panik":
                        row["Peak % Panik"],

                    "Waktu Pulih":
                        row["Waktu Pulih"],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <span class="badge">
            Tugas Besar Pemodelan & Simulasi
        </span>

        <span class="badge">
            Agent-Based Modeling
        </span>

        <span class="badge">
            CBT Protocol
        </span>

        <span class="badge">
            Monte Carlo
        </span>

        <h1>
            Dashboard Akademik Simulasi Kecemasan
            ABM + Intervensi CBT
        </h1>

        <p>
            Simulasi membandingkan respons agen heterogen
            terhadap stressor, distorsi kognitif,
            resilience, interaksi sosial, serta
            intervensi CBT reaktif dan preventif.
            Dashboard menyediakan analisis hasil,
            ranking skenario, sensitivitas parameter,
            ekspor data, dan narasi untuk laporan
            maupun dokumen HKI.
        </p>
    </div>

    <div class="notice">
        <b>Catatan etis:</b>
        aplikasi ini adalah model edukatif,
        bukan alat diagnosis klinis dan bukan
        pengganti psikolog atau psikiater.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("Panel Kontrol")

    st.caption(
        "Atur parameter simulasi dan pilih "
        "skenario yang akan dibandingkan."
    )

    preset = st.radio(
        "Mode eksekusi",
        [
            "Demo cepat",
            "Akademik",
            "Final 1000",
        ],
        index=1,
    )

    defaults = {
        "Demo cepat":
            (80, 50, 90),

        "Akademik":
            (250, 70, 120),

        "Final 1000":
            (1000, 80, 150),
    }

    (
        default_iterations,
        default_agents,
        default_steps,
    ) = defaults[preset]

    st.divider()

    st.subheader("Skenario")

    selected = st.multiselect(
        "Pilih skenario",
        SCENARIOS,
        default=SCENARIOS,
    )

    st.divider()

    st.subheader("Ukuran Simulasi")

    iterations = st.slider(
        "Iterasi Monte Carlo",
        min_value=50,
        max_value=1000,
        value=default_iterations,
        step=50,
    )

    agents = st.slider(
        "Jumlah agen",
        min_value=20,
        max_value=200,
        value=default_agents,
        step=10,
    )

    steps = st.slider(
        "Jumlah tick",
        min_value=50,
        max_value=300,
        value=default_steps,
        step=10,
    )

    grid_size = st.slider(
        "Ukuran grid",
        min_value=10,
        max_value=40,
        value=20,
        step=2,
    )

    seed = st.number_input(
        "Seed",
        min_value=1,
        max_value=999999,
        value=2026,
    )

    st.divider()

    st.subheader("Stressor")

    stress_prob = st.slider(
        "Peluang stressor",
        min_value=0.05,
        max_value=0.80,
        value=0.35,
        step=0.01,
    )

    stress_min = st.slider(
        "Stressor minimum",
        min_value=0.00,
        max_value=0.15,
        value=0.03,
        step=0.005,
        format="%.3f",
    )

    stress_max = st.slider(
        "Stressor maksimum",
        min_value=0.03,
        max_value=0.30,
        value=0.11,
        step=0.005,
        format="%.3f",
    )

    tick_text = st.text_input(
        "Tick stressor terjadwal",
        "30, 60, 90",
    )

    scheduled_boost = st.slider(
        "Boost stressor terjadwal",
        min_value=0.00,
        max_value=0.25,
        value=0.10,
        step=0.005,
        format="%.3f",
    )

    st.divider()

    st.subheader(
        "Atribut Psikologis"
    )

    mean_r = st.slider(
        "Rata-rata Resilience (R)",
        min_value=0.010,
        max_value=0.080,
        value=0.025,
        step=0.001,
        format="%.3f",
    )

    mean_d = st.slider(
        "Rata-rata Distortion (D)",
        min_value=0.05,
        max_value=1.20,
        value=0.45,
        step=0.01,
    )

    high_d = st.slider(
        "D untuk skenario tinggi",
        min_value=0.60,
        max_value=1.50,
        value=0.95,
        step=0.01,
    )

    st.divider()

    st.subheader("Intervensi CBT")

    cbt_strength = st.slider(
        "Kekuatan CBT",
        min_value=0.00,
        max_value=0.12,
        value=0.04,
        step=0.005,
    )

    reactive_threshold = st.slider(
        "Ambang reaktif",
        min_value=0.60,
        max_value=0.95,
        value=0.80,
        step=0.01,
    )

    preventive_base = st.slider(
        "CBT preventif rutin",
        min_value=0.00,
        max_value=0.80,
        value=0.30,
        step=0.01,
    )

    preventive_boost = st.slider(
        "Boost latihan pagi",
        min_value=0.00,
        max_value=0.80,
        value=0.30,
        step=0.01,
    )

    with st.expander(
        "Parameter sosial dan noise"
    ):
        social_contagion = st.slider(
            "Social contagion",
            min_value=0.000,
            max_value=0.060,
            value=0.020,
            step=0.002,
            format="%.3f",
        )

        social_support = st.slider(
            "Social support",
            min_value=0.000,
            max_value=0.050,
            value=0.012,
            step=0.002,
            format="%.3f",
        )

        noise_sd = st.slider(
            "Noise",
            min_value=0.000,
            max_value=0.030,
            value=0.008,
            step=0.001,
            format="%.3f",
        )

    st.info(
        "Gunakan mode Final 1000 "
        "untuk hasil akhir laporan."
    )


# ============================================================
# VALIDASI INPUT
# ============================================================

if not selected:
    st.warning(
        "Pilih minimal satu skenario."
    )
    st.stop()

if stress_max < stress_min:
    st.error(
        "Stressor maksimum harus lebih besar "
        "atau sama dengan stressor minimum."
    )
    st.stop()


# ============================================================
# PARAMETER SIMULASI
# ============================================================

scheduled_ticks = parse_ticks(
    tick_text,
    steps,
)

params = {
    "agents":
        int(agents),

    "steps":
        int(steps),

    "grid_size":
        int(grid_size),

    "seed":
        int(seed),

    "stress_prob":
        float(stress_prob),

    "stress_min":
        float(stress_min),

    "stress_max":
        float(stress_max),

    "scheduled_ticks":
        tuple(scheduled_ticks),

    "scheduled_boost":
        float(scheduled_boost),

    "mean_r":
        float(mean_r),

    "mean_d":
        float(mean_d),

    "high_d":
        float(high_d),

    "cbt_strength":
        float(cbt_strength),

    "reactive_threshold":
        float(reactive_threshold),

    "preventive_base":
        float(preventive_base),

    "preventive_boost":
        float(preventive_boost),

    "social_contagion":
        float(social_contagion),

    "social_support":
        float(social_support),

    "noise_sd":
        float(noise_sd),
}


# ============================================================
# MENJALANKAN SIMULASI
# ============================================================

with st.spinner(
    "Menjalankan simulasi ABM dan Monte Carlo..."
):
    summary, history, sample = run_experiment(
        tuple(selected),
        int(iterations),
        params,
    )

ranked = (
    stability_score(summary)
    .sort_values(
        "Skor Stabilitas",
        ascending=False,
    )
    .reset_index(drop=True)
)

best = ranked.iloc[0]


# ============================================================
# KARTU METRIK
# ============================================================

metric_columns = st.columns(4)

with metric_columns[0]:
    metric_card(
        "Skenario terbaik",
        str(best["Skenario"]),
        "Berdasarkan skor stabilitas gabungan",
    )

with metric_columns[1]:
    metric_card(
        "Rata-rata A terbaik",
        f"{best['Rata-rata A']:.3f}",
        "Semakin rendah semakin stabil",
    )

with metric_columns[2]:
    metric_card(
        "Peak panik terendah",
        f"{summary['Peak % Panik'].min():.2f}%",
        "Persentase maksimum agen panik",
    )

with metric_columns[3]:
    metric_card(
        "Konfigurasi",
        f"{iterations} iterasi",
        f"{agents} agen · {steps} tick",
    )


# ============================================================
# TAB DASHBOARD
# ============================================================

(
    overview_tab,
    result_tab,
    dynamic_tab,
    detail_tab,
    sensitivity_tab,
    export_tab,
) = st.tabs(
    [
        "Overview Akademik",
        "Ringkasan & Ranking",
        "Dinamika",
        "Analisis Skenario",
        "Sensitivitas",
        "Export & HKI",
    ]
)


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with overview_tab:
    left_column, right_column = st.columns(
        [1.15, 1]
    )

    with left_column:
        st.subheader(
            "Tujuan dan Rumus Model"
        )

        st.write(
            "Model mengevaluasi bagaimana kecemasan "
            "agen berubah akibat stressor dan bagaimana "
            "CBT reaktif maupun preventif memengaruhi "
            "stabilitas populasi."
        )

        st.markdown(
            """
            <div class="formula">
            A(t+1) = clip[
                A(t) + S(t)(1+D)
                + SocialEffect
                − R(1+0.70P)
                − CBT(P)
                + ε,
                0,
                1
            ]
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            VARIABLE_TABLE,
            use_container_width=True,
            hide_index=True,
        )

    with right_column:
        st.subheader("State Chart")

        st.graphviz_chart(
            """
            digraph {
                rankdir=LR;

                node [
                    shape=box,
                    style="rounded,filled",
                    fillcolor="#F0F7FA",
                    color="#1F6F8B"
                ];

                Tenang -> "Cemas Ringan";
                "Cemas Ringan" -> "Cemas Tinggi";
                "Cemas Tinggi" -> Panik;
                Panik -> Pulih;
                Pulih -> Tenang;
                "Cemas Tinggi" -> "Cemas Ringan";
                "Cemas Ringan" -> Tenang;
            }
            """,
            use_container_width=True,
        )

        st.dataframe(
            STATE_TABLE,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(
        "Checklist Kesesuaian Tugas"
    )

    st.dataframe(
        CHECKLIST_TABLE,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 2: RINGKASAN DAN RANKING
# ============================================================

with result_tab:
    ranking_view = ranked.copy()

    ranking_view.insert(
        0,
        "Peringkat",
        np.arange(
            1,
            len(ranking_view) + 1,
        ),
    )

    st.subheader(
        "Tabel Hasil dan Ranking Skenario"
    )

    st.dataframe(
        ranking_view.round(3),
        use_container_width=True,
        hide_index=True,
    )

    chart_column_1, chart_column_2 = st.columns(
        2
    )

    with chart_column_1:
        score_chart = px.bar(
            ranked,
            x="Skenario",
            y="Skor Stabilitas",
            text="Skor Stabilitas",
            title="Skor Stabilitas Skenario",
        )

        score_chart.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )

        score_chart.update_layout(
            xaxis_title="Skenario",
            yaxis_title="Skor 0–100",
            height=430,
        )

        st.plotly_chart(
            score_chart,
            use_container_width=True,
        )

    with chart_column_2:
        tradeoff_chart = px.scatter(
            ranked,
            x="Rata-rata A",
            y="Peak % Panik",
            size="Waktu Pulih",
            text="Skenario",
            hover_name="Skenario",
            title="Trade-off Anxiety dan Panik",
        )

        tradeoff_chart.update_traces(
            textposition="top center"
        )

        tradeoff_chart.update_layout(
            height=430
        )

        st.plotly_chart(
            tradeoff_chart,
            use_container_width=True,
        )

    st.subheader(
        "Interpretasi Setiap Skenario"
    )

    for scenario in ranked["Skenario"]:
        row = ranked[
            ranked["Skenario"] == scenario
        ].iloc[0]

        with st.expander(
            (
                f"{scenario} · "
                f"skor {row['Skor Stabilitas']:.1f}"
            ),
            expanded=(
                scenario
                == best["Skenario"]
            ),
        ):
            st.write(
                SCENARIO_INFO[scenario]
            )

            st.write(
                f"Rata-rata A "
                f"**{row['Rata-rata A']:.3f}**, "
                f"peak panik "
                f"**{row['Peak % Panik']:.2f}%**, "
                f"waktu pulih "
                f"**{row['Waktu Pulih']:.2f} tick**."
            )


# ============================================================
# TAB 3: DINAMIKA
# ============================================================

with dynamic_tab:
    show_band = st.toggle(
        "Tampilkan pita kuantil 25–75",
        value=True,
    )

    anxiety_chart = go.Figure()

    for scenario in selected:
        scenario_history = history[
            history["scenario"] == scenario
        ]

        if show_band:
            anxiety_chart.add_trace(
                go.Scatter(
                    x=scenario_history["time"],
                    y=scenario_history["q75_A"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            anxiety_chart.add_trace(
                go.Scatter(
                    x=scenario_history["time"],
                    y=scenario_history["q25_A"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    name=f"IQR {scenario}",
                    opacity=0.16,
                    hoverinfo="skip",
                )
            )

        anxiety_chart.add_trace(
            go.Scatter(
                x=scenario_history["time"],
                y=scenario_history["mean_A"],
                mode="lines",
                name=scenario,
            )
        )

    anxiety_chart.update_layout(
        title=(
            "Dinamika Rata-rata "
            "Anxiety Level"
        ),
        xaxis_title="Tick",
        yaxis_title="Mean A",
        height=500,
    )

    st.plotly_chart(
        anxiety_chart,
        use_container_width=True,
    )

    left_chart, right_chart = st.columns(
        2
    )

    with left_chart:
        panic_chart = px.line(
            history,
            x="time",
            y="pct_panik",
            color="scenario",
            title="Persentase Agen Panik",
        )

        panic_chart.update_layout(
            xaxis_title="Tick",
            yaxis_title="Agen Panik (%)",
            height=420,
        )

        st.plotly_chart(
            panic_chart,
            use_container_width=True,
        )

    with right_chart:
        component_data = history.melt(
            id_vars=[
                "time",
                "scenario",
            ],
            value_vars=[
                "stressor_S",
                "mean_P",
            ],
            var_name="Komponen",
            value_name="Nilai",
        )

        component_chart = px.line(
            component_data,
            x="time",
            y="Nilai",
            color="scenario",
            line_dash="Komponen",
            title=(
                "Stressor dan "
                "CBT Rata-rata"
            ),
        )

        component_chart.update_layout(
            xaxis_title="Tick",
            yaxis_title="Nilai",
            height=420,
        )

        st.plotly_chart(
            component_chart,
            use_container_width=True,
        )

    final_state = (
        history
        .groupby(
            "scenario",
            as_index=False,
        )
        .tail(1)
        [
            [
                "scenario",
                "pct_tenang",
                "pct_cemas_ringan",
                "pct_cemas_tinggi",
                "pct_panik",
            ]
        ]
    )

    final_state_long = final_state.melt(
        id_vars="scenario",
        var_name="State",
        value_name="Persentase",
    )

    final_state_long["State"] = (
        final_state_long["State"]
        .replace(
            {
                "pct_tenang":
                    "Tenang",

                "pct_cemas_ringan":
                    "Cemas Ringan",

                "pct_cemas_tinggi":
                    "Cemas Tinggi",

                "pct_panik":
                    "Panik",
            }
        )
    )

    final_state_chart = px.bar(
        final_state_long,
        x="scenario",
        y="Persentase",
        color="State",
        barmode="group",
        title="Komposisi State Akhir",
    )

    final_state_chart.update_layout(
        xaxis_title="Skenario",
        yaxis_title="Persentase Agen (%)",
        height=460,
    )

    st.plotly_chart(
        final_state_chart,
        use_container_width=True,
    )


# ============================================================
# TAB 4: ANALISIS SKENARIO
# ============================================================

with detail_tab:
    chosen = st.selectbox(
        "Pilih skenario",
        selected,
    )

    chosen_history = history[
        history["scenario"] == chosen
    ]

    chosen_sample = sample[
        sample["scenario"] == chosen
    ]

    chosen_row = ranked[
        ranked["Skenario"] == chosen
    ].iloc[0]

    st.markdown(
        f"""
        <div class="info-box">
            <b>{chosen}</b><br>
            {SCENARIO_INFO[chosen]}
            <br><br>
            Rata-rata A:
            <b>{chosen_row['Rata-rata A']:.3f}</b><br>
            Peak panik:
            <b>{chosen_row['Peak % Panik']:.2f}%</b><br>
            Waktu pulih:
            <b>{chosen_row['Waktu Pulih']:.2f} tick</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scenario_column, grid_column = st.columns(
        [1.1, 0.9]
    )

    with scenario_column:
        state_history = chosen_history.melt(
            id_vars=[
                "time",
                "scenario",
            ],
            value_vars=[
                "pct_tenang",
                "pct_cemas_ringan",
                "pct_cemas_tinggi",
                "pct_panik",
            ],
            var_name="State",
            value_name="Persentase",
        )

        state_history["State"] = (
            state_history["State"]
            .replace(
                {
                    "pct_tenang":
                        "Tenang",

                    "pct_cemas_ringan":
                        "Cemas Ringan",

                    "pct_cemas_tinggi":
                        "Cemas Tinggi",

                    "pct_panik":
                        "Panik",
                }
            )
        )

        state_area_chart = px.area(
            state_history,
            x="time",
            y="Persentase",
            color="State",
            title=(
                "Perubahan Komposisi State"
            ),
        )

        state_area_chart.update_layout(
            xaxis_title="Tick",
            yaxis_title="Persentase Agen (%)",
            height=460,
        )

        st.plotly_chart(
            state_area_chart,
            use_container_width=True,
        )

    with grid_column:
        grid_frame = (
            chosen_sample
            .groupby(
                ["y", "x"],
                as_index=False,
            )["A"]
            .mean()
        )

        grid = np.full(
            (grid_size, grid_size),
            np.nan,
        )

        for _, row in grid_frame.iterrows():
            grid[
                int(row["y"]),
                int(row["x"]),
            ] = row["A"]

        grid_chart = px.imshow(
            grid,
            origin="lower",
            aspect="auto",
            color_continuous_scale=(
                "RdYlBu_r"
            ),
            title="Snapshot Grid Agen",
        )

        grid_chart.update_layout(
            height=460,
            xaxis_title="Posisi X",
            yaxis_title="Posisi Y",
            coloraxis_colorbar_title="A",
        )

        st.plotly_chart(
            grid_chart,
            use_container_width=True,
        )

    histogram_column_1, histogram_column_2, histogram_column_3 = (
        st.columns(3)
    )

    with histogram_column_1:
        anxiety_histogram = px.histogram(
            chosen_sample,
            x="A",
            color="state",
            nbins=24,
            title="Distribusi Anxiety Level",
        )

        anxiety_histogram.update_layout(
            height=350,
            xaxis_title="A",
        )

        st.plotly_chart(
            anxiety_histogram,
            use_container_width=True,
        )

    with histogram_column_2:
        resilience_histogram = px.histogram(
            chosen_sample,
            x="R",
            nbins=20,
            title="Distribusi Resilience",
        )

        resilience_histogram.update_layout(
            height=350,
            xaxis_title="R",
        )

        st.plotly_chart(
            resilience_histogram,
            use_container_width=True,
        )

    with histogram_column_3:
        distortion_histogram = px.histogram(
            chosen_sample,
            x="D",
            nbins=20,
            title=(
                "Distribusi Cognitive Distortion"
            ),
        )

        distortion_histogram.update_layout(
            height=350,
            xaxis_title="D",
        )

        st.plotly_chart(
            distortion_histogram,
            use_container_width=True,
        )


# ============================================================
# TAB 5: SENSITIVITAS
# ============================================================

with sensitivity_tab:
    st.subheader(
        "Analisis Sensitivitas Parameter"
    )

    st.write(
        "Analisis sensitivitas menguji perubahan hasil "
        "ketika satu parameter diubah, sedangkan parameter "
        "lain dipertahankan."
    )

    sensitivity_column_1, sensitivity_column_2, sensitivity_column_3 = (
        st.columns(3)
    )

    with sensitivity_column_1:
        parameter = st.selectbox(
            "Parameter",
            [
                "Kekuatan CBT",
                "Peluang Stressor",
                "Distorsi Kognitif",
                "Resilience",
            ],
        )

    with sensitivity_column_2:
        sensitivity_iterations = st.slider(
            "Iterasi sensitivitas",
            min_value=40,
            max_value=250,
            value=80,
            step=20,
        )

    with sensitivity_column_3:
        run_button = st.button(
            "Jalankan sensitivitas",
            type="primary",
            use_container_width=True,
        )

    value_map = {
        "Kekuatan CBT":
            tuple(
                np.round(
                    np.linspace(
                        0.0,
                        0.12,
                        7,
                    ),
                    3,
                )
            ),

        "Peluang Stressor":
            tuple(
                np.round(
                    np.linspace(
                        0.10,
                        0.70,
                        7,
                    ),
                    2,
                )
            ),

        "Distorsi Kognitif":
            tuple(
                np.round(
                    np.linspace(
                        0.15,
                        1.10,
                        7,
                    ),
                    2,
                )
            ),

        "Resilience":
            tuple(
                np.round(
                    np.linspace(
                        0.01,
                        0.07,
                        7,
                    ),
                    3,
                )
            ),
    }

    if run_button:
        with st.spinner(
            "Menghitung sensitivitas..."
        ):
            sensitivity = run_sensitivity(
                tuple(selected),
                parameter,
                value_map[parameter],
                params,
                sensitivity_iterations,
            )

        st.dataframe(
            sensitivity.round(3),
            use_container_width=True,
            hide_index=True,
        )

        sensitivity_chart_1, sensitivity_chart_2 = (
            st.columns(2)
        )

        with sensitivity_chart_1:
            mean_sensitivity_chart = px.line(
                sensitivity,
                x="Nilai",
                y="Rata-rata A",
                color="Skenario",
                markers=True,
                title=(
                    "Sensitivitas terhadap "
                    "Rata-rata A"
                ),
            )

            mean_sensitivity_chart.update_layout(
                height=420
            )

            st.plotly_chart(
                mean_sensitivity_chart,
                use_container_width=True,
            )

        with sensitivity_chart_2:
            panic_sensitivity_chart = px.line(
                sensitivity,
                x="Nilai",
                y="Peak % Panik",
                color="Skenario",
                markers=True,
                title=(
                    "Sensitivitas terhadap "
                    "Peak Panik"
                ),
            )

            panic_sensitivity_chart.update_layout(
                height=420
            )

            st.plotly_chart(
                panic_sensitivity_chart,
                use_container_width=True,
            )

        st.download_button(
            label=(
                "Download sensitivitas CSV"
            ),
            data=(
                sensitivity
                .to_csv(index=False)
                .encode("utf-8")
            ),
            file_name=(
                "hasil_sensitivitas.csv"
            ),
            mime="text/csv",
        )

    else:
        st.info(
            "Klik tombol Jalankan sensitivitas "
            "untuk menghitung pengaruh parameter."
        )


# ============================================================
# TAB 6: EXPORT DAN HKI
# ============================================================

with export_tab:
    st.subheader(
        "Export Data Simulasi"
    )

    export_column_1, export_column_2, export_column_3 = (
        st.columns(3)
    )

    with export_column_1:
        st.download_button(
            label="Download ringkasan CSV",
            data=(
                ranked
                .to_csv(index=False)
                .encode("utf-8")
            ),
            file_name=(
                "ringkasan_abm_cbt.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with export_column_2:
        st.download_button(
            label="Download history CSV",
            data=(
                history
                .to_csv(index=False)
                .encode("utf-8")
            ),
            file_name=(
                "history_abm_cbt.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with export_column_3:
        st.download_button(
            label="Download snapshot CSV",
            data=(
                sample
                .to_csv(index=False)
                .encode("utf-8")
            ),
            file_name=(
                "snapshot_agen.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    baseline = ranked[
        ranked["Skenario"]
        == "Tanpa Intervensi"
    ]

    difference = (
        float(
            baseline.iloc[0]["Rata-rata A"]
            - best["Rata-rata A"]
        )
        if len(baseline)
        else np.nan
    )

    conclusion = (
        f"Simulasi menggunakan {iterations} iterasi "
        f"Monte Carlo, {agents} agen, dan {steps} tick. "
        f"Skenario paling stabil adalah "
        f"{best['Skenario']} dengan rata-rata A "
        f"{best['Rata-rata A']:.3f}, peak panik "
        f"{best['Peak % Panik']:.2f}%, dan waktu "
        f"pulih {best['Waktu Pulih']:.2f} tick."
    )

    if not np.isnan(difference):
        conclusion += (
            " Dibandingkan skenario tanpa intervensi, "
            f"penurunan rata-rata A mencapai "
            f"{difference:.3f}."
        )

    scheduled_description = (
        ", ".join(
            map(str, scheduled_ticks)
        )
        if scheduled_ticks
        else "tidak ada"
    )

    methodology = (
        "Model menggunakan Agent-Based Modeling "
        "dengan variabel Anxiety Level (A), "
        "Stressor (S), Cognitive Distortion (D), "
        "Resilience (R), dan CBT Protocol (P). "
        f"Stressor acak memiliki peluang "
        f"{stress_prob:.2f} per tick, sedangkan "
        f"stressor terjadwal diberikan pada tick "
        f"{scheduled_description}. "
        "Transisi kecemasan menggabungkan stressor, "
        "distorsi kognitif, pengaruh sosial, "
        "resilience, CBT, dan noise acak."
    )

    hki = (
        "Luaran HKI yang diusulkan adalah Hak Cipta "
        "Program Komputer untuk aplikasi simulasi "
        "dinamika kecemasan berbasis Agent-Based "
        "Modeling. Unsur yang diwujudkan mencakup "
        "kode sumber, struktur modul simulasi, "
        "dashboard Streamlit, state chart, logika "
        "skenario, visualisasi, analisis sensitivitas, "
        "ekspor data, dan dokumentasi penggunaan. "
        "Model bersifat edukatif dan tidak diklaim "
        "sebagai alat diagnosis klinis."
    )

    st.subheader(
        "Narasi Otomatis"
    )

    st.text_area(
        "Kesimpulan",
        conclusion,
        height=140,
    )

    st.text_area(
        "Metodologi",
        methodology,
        height=140,
    )

    st.text_area(
        "Uraian HKI",
        hki,
        height=140,
    )

    report = f"""# Ringkasan Dashboard ABM CBT

Tanggal ekspor: {datetime.now():%Y-%m-%d %H:%M:%S}

## Parameter Simulasi

- Iterasi Monte Carlo: {iterations}
- Jumlah agen: {agents}
- Jumlah tick: {steps}
- Peluang stressor: {stress_prob:.2f}
- Mean Resilience: {mean_r:.3f}
- Mean Cognitive Distortion: {mean_d:.2f}
- Kekuatan CBT: {cbt_strength:.3f}

## Kesimpulan

{conclusion}

## Metodologi

{methodology}

## Uraian HKI

{hki}
"""

    st.download_button(
        label=(
            "Download ringkasan Markdown"
        ),
        data=report.encode("utf-8"),
        file_name=(
            "ringkasan_laporan_abm_cbt.md"
        ),
        mime="text/markdown",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="foot">
        Dashboard akademik ABM + CBT.
        Gunakan 1000 iterasi untuk hasil final.
        Semua parameter merupakan representasi
        operasional untuk simulasi pendidikan.
    </div>
    """,
    unsafe_allow_html=True,
)
