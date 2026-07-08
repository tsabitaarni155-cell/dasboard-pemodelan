from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import simpy
import streamlit as st


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Dashboard Kiosk Bioskop",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# KONFIGURASI SKENARIO
# ============================================================

SCENARIOS = [
    "Tanpa Intervensi",
    "Intervensi Reaktif",
    "Intervensi Preventif",
    "Beban Tinggi",
]

CONFIGS: Dict[str, Dict[str, float | int | str]] = {
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

DESCRIPTIONS = {
    "Tanpa Intervensi": (
        "Kondisi dasar dengan dua kiosk dan tanpa strategi tambahan."
    ),
    "Intervensi Reaktif": (
        "Pelayanan dipercepat ketika antrean mencapai ambang tertentu."
    ),
    "Intervensi Preventif": (
        "Kapasitas disiapkan sejak awal melalui tiga kiosk dan "
        "pelayanan yang lebih cepat."
    ),
    "Beban Tinggi": (
        "Kondisi ramai dengan kedatangan lebih cepat dan "
        "batas kesabaran pelanggan lebih rendah."
    ),
}

COLORS = {
    "Tanpa Intervensi": "#475569",
    "Intervensi Reaktif": "#2563EB",
    "Intervensi Preventif": "#0F766E",
    "Beban Tinggi": "#C2410C",
}


# ============================================================
# HASIL ACUAN DARI NOTEBOOK PROJECT
# 100 ITERASI MONTE CARLO PER SKENARIO
# ============================================================

NOTEBOOK_RESULTS = pd.DataFrame(
    [
        {
            "scenario": "Tanpa Intervensi",
            "total_arrivals": 158.91,
            "served_customers": 158.43,
            "abandoned_customers": 0.48,
            "avg_wait_time": 1.572541,
            "avg_queue_length": 0.537958,
            "max_queue_length": 5.16,
            "utilization_percent": 66.530257,
            "throughput_per_hour": 19.80375,
            "abandonment_percent": 0.283632,
        },
        {
            "scenario": "Intervensi Reaktif",
            "total_arrivals": 159.46,
            "served_customers": 158.92,
            "abandoned_customers": 0.54,
            "avg_wait_time": 1.673533,
            "avg_queue_length": 0.573833,
            "max_queue_length": 5.30,
            "utilization_percent": 66.652307,
            "throughput_per_hour": 19.86500,
            "abandonment_percent": 0.322663,
        },
        {
            "scenario": "Intervensi Preventif",
            "total_arrivals": 160.79,
            "served_customers": 160.79,
            "abandoned_customers": 0.00,
            "avg_wait_time": 0.111570,
            "avg_queue_length": 0.038521,
            "max_queue_length": 2.26,
            "utilization_percent": 36.510208,
            "throughput_per_hour": 20.09875,
            "abandonment_percent": 0.000000,
        },
        {
            "scenario": "Beban Tinggi",
            "total_arrivals": 234.39,
            "served_customers": 199.67,
            "abandoned_customers": 34.72,
            "avg_wait_time": 5.797899,
            "avg_queue_length": 2.892875,
            "max_queue_length": 10.40,
            "utilization_percent": 94.430388,
            "throughput_per_hour": 24.95875,
            "abandonment_percent": 14.554703,
        },
    ]
)


# ============================================================
# CSS DAN TEMA VISUAL
# ============================================================

st.html("""
    <style>
    :root {
        --navy: #102A43;
        --blue: #2563EB;
        --gold: #D6A84B;
        --teal: #0F766E;
        --ink: #1E293B;
        --muted: #64748B;
        --line: #DCE3EA;
        --paper: #F5F7FA;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 92% 0%,
                rgba(214, 168, 75, 0.12),
                transparent 24rem
            ),
            linear-gradient(
                180deg,
                #FBFCFE 0%,
                #F4F6F9 100%
            );
    }

    .main .block-container {
        max-width: 1480px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--navy);
        letter-spacing: -0.025em;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 22px;
        padding: 30px 34px;
        background:
            linear-gradient(
                115deg,
                #0F2742 0%,
                #163D63 65%,
                #1D4ED8 100%
            );
        box-shadow:
            0 18px 45px rgba(15, 39, 66, 0.18);
        color: white;
        margin-bottom: 1rem;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 245px;
        height: 245px;
        border:
            34px solid rgba(214, 168, 75, 0.16);
        border-radius: 50%;
        right: -65px;
        top: -95px;
    }

    .hero small {
        color: #F4D68A;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .hero h1 {
        color: white;
        margin: 0.45rem 0 0.5rem;
        font-size:
            clamp(1.8rem, 3vw, 2.6rem);
    }

    .hero p {
        max-width: 1040px;
        color: rgba(255, 255, 255, 0.88);
        line-height: 1.65;
        margin: 0;
    }

    .tags {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }

    .tag {
        padding: 0.32rem 0.68rem;
        border:
            1px solid rgba(255, 255, 255, 0.2);
        border-radius: 999px;
        background:
            rgba(255, 255, 255, 0.08);
        font-size: 0.75rem;
        font-weight: 700;
    }

    .metric-card {
        background:
            rgba(255, 255, 255, 0.96);
        border:
            1px solid var(--line);
        border-top:
            4px solid var(--gold);
        border-radius: 16px;
        padding: 16px 18px;
        min-height: 118px;
        box-shadow:
            0 8px 24px rgba(15, 39, 66, 0.06);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.065em;
    }

    .metric-value {
        color: var(--navy);
        font-size: 1.55rem;
        font-weight: 850;
        line-height: 1.15;
        margin-top: 0.35rem;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.8rem;
        margin-top: 0.42rem;
    }

    .box {
        background: white;
        border:
            1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow:
            0 6px 18px rgba(15, 39, 66, 0.04);
        height: 100%;
    }

    .box h3 {
        margin-top: 0;
        font-size: 1.05rem;
    }

    .formula {
        background: #F1F5F9;
        border-left:
            5px solid var(--blue);
        border-radius: 12px;
        padding: 14px 16px;
        font-family:
            ui-monospace,
            SFMono-Regular,
            Consolas,
            monospace;
        line-height: 1.65;
    }

    .flow {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.5rem 0;
    }

    .node {
        padding: 0.55rem 0.78rem;
        border-radius: 10px;
        border: 1px solid #CBD5E1;
        background: #F8FAFC;
        color: #1E293B;
        font-size: 0.82rem;
        font-weight: 750;
    }

    .arrow {
        color: #94A3B8;
        font-weight: 900;
    }

    .insight {
        border-left:
            5px solid var(--teal);
        background: #F0FDFA;
        color: #134E4A;
        border-radius: 12px;
        padding: 13px 16px;
        margin: 0.75rem 0;
    }

    .warning {
        border-left:
            5px solid #C2410C;
        background: #FFF7ED;
        color: #7C2D12;
        border-radius: 12px;
        padding: 13px 16px;
        margin: 0.75rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        border:
            1px solid var(--line);
        background: white;
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: #F7F9FC;
        border-right:
            1px solid #E2E8F0;
    }

    div[data-testid="stDataFrame"] {
        border:
            1px solid var(--line);
        border-radius: 13px;
        overflow: hidden;
    }

    div[data-testid="stForm"] {
        border:
            1px solid var(--line);
        border-radius: 16px;
        background:
            rgba(255, 255, 255, 0.82);
    }

    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top:
            1px solid var(--line);
        color: var(--muted);
        font-size: 0.82rem;
    }
    </style>
    """)


# ============================================================
# FUNGSI TAMPILAN
# ============================================================

def metric_card(
    label: str,
    value: str,
    note: str,
) -> None:
    st.html(f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-note">
                {note}
            </div>
        </div>
        """)


def ordered(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result["scenario"] = pd.Categorical(
        result["scenario"],
        categories=SCENARIOS,
        ordered=True,
    )

    return (
        result
        .sort_values("scenario")
        .reset_index(drop=True)
    )


def show_table(
    dataframe: pd.DataFrame,
    digits: int = 3,
) -> None:
    st.dataframe(
        dataframe.round(digits),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DISTRIBUSI WAKTU
# ============================================================

def generate_service_time(
    mean_service_time: float,
    rng: np.random.Generator,
) -> float:
    sigma = 0.25

    mu = (
        np.log(mean_service_time)
        - 0.5 * sigma**2
    )

    return max(
        0.1,
        float(
            rng.lognormal(
                mean=mu,
                sigma=sigma,
            )
        ),
    )


def generate_interarrival_time(
    mean_interarrival_time: float,
    rng: np.random.Generator,
) -> float:
    return max(
        0.1,
        float(
            rng.exponential(
                mean_interarrival_time
            )
        ),
    )


# ============================================================
# FUNGSI SIMULASI UTAMA
# ============================================================

def simulate_kiosk(
    seed: int = 42,
    duration: int = 480,
    num_kiosks: int = 2,
    mean_interarrival_time: float = 3.0,
    mean_service_time: float = 4.0,
    patience_time: float = 12.0,
    scenario_name: str = "Simulasi",
    scenario_type: str = "base",
    reactive_threshold: int = 6,
    reactive_service_reduction: float = 0.85,
    preventive_service_reduction: float = 0.90,
) -> Tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
]:
    rng = np.random.default_rng(seed)

    environment = simpy.Environment()

    kiosk = simpy.Resource(
        environment,
        capacity=num_kiosks,
    )

    customers = []
    queue_records = []

    busy_time = 0.0

    customer_counter = {
        "value": 0
    }

    def monitor_queue():
        while True:
            queue_records.append(
                {
                    "time":
                        float(environment.now),

                    "queue_length":
                        len(kiosk.queue),

                    "in_service":
                        kiosk.count,

                    "scenario":
                        scenario_name,
                }
            )

            yield environment.timeout(1)

    def customer_process(
        customer_id: int,
    ):
        nonlocal busy_time

        arrival_time = float(
            environment.now
        )

        customer_record = {
            "customer_id":
                customer_id,

            "scenario":
                scenario_name,

            "arrival_time":
                arrival_time,

            "start_service_time":
                np.nan,

            "departure_time":
                np.nan,

            "wait_time":
                np.nan,

            "service_time":
                np.nan,

            "system_time":
                np.nan,

            "state":
                "menunggu",

            "patience_time":
                patience_time,
        }

        with kiosk.request() as request:
            result = yield (
                request
                | environment.timeout(
                    patience_time
                )
            )

            if request not in result:
                customer_record.update(
                    {
                        "departure_time":
                            float(
                                environment.now
                            ),

                        "wait_time":
                            float(
                                environment.now
                                - arrival_time
                            ),

                        "service_time":
                            0.0,

                        "system_time":
                            float(
                                environment.now
                                - arrival_time
                            ),

                        "state":
                            "batal",
                    }
                )

                customers.append(
                    customer_record
                )

                return

            start_service_time = float(
                environment.now
            )

            effective_service_time = (
                mean_service_time
            )

            if (
                scenario_type
                == "reactive"
                and len(kiosk.queue)
                >= reactive_threshold
            ):
                effective_service_time *= (
                    reactive_service_reduction
                )

            elif (
                scenario_type
                == "preventive"
            ):
                effective_service_time *= (
                    preventive_service_reduction
                )

            service_time = (
                generate_service_time(
                    effective_service_time,
                    rng,
                )
            )

            busy_time += service_time

            yield environment.timeout(
                service_time
            )

            departure_time = float(
                environment.now
            )

            customer_record.update(
                {
                    "start_service_time":
                        start_service_time,

                    "departure_time":
                        departure_time,

                    "wait_time":
                        (
                            start_service_time
                            - arrival_time
                        ),

                    "service_time":
                        service_time,

                    "system_time":
                        (
                            departure_time
                            - arrival_time
                        ),

                    "state":
                        "selesai",
                }
            )

            customers.append(
                customer_record
            )

    def arrival_process():
        while environment.now < duration:
            customer_counter["value"] += 1

            environment.process(
                customer_process(
                    customer_counter["value"]
                )
            )

            yield environment.timeout(
                generate_interarrival_time(
                    mean_interarrival_time,
                    rng,
                )
            )

    environment.process(
        arrival_process()
    )

    environment.process(
        monitor_queue()
    )

    environment.run(
        until=duration
    )

    customers_df = pd.DataFrame(
        customers
    )

    queue_df = pd.DataFrame(
        queue_records
    )

    if customers_df.empty:
        summary = {
            "scenario":
                scenario_name,

            "total_arrivals":
                0,

            "served_customers":
                0,

            "abandoned_customers":
                0,

            "avg_wait_time":
                0.0,

            "avg_system_time":
                0.0,

            "avg_queue_length":
                0.0,

            "max_queue_length":
                0.0,

            "utilization_percent":
                0.0,

            "throughput_per_hour":
                0.0,

            "abandonment_percent":
                0.0,

            "num_kiosks":
                num_kiosks,

            "mean_interarrival_time":
                mean_interarrival_time,

            "mean_service_time":
                mean_service_time,

            "patience_time":
                patience_time,

            "duration":
                duration,
        }

        return (
            summary,
            customers_df,
            queue_df,
        )

    served = customers_df[
        customers_df["state"]
        == "selesai"
    ]

    abandoned = customers_df[
        customers_df["state"]
        == "batal"
    ]

    total_customers = len(
        customers_df
    )

    served_customers = len(
        served
    )

    abandoned_customers = len(
        abandoned
    )

    summary = {
        "scenario":
            scenario_name,

        "total_arrivals":
            total_customers,

        "served_customers":
            served_customers,

        "abandoned_customers":
            abandoned_customers,

        "avg_wait_time":
            float(
                customers_df[
                    "wait_time"
                ].mean()
            ),

        "avg_system_time":
            float(
                customers_df[
                    "system_time"
                ].mean()
            ),

        "avg_queue_length":
            float(
                queue_df[
                    "queue_length"
                ].mean()
            ),

        "max_queue_length":
            float(
                queue_df[
                    "queue_length"
                ].max()
            ),

        "utilization_percent":
            float(
                min(
                    100.0,
                    (
                        busy_time
                        / (
                            num_kiosks
                            * duration
                        )
                        * 100
                    ),
                )
            ),

        "throughput_per_hour":
            float(
                served_customers
                / (
                    duration / 60
                )
            ),

        "abandonment_percent":
            float(
                (
                    abandoned_customers
                    / total_customers
                    * 100
                )
                if total_customers > 0
                else 0
            ),

        "num_kiosks":
            num_kiosks,

        "mean_interarrival_time":
            mean_interarrival_time,

        "mean_service_time":
            mean_service_time,

        "patience_time":
            patience_time,

        "duration":
            duration,
    }

    return (
        summary,
        customers_df,
        queue_df,
    )


# ============================================================
# MONTE CARLO
# ============================================================

@st.cache_data(
    show_spinner=False
)
def run_monte_carlo(
    n_iterations: int,
    duration: int,
    num_kiosks: int,
    mean_interarrival_time: float,
    mean_service_time: float,
    patience_time: float,
    scenario_name: str,
    scenario_type: str,
    start_seed: int,
    reactive_threshold: int = 6,
) -> pd.DataFrame:
    rows = []

    for index in range(
        n_iterations
    ):
        summary, _, _ = simulate_kiosk(
            seed=(
                start_seed
                + index
            ),

            duration=
                duration,

            num_kiosks=
                num_kiosks,

            mean_interarrival_time=
                mean_interarrival_time,

            mean_service_time=
                mean_service_time,

            patience_time=
                patience_time,

            scenario_name=
                scenario_name,

            scenario_type=
                scenario_type,

            reactive_threshold=
                reactive_threshold,
        )

        summary["iteration"] = (
            index + 1
        )

        rows.append(
            summary
        )

    return pd.DataFrame(
        rows
    )


def summarize(
    raw_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "total_arrivals",
        "served_customers",
        "abandoned_customers",
        "avg_wait_time",
        "avg_system_time",
        "avg_queue_length",
        "max_queue_length",
        "utilization_percent",
        "throughput_per_hour",
        "abandonment_percent",
    ]

    summary = (
        raw_dataframe
        .groupby(
            "scenario",
            as_index=False,
        )[metrics]
        .mean()
    )

    return ordered(
        summary
    )


@st.cache_data(
    show_spinner=False
)
def run_all_scenarios(
    iterations: int,
    duration: int,
    reactive_threshold: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    frames = []

    for index, name in enumerate(
        SCENARIOS
    ):
        config = CONFIGS[name]

        frame = run_monte_carlo(
            n_iterations=
                iterations,

            duration=
                duration,

            num_kiosks=
                int(
                    config[
                        "num_kiosks"
                    ]
                ),

            mean_interarrival_time=
                float(
                    config[
                        "mean_interarrival_time"
                    ]
                ),

            mean_service_time=
                float(
                    config[
                        "mean_service_time"
                    ]
                ),

            patience_time=
                float(
                    config[
                        "patience_time"
                    ]
                ),

            scenario_name=
                name,

            scenario_type=
                str(
                    config[
                        "scenario_type"
                    ]
                ),

            start_seed=
                (
                    1000
                    + index * 10000
                ),

            reactive_threshold=
                reactive_threshold,
        )

        frames.append(
            frame
        )

    raw_results = pd.concat(
        frames,
        ignore_index=True,
    )

    return (
        summarize(
            raw_results
        ),
        raw_results,
    )


# ============================================================
# OPTIMASI JUMLAH KIOSK
# ============================================================

@st.cache_data(
    show_spinner=False
)
def optimize_kiosks(
    max_kiosk: int,
    iterations: int,
    duration: int,
    interarrival: float,
    service: float,
    patience: float,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    frames = []

    for kiosk_count in range(
        1,
        max_kiosk + 1,
    ):
        frame = run_monte_carlo(
            n_iterations=
                iterations,

            duration=
                duration,

            num_kiosks=
                kiosk_count,

            mean_interarrival_time=
                interarrival,

            mean_service_time=
                service,

            patience_time=
                patience,

            scenario_name=
                f"{kiosk_count} Kiosk",

            scenario_type=
                "base",

            start_seed=
                (
                    5000
                    + kiosk_count * 1000
                ),
        )

        frames.append(
            frame
        )

    raw_results = pd.concat(
        frames,
        ignore_index=True,
    )

    summary = (
        raw_results
        .groupby(
            "num_kiosks",
            as_index=False,
        )
        .agg(
            avg_wait_time=(
                "avg_wait_time",
                "mean",
            ),

            avg_queue_length=(
                "avg_queue_length",
                "mean",
            ),

            utilization_percent=(
                "utilization_percent",
                "mean",
            ),

            throughput_per_hour=(
                "throughput_per_hour",
                "mean",
            ),

            abandonment_percent=(
                "abandonment_percent",
                "mean",
            ),
        )
    )

    return (
        summary,
        raw_results,
    )


# ============================================================
# ANALISIS SENSITIVITAS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def sensitivity_analysis(
    parameter: str,
    values: Tuple[
        float,
        ...
    ],
    iterations: int,
    kiosk_count: int,
) -> pd.DataFrame:
    rows = []

    for index, value in enumerate(
        values
    ):
        interarrival = 3.0
        service = 4.0
        patience = 12.0

        if (
            parameter
            == "Interval kedatangan"
        ):
            interarrival = value

        elif (
            parameter
            == "Waktu pelayanan"
        ):
            service = value

        else:
            patience = value

        raw_results = run_monte_carlo(
            n_iterations=
                iterations,

            duration=
                480,

            num_kiosks=
                kiosk_count,

            mean_interarrival_time=
                interarrival,

            mean_service_time=
                service,

            patience_time=
                patience,

            scenario_name=
                (
                    f"{parameter} "
                    f"{value:.2f}"
                ),

            scenario_type=
                "base",

            start_seed=
                (
                    20000
                    + index * 1000
                ),
        )

        rows.append(
            {
                "parameter":
                    parameter,

                "value":
                    value,

                "avg_wait_time":
                    raw_results[
                        "avg_wait_time"
                    ].mean(),

                "avg_queue_length":
                    raw_results[
                        "avg_queue_length"
                    ].mean(),

                "utilization_percent":
                    raw_results[
                        "utilization_percent"
                    ].mean(),

                "throughput_per_hour":
                    raw_results[
                        "throughput_per_hour"
                    ].mean(),

                "abandonment_percent":
                    raw_results[
                        "abandonment_percent"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# NARASI HASIL OTOMATIS
# ============================================================

def narrative(
    summary_dataframe: pd.DataFrame,
    iterations: int,
) -> str:
    best = summary_dataframe.loc[
        summary_dataframe[
            "avg_wait_time"
        ].idxmin()
    ]

    worst = summary_dataframe.loc[
        summary_dataframe[
            "avg_wait_time"
        ].idxmax()
    ]

    highest_throughput = (
        summary_dataframe.loc[
            summary_dataframe[
                "throughput_per_hour"
            ].idxmax()
        ]
    )

    return (
        f"Simulasi menggunakan "
        f"{iterations} iterasi Monte Carlo "
        f"per skenario. "

        f"{best['scenario']} memberikan "
        f"waktu tunggu terendah sebesar "
        f"{best['avg_wait_time']:.2f} menit "
        f"dengan pembatalan "
        f"{best['abandonment_percent']:.2f}%. "

        f"Kondisi paling berat terdapat pada "
        f"{worst['scenario']} dengan waktu "
        f"tunggu {worst['avg_wait_time']:.2f} "
        f"menit dan pembatalan "
        f"{worst['abandonment_percent']:.2f}%. "

        f"Throughput tertinggi terdapat pada "
        f"{highest_throughput['scenario']} "
        f"sebesar "
        f"{highest_throughput['throughput_per_hour']:.2f} "
        f"pelanggan per jam. "

        "Hasil ini menunjukkan bahwa "
        "persiapan kapasitas secara preventif "
        "lebih stabil daripada menunggu "
        "antrean memburuk."
    )


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION = {
    "comparison_summary":
        ordered(
            NOTEBOOK_RESULTS
        ),

    "comparison_raw":
        None,

    "comparison_iterations":
        100,

    "comparison_source":
        "Hasil notebook",

    "single_result":
        None,

    "optimization":
        None,

    "optimization_raw":
        None,

    "sensitivity":
        None,
}

for key, value in (
    DEFAULT_SESSION.items()
):
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.html("""
    <div class="hero">
        <small>
            Tugas Besar Pemodelan dan Simulasi
        </small>

        <h1>
            Simulasi Sistem Antrean
            Self-Service Kiosk Bioskop
        </h1>

        <p>
            Dashboard akademik berbasis
            Agent-Based Discrete Event Simulation
            untuk menganalisis waktu tunggu,
            panjang antrean, utilisasi kiosk,
            throughput, serta pelanggan yang
            meninggalkan antrean.
        </p>

        <div class="tags">
            <span class="tag">
                Agent-Based Modeling
            </span>

            <span class="tag">
                Discrete Event Simulation
            </span>

            <span class="tag">
                Monte Carlo
            </span>

            <span class="tag">
                What-If Analysis
            </span>
        </div>
    </div>
    """)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.html("## Informasi Project"
    )

    st.caption(
        "Objek: antrean pembelian tiket "
        "melalui self-service kiosk bioskop."
    )

    st.write(
        "Hasil acuan: "
        "**100 iterasi/skenario**"
    )

    st.write(
        "Durasi acuan: "
        "**480 menit**"
    )

    st.divider()

    st.markdown(
        "### Indikator"
    )

    st.caption(
        "Waktu tunggu: durasi sebelum "
        "pelanggan memperoleh kiosk."
    )

    st.caption(
        "Utilisasi: persentase kapasitas "
        "kiosk yang terpakai."
    )

    st.caption(
        "Throughput: pelanggan selesai "
        "dilayani per jam."
    )

    st.caption(
        "Abandonment: pelanggan pergi "
        "karena melewati batas sabar."
    )

    st.divider()

    st.info(
        "Gunakan 100 iterasi untuk "
        "mereplikasi notebook. "
        "Mode 1000 iterasi memerlukan "
        "waktu proses lebih lama."
    )


# ============================================================
# KPI UTAMA
# ============================================================

current_summary = (
    st.session_state[
        "comparison_summary"
    ]
)

best_scenario = (
    current_summary.loc[
        current_summary[
            "avg_wait_time"
        ].idxmin()
    ]
)

worst_cancel = (
    current_summary.loc[
        current_summary[
            "abandonment_percent"
        ].idxmax()
    ]
)

top_throughput = (
    current_summary.loc[
        current_summary[
            "throughput_per_hour"
        ].idxmax()
    ]
)

metric_columns = st.columns(4)

with metric_columns[0]:
    metric_card(
        "Skenario terbaik",
        str(
            best_scenario[
                "scenario"
            ]
        ),
        (
            "Waktu tunggu "
            f"{best_scenario['avg_wait_time']:.2f} "
            "menit"
        ),
    )

with metric_columns[1]:
    metric_card(
        "Waktu tunggu terendah",
        (
            f"{best_scenario['avg_wait_time']:.2f} "
            "menit"
        ),
        "Rata-rata seluruh iterasi",
    )

with metric_columns[2]:
    metric_card(
        "Pembatalan tertinggi",
        (
            f"{worst_cancel['abandonment_percent']:.2f}%"
        ),
        str(
            worst_cancel[
                "scenario"
            ]
        ),
    )

with metric_columns[3]:
    metric_card(
        "Throughput tertinggi",
        (
            f"{top_throughput['throughput_per_hour']:.2f}"
            "/jam"
        ),
        str(
            top_throughput[
                "scenario"
            ]
        ),
    )


# ============================================================
# TAB DASHBOARD
# ============================================================

(
    overview_tab,
    result_tab,
    simulation_tab,
    optimization_tab,
    sensitivity_tab,
    export_tab,
) = st.tabs(
    [
        "Ringkasan Project",
        "Hasil Eksperimen",
        "Simulasi Interaktif",
        "Optimasi Kiosk",
        "Sensitivitas",
        "Metodologi & Ekspor",
    ]
)


# ============================================================
# TAB 1 - RINGKASAN PROJECT
# ============================================================

with overview_tab:
    st.subheader(
        "Latar Belakang dan Tujuan"
    )

    problem_column, objective_column = (
        st.columns(2)
    )

    with problem_column:
        st.markdown(
            """
            <div class="box">
                <h3>
                    Masalah yang dimodelkan
                </h3>

                <p>
                    Lonjakan pengunjung dapat
                    membuat kapasitas kiosk tidak
                    mencukupi. Dampaknya berupa
                    waktu tunggu tinggi, antrean
                    panjang, utilisasi mendekati
                    penuh, dan pelanggan yang
                    membatalkan transaksi.
                </p>
            </div>
            """)

    with objective_column:
        st.html("""
            <div class="box">
                <h3>
                    Tujuan simulasi
                </h3>

                <p>
                    Menguji pengaruh laju kedatangan,
                    jumlah kiosk, waktu pelayanan,
                    batas kesabaran, serta intervensi
                    reaktif dan preventif terhadap
                    kinerja sistem antrean.
                </p>
            </div>
            """)

    st.subheader(
        "State Chart Pelanggan"
    )

    st.html("""
        <div class="box">
            <div class="flow">
                <span class="node">
                    Datang
                </span>

                <span class="arrow">
                    →
                </span>

                <span class="node">
                    Menunggu
                </span>

                <span class="arrow">
                    →
                </span>

                <span class="node">
                    Dilayani
                </span>

                <span class="arrow">
                    →
                </span>

                <span class="node">
                    Selesai
                </span>
            </div>

            <div class="flow">
                <span class="node">
                    Menunggu
                </span>

                <span class="arrow">
                    → jika melebihi batas sabar →
                </span>

                <span class="node">
                    Batal
                </span>
            </div>
        </div>
        """)

    st.subheader(
        "Formulasi Indikator"
    )

    formula_column_1, formula_column_2 = (
        st.columns(2)
    )

    with formula_column_1:
        st.html("""
            <div class="formula">
                Waktu tunggu =
                mulai dilayani − waktu datang
                <br>

                Waktu sistem =
                waktu keluar − waktu datang
                <br>

                Throughput =
                pelanggan selesai / durasi (jam)
            </div>
            """)

    with formula_column_2:
        st.html("""
            <div class="formula">
                Utilisasi =
                busy time / (kiosk × durasi) × 100%
                <br>

                Abandonment =
                pelanggan batal / total pelanggan × 100%
                <br>

                Panjang antrean =
                agen yang menunggu pada waktu t
            </div>
            """)

    scenario_rows = []

    for scenario_name in SCENARIOS:
        config = CONFIGS[
            scenario_name
        ]

        scenario_rows.append(
            {
                "Skenario":
                    scenario_name,

                "Kiosk":
                    config[
                        "num_kiosks"
                    ],

                "Interval kedatangan":
                    config[
                        "mean_interarrival_time"
                    ],

                "Waktu pelayanan":
                    config[
                        "mean_service_time"
                    ],

                "Batas sabar":
                    config[
                        "patience_time"
                    ],

                "Keterangan":
                    DESCRIPTIONS[
                        scenario_name
                    ],
            }
        )

    st.subheader(
        "Skenario What-If"
    )

    show_table(
        pd.DataFrame(
            scenario_rows
        ),
        2,
    )


# ============================================================
# TAB 2 - HASIL EKSPERIMEN
# ============================================================

with result_tab:
    st.subheader(
        "Perbandingan Empat Skenario"
    )

    st.caption(
        "Data aktif: "
        f"{st.session_state['comparison_source']} · "
        f"{st.session_state['comparison_iterations']} "
        "iterasi per skenario."
    )

    with st.form(
        "comparison_form"
    ):
        form_column_1, form_column_2, form_column_3 = (
            st.columns(3)
        )

        with form_column_1:
            comparison_iterations = (
                st.select_slider(
                    "Iterasi Monte Carlo",
                    options=[
                        30,
                        50,
                        100,
                        200,
                        500,
                        1000,
                    ],
                    value=100,
                )
            )

        with form_column_2:
            comparison_duration = (
                st.slider(
                    "Durasi simulasi (menit)",
                    min_value=120,
                    max_value=720,
                    value=480,
                    step=60,
                )
            )

        with form_column_3:
            comparison_threshold = (
                st.slider(
                    "Ambang antrean reaktif",
                    min_value=2,
                    max_value=12,
                    value=6,
                    step=1,
                )
            )

        run_comparison = (
            st.form_submit_button(
                "Jalankan ulang empat skenario",
                type="primary",
                use_container_width=True,
            )
        )

    if run_comparison:
        with st.spinner(
            "Menjalankan eksperimen "
            "Monte Carlo..."
        ):
            summary_dataframe, raw_dataframe = (
                run_all_scenarios(
                    int(
                        comparison_iterations
                    ),
                    int(
                        comparison_duration
                    ),
                    int(
                        comparison_threshold
                    ),
                )
            )

        st.session_state[
            "comparison_summary"
        ] = summary_dataframe

        st.session_state[
            "comparison_raw"
        ] = raw_dataframe

        st.session_state[
            "comparison_iterations"
        ] = int(
            comparison_iterations
        )

        st.session_state[
            "comparison_source"
        ] = "Hasil simulasi dashboard"

        st.rerun()

    summary_dataframe = ordered(
        st.session_state[
            "comparison_summary"
        ]
    )

    shown_columns = [
        "scenario",
        "avg_wait_time",
        "avg_queue_length",
        "max_queue_length",
        "utilization_percent",
        "throughput_per_hour",
        "abandonment_percent",
        "served_customers",
        "abandoned_customers",
    ]

    show_table(
        summary_dataframe[
            shown_columns
        ],
        3,
    )

    metric_options = {
        "avg_wait_time":
            "Waktu tunggu rata-rata (menit)",

        "avg_queue_length":
            "Panjang antrean rata-rata",

        "utilization_percent":
            "Utilisasi kiosk (%)",

        "throughput_per_hour":
            "Throughput (pelanggan/jam)",

        "abandonment_percent":
            "Pelanggan batal (%)",
    }

    selected_metric = st.selectbox(
        "Indikator grafik",
        options=list(
            metric_options
        ),
        format_func=lambda value: (
            metric_options[value]
        ),
    )

    comparison_chart = px.bar(
        summary_dataframe,
        x="scenario",
        y=selected_metric,
        color="scenario",
        color_discrete_map=COLORS,
        text=selected_metric,
        title=(
            metric_options[
                selected_metric
            ]
        ),
    )

    comparison_chart.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    comparison_chart.update_layout(
        xaxis_title="Skenario",
        yaxis_title=(
            metric_options[
                selected_metric
            ]
        ),
        showlegend=False,
        height=460,
        margin=dict(
            t=70,
            b=70,
        ),
    )

    st.plotly_chart(
        comparison_chart,
        use_container_width=True,
    )

    raw_comparison = (
        st.session_state[
            "comparison_raw"
        ]
    )

    if raw_comparison is not None:
        box_chart = px.box(
            raw_comparison,
            x="scenario",
            y="avg_wait_time",
            color="scenario",
            color_discrete_map=COLORS,
            points=False,
            title=(
                "Sebaran Waktu Tunggu "
                "Monte Carlo"
            ),
        )

        box_chart.update_layout(
            xaxis_title="Skenario",
            yaxis_title=(
                "Waktu tunggu (menit)"
            ),
            showlegend=False,
            height=420,
        )

        st.plotly_chart(
            box_chart,
            use_container_width=True,
        )

    automatic_narrative = narrative(
        summary_dataframe,
        st.session_state[
            "comparison_iterations"
        ],
    )

    st.html((
            '<div class="insight">'
            '<b>Interpretasi:</b> '
            f'{automatic_narrative}'
            '</div>'
        ))


# ============================================================
# TAB 3 - SIMULASI INTERAKTIF
# ============================================================

with simulation_tab:
    st.subheader(
        "Eksperimen Satu Skenario"
    )

    with st.form(
        "single_form"
    ):
        row_1_column_1, row_1_column_2, row_1_column_3 = (
            st.columns(3)
        )

        with row_1_column_1:
            selected_scenario = (
                st.selectbox(
                    "Skenario",
                    SCENARIOS,
                )
            )

        selected_config = (
            CONFIGS[
                selected_scenario
            ]
        )

        with row_1_column_2:
            single_iterations = (
                st.select_slider(
                    "Iterasi Monte Carlo",
                    options=[
                        10,
                        30,
                        50,
                        100,
                        200,
                        500,
                    ],
                    value=50,
                )
            )

        with row_1_column_3:
            single_seed = (
                st.number_input(
                    "Seed",
                    min_value=1,
                    max_value=999999,
                    value=123,
                )
            )

        row_2_column_1, row_2_column_2, row_2_column_3 = (
            st.columns(3)
        )

        with row_2_column_1:
            single_duration = (
                st.slider(
                    "Durasi (menit)",
                    min_value=60,
                    max_value=720,
                    value=480,
                    step=30,
                )
            )

        with row_2_column_2:
            single_kiosks = (
                st.slider(
                    "Jumlah kiosk",
                    min_value=1,
                    max_value=8,
                    value=int(
                        selected_config[
                            "num_kiosks"
                        ]
                    ),
                    step=1,
                )
            )

        with row_2_column_3:
            single_patience = (
                st.slider(
                    "Batas kesabaran",
                    min_value=3.0,
                    max_value=30.0,
                    value=float(
                        selected_config[
                            "patience_time"
                        ]
                    ),
                    step=1.0,
                )
            )

        row_3_column_1, row_3_column_2, row_3_column_3 = (
            st.columns(3)
        )

        with row_3_column_1:
            single_interarrival = (
                st.slider(
                    "Interval kedatangan",
                    min_value=1.0,
                    max_value=8.0,
                    value=float(
                        selected_config[
                            "mean_interarrival_time"
                        ]
                    ),
                    step=0.25,
                )
            )

        with row_3_column_2:
            single_service = (
                st.slider(
                    "Waktu pelayanan",
                    min_value=1.0,
                    max_value=10.0,
                    value=float(
                        selected_config[
                            "mean_service_time"
                        ]
                    ),
                    step=0.25,
                )
            )

        with row_3_column_3:
            single_threshold = (
                st.slider(
                    "Ambang reaktif",
                    min_value=2,
                    max_value=12,
                    value=6,
                    step=1,
                )
            )

        run_single = (
            st.form_submit_button(
                "Jalankan simulasi interaktif",
                type="primary",
                use_container_width=True,
            )
        )

    if run_single:
        with st.spinner(
            "Menghitung simulasi..."
        ):
            one_summary, customers_dataframe, queue_dataframe = (
                simulate_kiosk(
                    seed=int(
                        single_seed
                    ),

                    duration=int(
                        single_duration
                    ),

                    num_kiosks=int(
                        single_kiosks
                    ),

                    mean_interarrival_time=float(
                        single_interarrival
                    ),

                    mean_service_time=float(
                        single_service
                    ),

                    patience_time=float(
                        single_patience
                    ),

                    scenario_name=
                        selected_scenario,

                    scenario_type=str(
                        selected_config[
                            "scenario_type"
                        ]
                    ),

                    reactive_threshold=int(
                        single_threshold
                    ),
                )
            )

            single_monte_carlo = (
                run_monte_carlo(
                    n_iterations=int(
                        single_iterations
                    ),

                    duration=int(
                        single_duration
                    ),

                    num_kiosks=int(
                        single_kiosks
                    ),

                    mean_interarrival_time=float(
                        single_interarrival
                    ),

                    mean_service_time=float(
                        single_service
                    ),

                    patience_time=float(
                        single_patience
                    ),

                    scenario_name=
                        selected_scenario,

                    scenario_type=str(
                        selected_config[
                            "scenario_type"
                        ]
                    ),

                    start_seed=9000,

                    reactive_threshold=int(
                        single_threshold
                    ),
                )
            )

        st.session_state[
            "single_result"
        ] = (
            one_summary,
            customers_dataframe,
            queue_dataframe,
            single_monte_carlo,
        )

    if (
        st.session_state[
            "single_result"
        ]
        is None
    ):
        st.info(
            "Atur parameter, lalu "
            "jalankan simulasi."
        )

    else:
        (
            one_summary,
            customers_dataframe,
            queue_dataframe,
            single_monte_carlo,
        ) = st.session_state[
            "single_result"
        ]

        mean_result = (
            single_monte_carlo
            .mean(
                numeric_only=True
            )
        )

        result_cards = (
            st.columns(5)
        )

        card_values = [
            (
                "Waktu tunggu",
                (
                    f"{mean_result['avg_wait_time']:.2f} "
                    "menit"
                ),
                "Rata-rata Monte Carlo",
            ),
            (
                "Antrean",
                (
                    f"{mean_result['avg_queue_length']:.2f}"
                ),
                (
                    "Maksimum "
                    f"{mean_result['max_queue_length']:.2f}"
                ),
            ),
            (
                "Utilisasi",
                (
                    f"{mean_result['utilization_percent']:.2f}%"
                ),
                "Kapasitas terpakai",
            ),
            (
                "Throughput",
                (
                    f"{mean_result['throughput_per_hour']:.2f}"
                    "/jam"
                ),
                "Pelanggan selesai",
            ),
            (
                "Pelanggan batal",
                (
                    f"{mean_result['abandonment_percent']:.2f}%"
                ),
                "Melewati batas sabar",
            ),
        ]

        for column, card_value in zip(
            result_cards,
            card_values,
        ):
            with column:
                metric_card(
                    *card_value
                )

        chart_column_1, chart_column_2 = (
            st.columns(2)
        )

        with chart_column_1:
            queue_chart = go.Figure()

            queue_chart.add_trace(
                go.Scatter(
                    x=queue_dataframe[
                        "time"
                    ],

                    y=queue_dataframe[
                        "queue_length"
                    ],

                    mode="lines",

                    name="Menunggu",

                    line=dict(
                        color="#2563EB",
                        width=2,
                    ),

                    fill="tozeroy",

                    fillcolor=(
                        "rgba(37,99,235,0.12)"
                    ),
                )
            )

            queue_chart.add_trace(
                go.Scatter(
                    x=queue_dataframe[
                        "time"
                    ],

                    y=queue_dataframe[
                        "in_service"
                    ],

                    mode="lines",

                    name="Dilayani",

                    line=dict(
                        color="#D6A84B",
                        width=2,
                    ),
                )
            )

            queue_chart.update_layout(
                title="Dinamika Antrean",

                xaxis_title=
                    "Waktu (menit)",

                yaxis_title=
                    "Jumlah pelanggan",

                height=420,
            )

            st.plotly_chart(
                queue_chart,
                use_container_width=True,
            )

        with chart_column_2:
            histogram = px.histogram(
                customers_dataframe,

                x="wait_time",

                color="state",

                nbins=24,

                color_discrete_map={
                    "selesai":
                        "#0F766E",

                    "batal":
                        "#B42318",
                },

                title=(
                    "Distribusi Waktu Tunggu"
                ),
            )

            histogram.update_layout(
                xaxis_title=(
                    "Waktu tunggu (menit)"
                ),

                yaxis_title=(
                    "Jumlah pelanggan"
                ),

                height=420,
            )

            st.plotly_chart(
                histogram,
                use_container_width=True,
            )

        monte_carlo_box = px.box(
            single_monte_carlo,

            x="scenario",

            y="avg_wait_time",

            points="outliers",

            title=(
                "Sebaran Waktu Tunggu "
                "Monte Carlo"
            ),
        )

        monte_carlo_box.update_layout(
            xaxis_title="",

            yaxis_title=(
                "Waktu tunggu (menit)"
            ),

            height=380,
        )

        st.plotly_chart(
            monte_carlo_box,
            use_container_width=True,
        )

        st.subheader(
            "Data Pelanggan"
        )

        show_table(
            customers_dataframe.head(
                100
            ),
            3,
        )

        if (
            mean_result[
                "abandonment_percent"
            ] > 10
            or mean_result[
                "avg_wait_time"
            ] > 8
        ):
            recommendation = (
                "Sistem belum optimal. "
                "Tambahkan kiosk, percepat "
                "pelayanan, atau gunakan "
                "strategi preventif."
            )

            css_class = "warning"

        elif (
            mean_result[
                "utilization_percent"
            ] > 90
        ):
            recommendation = (
                "Kiosk mendekati kapasitas "
                "penuh dan rentan mengalami "
                "antrean ketika terjadi "
                "lonjakan kedatangan."
            )

            css_class = "warning"

        else:
            recommendation = (
                "Kinerja sistem relatif "
                "terkendali pada konfigurasi ini."
            )

            css_class = "insight"

        st.html((
                f'<div class="{css_class}">'
                '<b>Rekomendasi:</b> '
                f'{recommendation}'
                '</div>'
            ))


# ============================================================
# TAB 4 - OPTIMASI KIOSK
# ============================================================

with optimization_tab:
    st.subheader(
        "Optimasi Jumlah Kiosk"
    )

    with st.form(
        "optimization_form"
    ):
        opt_column_1, opt_column_2, opt_column_3 = (
            st.columns(3)
        )

        with opt_column_1:
            maximum_kiosk = (
                st.slider(
                    "Kiosk maksimum",
                    min_value=3,
                    max_value=8,
                    value=6,
                    step=1,
                )
            )

        with opt_column_2:
            optimization_iterations = (
                st.select_slider(
                    "Iterasi per konfigurasi",
                    options=[
                        20,
                        30,
                        50,
                        100,
                        200,
                    ],
                    value=50,
                )
            )

        with opt_column_3:
            optimization_duration = (
                st.slider(
                    "Durasi (menit)",
                    min_value=120,
                    max_value=720,
                    value=480,
                    step=60,
                )
            )

        opt_column_4, opt_column_5, opt_column_6 = (
            st.columns(3)
        )

        with opt_column_4:
            optimization_interarrival = (
                st.slider(
                    "Interval kedatangan",
                    min_value=1.0,
                    max_value=8.0,
                    value=3.0,
                    step=0.25,
                    key="opt_iat",
                )
            )

        with opt_column_5:
            optimization_service = (
                st.slider(
                    "Waktu pelayanan",
                    min_value=1.0,
                    max_value=10.0,
                    value=4.0,
                    step=0.25,
                    key="opt_service",
                )
            )

        with opt_column_6:
            optimization_patience = (
                st.slider(
                    "Batas sabar",
                    min_value=3.0,
                    max_value=30.0,
                    value=12.0,
                    step=1.0,
                    key="opt_patience",
                )
            )

        run_optimization = (
            st.form_submit_button(
                "Jalankan optimasi",
                type="primary",
                use_container_width=True,
            )
        )

    if run_optimization:
        with st.spinner(
            "Menguji jumlah kiosk..."
        ):
            optimization_summary, optimization_raw = (
                optimize_kiosks(
                    int(
                        maximum_kiosk
                    ),

                    int(
                        optimization_iterations
                    ),

                    int(
                        optimization_duration
                    ),

                    float(
                        optimization_interarrival
                    ),

                    float(
                        optimization_service
                    ),

                    float(
                        optimization_patience
                    ),
                )
            )

        st.session_state[
            "optimization"
        ] = optimization_summary

        st.session_state[
            "optimization_raw"
        ] = optimization_raw

    if (
        st.session_state[
            "optimization"
        ]
        is None
    ):
        st.info(
            "Jalankan optimasi untuk "
            "membandingkan beberapa jumlah kiosk."
        )

    else:
        optimization_summary = (
            st.session_state[
                "optimization"
            ]
        )

        show_table(
            optimization_summary,
            3,
        )

        feasible_result = (
            optimization_summary[
                (
                    optimization_summary[
                        "avg_wait_time"
                    ] <= 1.0
                )
                & (
                    optimization_summary[
                        "abandonment_percent"
                    ] <= 1.0
                )
                & (
                    optimization_summary[
                        "utilization_percent"
                    ] <= 85.0
                )
            ]
        )

        if not feasible_result.empty:
            recommended_kiosk = int(
                feasible_result.iloc[0][
                    "num_kiosks"
                ]
            )

            st.html((
                    '<div class="insight">'
                    '<b>Rekomendasi:</b> '
                    f'{recommended_kiosk} kiosk '
                    'merupakan kapasitas minimum '
                    'yang memenuhi waktu tunggu '
                    '≤ 1 menit, pembatalan ≤ 1%, '
                    'dan utilisasi ≤ 85%.'
                    '</div>'
                ))

        else:
            st.html("""
                <div class="warning">
                    Belum ada konfigurasi yang
                    memenuhi seluruh sasaran.
                </div>
                """)

        optimization_chart_column_1, optimization_chart_column_2 = (
            st.columns(2)
        )

        with optimization_chart_column_1:
            waiting_chart = px.line(
                optimization_summary,

                x="num_kiosks",

                y="avg_wait_time",

                markers=True,

                title=(
                    "Jumlah Kiosk vs "
                    "Waktu Tunggu"
                ),
            )

            waiting_chart.update_layout(
                xaxis_title=(
                    "Jumlah kiosk"
                ),

                yaxis_title=(
                    "Waktu tunggu (menit)"
                ),

                height=410,
            )

            st.plotly_chart(
                waiting_chart,
                use_container_width=True,
            )

        with optimization_chart_column_2:
            cancellation_chart = px.line(
                optimization_summary,

                x="num_kiosks",

                y="abandonment_percent",

                markers=True,

                title=(
                    "Jumlah Kiosk vs "
                    "Pembatalan"
                ),
            )

            cancellation_chart.update_layout(
                xaxis_title=(
                    "Jumlah kiosk"
                ),

                yaxis_title=(
                    "Pelanggan batal (%)"
                ),

                height=410,
            )

            st.plotly_chart(
                cancellation_chart,
                use_container_width=True,
            )

        utilization_chart = px.bar(
            optimization_summary,

            x="num_kiosks",

            y="utilization_percent",

            text="utilization_percent",

            title=(
                "Utilisasi berdasarkan "
                "Jumlah Kiosk"
            ),
        )

        utilization_chart.add_hline(
            y=85,

            line_dash="dash",

            annotation_text=(
                "Sasaran 85%"
            ),
        )

        utilization_chart.update_traces(
            texttemplate=(
                "%{text:.1f}%"
            )
        )

        utilization_chart.update_layout(
            xaxis_title=(
                "Jumlah kiosk"
            ),

            yaxis_title=(
                "Utilisasi (%)"
            ),

            height=420,
        )

        st.plotly_chart(
            utilization_chart,
            use_container_width=True,
        )


# ============================================================
# TAB 5 - ANALISIS SENSITIVITAS
# ============================================================

with sensitivity_tab:
    st.subheader(
        "Analisis Sensitivitas"
    )

    with st.form(
        "sensitivity_form"
    ):
        sens_column_1, sens_column_2, sens_column_3 = (
            st.columns(3)
        )

        with sens_column_1:
            sensitivity_parameter = (
                st.selectbox(
                    "Parameter",
                    [
                        "Interval kedatangan",
                        "Waktu pelayanan",
                        "Batas kesabaran",
                    ],
                )
            )

        with sens_column_2:
            sensitivity_iterations = (
                st.select_slider(
                    "Iterasi per nilai",
                    options=[
                        20,
                        30,
                        50,
                        100,
                        200,
                    ],
                    value=50,
                )
            )

        with sens_column_3:
            sensitivity_kiosk = (
                st.slider(
                    "Jumlah kiosk",
                    min_value=1,
                    max_value=6,
                    value=2,
                    step=1,
                )
            )

        run_sensitivity = (
            st.form_submit_button(
                "Jalankan sensitivitas",
                type="primary",
                use_container_width=True,
            )
        )

    sensitivity_values = {
        "Interval kedatangan":
            tuple(
                np.round(
                    np.linspace(
                        1.5,
                        5.0,
                        8,
                    ),
                    2,
                )
            ),

        "Waktu pelayanan":
            tuple(
                np.round(
                    np.linspace(
                        2.0,
                        7.0,
                        8,
                    ),
                    2,
                )
            ),

        "Batas kesabaran":
            tuple(
                np.round(
                    np.linspace(
                        5.0,
                        20.0,
                        8,
                    ),
                    2,
                )
            ),
    }

    if run_sensitivity:
        with st.spinner(
            "Menghitung sensitivitas..."
        ):
            st.session_state[
                "sensitivity"
            ] = sensitivity_analysis(
                sensitivity_parameter,

                sensitivity_values[
                    sensitivity_parameter
                ],

                int(
                    sensitivity_iterations
                ),

                int(
                    sensitivity_kiosk
                ),
            )

    if (
        st.session_state[
            "sensitivity"
        ]
        is None
    ):
        st.info(
            "Jalankan analisis untuk melihat "
            "pengaruh perubahan parameter."
        )

    else:
        sensitivity_dataframe = (
            st.session_state[
                "sensitivity"
            ]
        )

        show_table(
            sensitivity_dataframe,
            3,
        )

        sensitivity_metrics = {
            "avg_wait_time":
                "Waktu tunggu (menit)",

            "avg_queue_length":
                "Panjang antrean",

            "utilization_percent":
                "Utilisasi (%)",

            "throughput_per_hour":
                "Throughput/jam",

            "abandonment_percent":
                "Pelanggan batal (%)",
        }

        sensitivity_metric = (
            st.selectbox(
                "Indikator",
                options=list(
                    sensitivity_metrics
                ),
                format_func=lambda value: (
                    sensitivity_metrics[
                        value
                    ]
                ),
                key="sens_metric",
            )
        )

        sensitivity_chart = px.line(
            sensitivity_dataframe,

            x="value",

            y=sensitivity_metric,

            markers=True,

            title=(
                "Pengaruh "
                f"{sensitivity_dataframe.iloc[0]['parameter']} "
                "terhadap "
                f"{sensitivity_metrics[sensitivity_metric]}"
            ),
        )

        sensitivity_chart.update_layout(
            xaxis_title=(
                sensitivity_dataframe.iloc[0][
                    "parameter"
                ]
            ),

            yaxis_title=(
                sensitivity_metrics[
                    sensitivity_metric
                ]
            ),

            height=450,
        )

        st.plotly_chart(
            sensitivity_chart,
            use_container_width=True,
        )


# ============================================================
# TAB 6 - METODOLOGI DAN EKSPOR
# ============================================================

with export_tab:
    st.subheader(
        "Metodologi"
    )

    methodology = pd.DataFrame(
        [
            [
                "Pendekatan",
                (
                    "Agent-Based Modeling dan "
                    "Discrete Event Simulation"
                ),
            ],
            [
                "Agen",
                "Pelanggan bioskop",
            ],
            [
                "Resource",
                "Self-service kiosk",
            ],
            [
                "Kedatangan",
                "Distribusi eksponensial",
            ],
            [
                "Pelayanan",
                "Distribusi lognormal",
            ],
            [
                "Disiplin antrean",
                "First come, first served",
            ],
            [
                "Ketidakpastian",
                (
                    "Monte Carlo dengan "
                    "seed berbeda"
                ),
            ],
        ],
        columns=[
            "Komponen",
            "Implementasi",
        ],
    )

    show_table(
        methodology,
        2,
    )

    st.subheader(
        "Asumsi dan Batasan"
    )

    st.html("""
        1. Setiap pelanggan hanya membutuhkan satu kiosk.
        2. Setiap kiosk hanya melayani satu pelanggan pada satu waktu.
        3. Pelanggan mengikuti antrean *first come, first served*.
        4. Pelanggan meninggalkan antrean setelah batas kesabaran terlewati.
        5. Model belum memasukkan kerusakan kiosk dan variasi jenis transaksi.
        6. Hasil simulasi perlu dikalibrasi dengan data nyata sebelum digunakan untuk keputusan operasional.
        """
    )

    export_summary = (
        st.session_state[
            "comparison_summary"
        ]
    )

    export_raw = (
        st.session_state[
            "comparison_raw"
        ]
    )

    result_narrative = narrative(
        export_summary,
        st.session_state[
            "comparison_iterations"
        ],
    )

    st.subheader(
        "Narasi Hasil"
    )

    st.text_area(
        "Narasi siap laporan",
        result_narrative,
        height=170,
    )

    download_column_1, download_column_2, download_column_3 = (
        st.columns(3)
    )

    with download_column_1:
        st.download_button(
            label=(
                "Download ringkasan CSV"
            ),

            data=(
                export_summary
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            ),

            file_name=(
                "ringkasan_skenario_"
                "kiosk_bioskop.csv"
            ),

            mime="text/csv",

            use_container_width=True,
        )

    with download_column_2:
        if export_raw is None:
            raw_data = b""
            raw_disabled = True

        else:
            raw_data = (
                export_raw
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )

            raw_disabled = False

        st.download_button(
            label=(
                "Download Monte Carlo CSV"
            ),

            data=raw_data,

            file_name=(
                "hasil_monte_carlo_"
                "kiosk_bioskop.csv"
            ),

            mime="text/csv",

            use_container_width=True,

            disabled=raw_disabled,
        )

    with download_column_3:
        report_text = (
            "RINGKASAN SIMULASI "
            "KIOSK BIOSKOP\n\n"

            f"Tanggal ekspor: "
            f"{datetime.now():%d-%m-%Y %H:%M}\n"

            f"Iterasi per skenario: "
            f"{st.session_state['comparison_iterations']}\n\n"

            f"{result_narrative}\n"
        )

        st.download_button(
            label=(
                "Download narasi TXT"
            ),

            data=(
                report_text.encode(
                    "utf-8"
                )
            ),

            file_name=(
                "ringkasan_analisis_"
                "kiosk_bioskop.txt"
            ),

            mime="text/plain",

            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Dashboard Simulasi Antrean
        Self-Service Kiosk Bioskop ·
        hasil awal mengacu pada notebook
        project dengan 100 iterasi Monte Carlo.
    </div>
    """)