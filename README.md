# Dashboard Final Simulasi Kiosk Bioskop

Dashboard Streamlit untuk tugas besar **Simulasi Sistem Antrean Self-Service Kiosk Bioskop Menggunakan Agent-Based Modeling, Discrete Event Simulation, dan Monte Carlo**.

## Perbaikan utama

- Membaca hasil final 1.000 iterasi per skenario dari CSV, sehingga tidak menghitung ulang 4.000 simulasi saat aplikasi dibuka.
- Simulasi interaktif dibatasi maksimal 100 iterasi agar stabil pada Streamlit Community Cloud.
- Fungsi simulasi disamakan dengan notebook final.
- Intervensi preventif tidak lagi dihitung dua kali.
- Utilisasi hanya menghitung busy time di dalam horizon operasional.
- Total kedatangan dihitung dari seluruh agen yang benar-benar datang.
- Menyediakan upload CSV/Excel, validasi, pemetaan kolom, analisis, dan download dataset bersih.
- Menggunakan `width="stretch"` untuk komponen data dan Plotly.
- Tidak menggunakan ngrok, token, Graphviz, atau `packages.txt`.

## Struktur repository

```text
.
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    ├── dataset_pelanggan_kiosk_bioskop.csv
    ├── dataset_jejak_antrean_per_menit.csv
    ├── hasil_monte_carlo_1000_iterasi.csv
    ├── ringkasan_skenario_1000_iterasi.csv
    ├── hasil_uji_statistik.csv
    ├── ringkasan_optimasi_kiosk.csv
    ├── hasil_uji_kondisi_ekstrem.csv
    ├── hasil_sensitivitas_interval_kedatangan.csv
    ├── kamus_data.csv
    └── parameter_skenario.csv
```

## Menjalankan secara lokal

```powershell
cd "C:\Users\tsabi\Downloads\ABM_CBT_Dashboard_Akademik"
py -m pip install -r requirements.txt
py -m streamlit run streamlit_app.py
```

## Deploy ke Streamlit Community Cloud

- Repository: `tsabitaarni155-cell/dasboard-pemodelan`
- Branch: `main`
- Main file path: `streamlit_app.py`

Setelah semua file disalin ke repository:

```powershell
git add -A
git commit -m "Update final dashboard kiosk bioskop"
git push
```

Streamlit biasanya melakukan deploy ulang otomatis. Gunakan **Manage app → Reboot app** apabila pembaruan belum tampil.

## Dataset upload

Dataset lain dapat menggunakan nama kolom yang sama ataupun berbeda. Pada tab **Upload & Validasi Dataset**, pengguna dapat memetakan kolom sumber ke kolom standar seperti `arrival_time`, `start_service_time`, `departure_time`, `wait_time`, `service_time`, `system_time`, `state`, dan `scenario`.
