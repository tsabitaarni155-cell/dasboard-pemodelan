# Dashboard Akademik ABM CBT Kecemasan

Dashboard Streamlit untuk tugas besar Pemodelan dan Simulasi menggunakan Agent-Based Modeling, skenario intervensi CBT, dan eksperimen Monte Carlo.

## Fitur Utama

- Empat skenario what-if.
- State chart Tenang, Cemas Ringan, Cemas Tinggi, Panik, dan Pulih.
- Parameter A, S, D, R, dan P.
- Monte Carlo sampai 1000 iterasi.
- Grafik dinamika kecemasan.
- Grafik persentase agen panik.
- Komposisi state agen.
- Ranking skenario.
- Skor stabilitas.
- Snapshot grid agen.
- Distribusi atribut agen.
- Analisis sensitivitas parameter.
- Export CSV.
- Narasi metodologi.
- Kesimpulan otomatis.
- Uraian luaran HKI.

## Menjalankan pada Windows

Buka PowerShell pada folder project, kemudian jalankan:

    Set-ExecutionPolicy -Scope Process Bypass

Selanjutnya jalankan:

    .\run_local.ps1

Dashboard akan terbuka melalui browser pada alamat lokal Streamlit.

## Deploy ke Streamlit Community Cloud

1. Upload seluruh isi project ke repository GitHub.
2. Masuk ke Streamlit Community Cloud.
3. Pilih repository dan branch main.
4. Gunakan streamlit_app.py sebagai Main file path.
5. Klik Deploy.

Ngrok tidak diperlukan untuk deployment permanen pada Streamlit Community Cloud.

## Struktur Project

    ABM_CBT_Dashboard_Akademik
    ├── .streamlit
    │   └── config.toml
    ├── data
    │   └── .gitkeep
    ├── outputs
    │   └── .gitkeep
    ├── streamlit_app.py
    ├── requirements.txt
    ├── packages.txt
    ├── run_local.ps1
    ├── README.md
    └── .gitignore

## Catatan Etis

Aplikasi ini merupakan simulasi edukatif. Aplikasi tidak digunakan sebagai alat diagnosis klinis dan bukan pengganti psikolog atau psikiater.
