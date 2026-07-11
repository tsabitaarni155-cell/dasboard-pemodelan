# Dashboard Simulasi Self-Service Kiosk Bioskop

Dashboard ini merupakan luaran Tugas Besar Pemodelan dan Simulasi dengan objek sistem antrean self-service kiosk pada bioskop.

## Pendekatan

Project menggabungkan:

- Agent-Based Modeling
- Discrete Event Simulation
- Monte Carlo Simulation
- What-If Analysis
- Capacity Optimization
- Sensitivity Analysis

## Skenario

1. Tanpa Intervensi
2. Intervensi Reaktif
3. Intervensi Preventif
4. Beban Tinggi

## Fitur Dashboard

- Hasil acuan dari notebook project
- Perbandingan empat skenario
- Simulasi parameter interaktif
- Grafik panjang antrean
- Distribusi waktu tunggu
- Boxplot Monte Carlo
- Optimasi jumlah kiosk
- Analisis sensitivitas
- Rekomendasi kapasitas
- Narasi hasil otomatis
- Download ringkasan CSV
- Download data Monte Carlo
- Download narasi laporan

## Menjalankan Secara Lokal

Buka PowerShell pada folder project, lalu jalankan:

    Set-ExecutionPolicy -Scope Process Bypass
    .\run_local.ps1

Dashboard akan terbuka melalui:

    http://localhost:8501

## Deploy ke Streamlit Community Cloud

Gunakan konfigurasi:

- Repository: tsabitaarni155-cell/dasboard-pemodelan
- Branch: main
- Main file path: streamlit_app.py

Project ini tidak menggunakan ngrok dan tidak membutuhkan packages.txt.

## Catatan

Hasil default pada dashboard mengacu pada eksperimen notebook sebanyak 100 iterasi Monte Carlo per skenario. Pengguna dapat menjalankan ulang eksperimen dari dashboard dengan parameter yang berbeda.