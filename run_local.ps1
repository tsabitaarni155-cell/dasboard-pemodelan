$ErrorActionPreference = "Stop"

if (
    Get-Command py
    -ErrorAction SilentlyContinue
) {
    $Python = "py"
}
elseif (
    Get-Command python
    -ErrorAction SilentlyContinue
) {
    $Python = "python"
}
else {
    throw (
        "Python tidak ditemukan. " +
        "Instal Python 3.10 atau lebih baru " +
        "dan aktifkan Add Python to PATH."
    )
}

if (-not (Test-Path ".venv")) {
    Write-Host (
        "Membuat virtual environment..."
    ) -ForegroundColor Yellow

    & $Python -m venv .venv
}

$VenvPython = Join-Path (
    Get-Location
) ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw (
        "Virtual environment gagal dibuat."
    )
}

Write-Host (
    "Memperbarui pip..."
) -ForegroundColor Yellow

& $VenvPython -m pip install --upgrade pip

Write-Host (
    "Menginstal dependency..."
) -ForegroundColor Yellow

& $VenvPython -m pip install -r requirements.txt

Write-Host (
    "Menjalankan Dashboard Streamlit..."
) -ForegroundColor Green

& $VenvPython -m streamlit run streamlit_app.py
