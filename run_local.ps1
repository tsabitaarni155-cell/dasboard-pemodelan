$ErrorActionPreference = "Stop"

if (
    Get-Command py
    -ErrorAction SilentlyContinue
) {
    $PythonCommand = "py"
}
elseif (
    Get-Command python
    -ErrorAction SilentlyContinue
) {
    $PythonCommand = "python"
}
else {
    throw (
        "Python tidak ditemukan. " +
        "Instal Python terlebih dahulu dan " +
        "aktifkan Add Python to PATH."
    )
}

if (
    -not (
        Test-Path ".venv"
    )
) {
    Write-Host (
        "Membuat virtual environment..."
    ) -ForegroundColor Yellow

    & $PythonCommand -m venv .venv
}

$VirtualPython = Join-Path `
    (Get-Location) `
    ".venv\Scripts\python.exe"

if (
    -not (
        Test-Path $VirtualPython
    )
) {
    throw (
        "Virtual environment gagal dibuat."
    )
}

Write-Host (
    "Memperbarui pip..."
) -ForegroundColor Yellow

& $VirtualPython `
    -m pip `
    install `
    --upgrade pip

Write-Host (
    "Menginstal dependency..."
) -ForegroundColor Yellow

& $VirtualPython `
    -m pip `
    install `
    -r requirements.txt

Write-Host (
    "Menjalankan dashboard..."
) -ForegroundColor Green

& $VirtualPython `
    -m streamlit `
    run `
    streamlit_app.py