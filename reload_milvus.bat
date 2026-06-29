@echo off
chcp 65001 >nul
echo ========================================
echo   Reload Embeddings & Milvus Data
echo ========================================
echo.

echo [1/2] Dang tao embeddings moi...
python create_embeddings.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Tao embeddings that bai!
    pause
    exit /b 1
)
echo [OK] Embeddings da tao xong
echo.

echo [2/2] Dang load vao Milvus (xoa data cu)...
python load_to_milvus.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Load vao Milvus that bai!
    pause
    exit /b 1
)
echo [OK] Da load xong
echo.

echo ========================================
echo   Hoan tat! Restart app de su dung
echo ========================================
echo.
pause
