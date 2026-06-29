@echo off
title UIT RAG System
cd /d "%~dp0"
echo Starting UIT RAG System...
echo.
streamlit run app.py >nul 2>&1
