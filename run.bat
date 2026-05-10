@echo off
call .venv\Scripts\activate
uvicorn backend.api:app --reload
pause