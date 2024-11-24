call activate openvoice
cd open_voice_v2
call uvicorn api:app --host 127.0.0.1  --port 5059 --workers 1 