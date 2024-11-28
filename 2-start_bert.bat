call activate openvino
cd bert_tools
call uvicorn api:app --host 127.0.0.1  --port 5518 --workers 1
