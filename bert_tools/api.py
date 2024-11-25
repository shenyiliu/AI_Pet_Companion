import os
import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Text
from predict import Predict


now_dir = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()
# pytorch
# predict = Predict(os.path.join("out_model", "best.pth"))
# openvino
model_path1 = os.path.join(now_dir, "out_model", "bert_ov", "bert.xml")
predict = Predict(model_path1, ov_device="NPU")


class Data(BaseModel):
    text_list: List[Text]
    threshold: float = 0.9


@app.post("/tool_classify/")
def tool_classify(data: Data):
    """
    工具调用分类<br>
    :param data 参数说明如下：<br>
        text_list: 可以一次性输入多个待分类文本<br> 
        threshold: 阀值，超过这个阀值的数据才认为是正确分类。默认0.8<br> 
    """
    st = time.time()
    data_list = predict.predict(data.text_list, data.threshold)
    et = time.time()
    during = round(et - st, 4)
    return {"data": data_list, "during": during}


if __name__ == '__main__':
    uvicorn.run(
        app='web:app',
        host="127.0.0.1",
        port=5518,
        reload=False,
        debug=False,
        workers=1,
    )
    """
    uvicorn api:app --host 127.0.0.1  --port 5518 --workers 1
    """
