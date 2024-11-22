import os
import time
import uvicorn
from fastapi import FastAPI
import math
from pydantic import BaseModel
from typing import List, Text
from fastapi.responses import HTMLResponse
from predict import Predict
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.mount('/static', StaticFiles(directory="static"), name='static')
english_parse = Predict("out_model/best.pth")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/")
async def root():
    """
    根路径，请忽略该接口<br>
    :return:
    """
    with open(os.path.join('templates', 'index.html'), encoding='utf-8') as f:
        str1 = f.read()
    return HTMLResponse(content=str1, status_code=200)


class Data(BaseModel):
    text_list: List[Text]
    is_one_intent = False
    is_strip = False
    batch_size = 32


@app.post("/academic_requirement/")
def academic_requirement(data: Data):
    """
学术要求文本解析
:param data 参数说明如下：<br>
:text_list: list类型，包含多个学术要求的字段，每个字段可以包含多种类型的学术要求，建议每个课程一个<br>
:is_one_intent: 是否为单分类，默认是False, 也就是会输出多分类。<br>
    单分类情况下，每个text只输出一种score或者score_type，默认为最低分数的那个<br>
    例如 A-LEVEL ABB-BBC, 则输出score_type: BBC, IB 38-43,则只输出38. <br>
:is_strip: 是否剔除不包含实体的多余信息，。比如一个学术要求里面夹杂多余的英语要求,<br>
    则会先根据句号分号切割，对于不含实体的句子则进行剔除。<br>
:batch_size: 批处理数据量，默认为32，设置越大则速度越快，一般为2的指数，如1，2， 4， 8，最大不应超过64，否则可能会导致显存溢出, 最小不低于1<br>
输入示例：<br>
<div class="highlight-code">
<pre class="body-param__example microlight" style="display: block; overflow-x: auto; padding: 0.5em; background: rgb(51, 51, 51); color: white;">
<code class="language-json" style="white-space: pre;">
{
  "text_list": [
  "IELTS: 6.0 overall (minimum 5.5 in any component)",
   "International Baccalaureate Diploma Programme - 29 points",
   "A strong upper second-class honours (65% or above) degree (or international equivalent) in finance, accounting, economics, business, mathematics or another quantitative subject, such as engineering, physics or computing. Evidence of strong performance in Economics and Mathematics related modules will also be required."
  ],
  "is_one_intent": false,
  "is_strip": false
}
</code></pre></div>
<br>
返回示例：<br>
<div class="highlight-code">
<pre class="body-param__example microlight" style="display: block; overflow-x: auto; padding: 0.5em; background: rgb(51, 51, 51); color: white;">
<code class="language-json" style="white-space: pre;">
[
  {
    "text": "IELTS: 6.0 overall (minimum 5.5 in any component)",
    "result": [
      {
        "score_type": null,
        "score": null,
        "condition_type": -100
      }
    ]
  },
  {
    "text": "International Baccalaureate Diploma Programme - 29 points",
    "result": [
      {
        "score_type": null,
        "score": 29,
        "condition_type": 4
      }
    ]
  },
  {
    "text": "A strong (65% or above) upper second-class honours degree (or international equivalent) in Economics, Mathematics, Physics, Acturarial Sciences, Engineering or other quantitative based subjects.",
    "result": [
      {
        "score_type": "2:1",
        "score": null,
        "condition_type": 11
      },
      {
        "score_type": null,
        "score": 65,
        "condition_type": 18
      }
    ]
  }
]
</code></pre></div>
<br>注：label_result为标注结果返回, score_result为转行成业务数据格式后的信息<br>
-100代表该文本不属于学术要求。
-200代表未知类型，一般是漏了，很少出现，出现了请联系作者。
    """
    # print(data)
    text_list = data.text_list
    is_one_intent = data.is_one_intent
    is_strip = data.is_strip
    # batch_size取整
    batch_size = max(min(data.batch_size, 64), 1)
    batch_size = 2 ** int(math.log2(batch_size))
    st = time.time()
    # 增加一个dict做去重映射
    text_list2 = list(set(text_list))
    data_list = english_parse.predict(
        text_list2, is_one_intent, is_strip, batch_size)
    text_dict = {text2: data for text2, data in zip(text_list2, data_list)}
    data_list2 = [text_dict[text] for text in text_list]
    et = time.time()
    during = round(et - st, 4)
    return {"data": data_list2, "during": during}



if __name__ == '__main__':
    uvicorn.run(
        app='web:app',
        host="0.0.0.0",
        port=5518,
        reload=True,
        debug=True,
        workers=1,
        log_config="log_conf.yaml"
    )
