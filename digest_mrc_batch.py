import re
import time
import traceback
from loguru import logger
from typing import Dict

import torch
from flask import Flask, request
from fastapi import HTTPException, status
import numpy as np
from openai import OpenAI
from text2vec import SentenceModel
from sentence_transformers import SentenceTransformer, util

SYSTEM_PROMPT = '你是一个阅读理解和摘抄大师，现在给你一篇文章和一个用户关心的问题，你需要从文章中找出所有能够回答该问题的片段。'
POST_PROMPT = '''
请从原文中摘抄出能够回答这个问题的片段，重复此过程直到你摘抄出所有能够回答问题的片段。如果原文中没有能够直接回答问题的语句，你也应该回答与问题有关的片段。以如下json格式回答：
[
    {
        "片段": "片段1",
    },
    ...
]
示例1：
文章：时不时长个痘，难受不？该不该挤呢？\n\n\n\n实话实说，痘痘的发病率是有明显的地域差别。跟人们普遍认知不同的是，南方的爆痘率要明显高于北方。\n\n\n\n华南地区更是长痘的重灾区。这是因为湿度过高、气温炎热的广州、三亚等地，人们的皮脂分泌通常也较旺盛，皮肤相对油腻、容易爆痘。\n\n\n\n那么，全国最不长痘的地方在哪里呢？\n\n\n\n依旧是川渝所在的西南地区，不愧“天府之国”啊！
问题：什么地区的人皮肤容易长痘？
回答：
[
    {
        "片段": "南方的爆痘率要明显高于北方。",
    },
    {
        "片段": "华南地区更是长痘的重灾区。这是因为湿度过高、气温炎热的广州、三亚等地，人们的皮脂分泌通常也较旺盛，皮肤相对油腻、容易爆痘。",
    }
]
示例2：
文章：锣鼓经是大陆传统器乐及戏曲里面常用的打击乐记谱方法，以中文字的声音模拟敲击乐的声音，纪录打击乐的各种不同的演奏方法。常用的节奏型称为「锣鼓点」。而锣鼓是戏曲节奏的支柱，除了加强演员身段动作的节奏感，也作为音乐的引子和尾声，提示音乐的板式和速度，以及作为唱腔和念白的伴奏，令诗句的韵律更加抑扬顿锉，段落分明。
问题：锣鼓经是什么?
回答：
[
    {
        "片段": "锣鼓经是大陆传统器乐及戏曲里面常用的打击乐记谱方法，以中文字的声音模拟敲击乐的声音，纪录打击乐的各种不同的演奏方法。",
    }
]
示例3：
文章：锣鼓经是大陆传统器乐及戏曲里面常用的打击乐记谱方法，以中文字的声音模拟敲击乐的声音，纪录打击乐的各种不同的演奏方法。常用的节奏型称为「锣鼓点」。而锣鼓是戏曲节奏的支柱，除了加强演员身段动作的节奏感，也作为音乐的引子和尾声，提示音乐的板式和速度，以及作为唱腔和念白的伴奏，令诗句的韵律更加抑扬顿锉，段落分明。
问题：锣鼓在戏曲中有什么作用?
回答：
[
    {
        "片段": "而锣鼓是戏曲节奏的支柱，除了加强演员身段动作的节奏感，也作为音乐的引子和尾声，提示音乐的板式和速度，以及作为唱腔和念白的伴奏，令诗句的韵律更加抑扬顿锉，段落分明。",
    }
]
现在请你做出回答：
'''


def parse_answer(ans):
    ans = ans.split("\"片段\":")[-1][1:-1].strip()[1:-2]
    # if args.debug: logger.info(f'片段为：{ans}')
    return ans


app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"device:{device}")

app_ready = False


retrival_model = SentenceTransformer("./opt/xiaobu")
bert_model = SentenceModel("./opt/M3E-large")


def inference_batch(system_prompt, prompts):
    client = OpenAI(api_key="")
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([{"role": "user", "content": prompt} for prompt in prompts])

    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        top_p=0.5,
        stop=None,
        n=len(prompts)  # Specify the number of completions you want
    )

    preds = [choice.message.content for choice in res.choices]
    logger.info("Inference batch finished")
    return preds

def retry_n_times_batch(question, context, n):
    system_prompt = SYSTEM_PROMPT
    prompt = f'文章：{context}\n\n问题：{question}\n\n' + POST_PROMPT
    prompts = [prompt] * n
    try:
        pred_mids = inference_batch(system_prompt, prompts)
    except Exception as e:
        logger.error(f'错误信息：{e}, {traceback.format_exc()}')
        return []

    predictions = []
    json_pattern = r"{.*?}"
    for pred_mid in pred_mids:
        pred_mid = re.findall(json_pattern, pred_mid, re.DOTALL)
        logger.info(pred_mid)
        pred = ''
        for answer in pred_mid:
            seg = parse_answer(answer)
            pred += f"{seg}\n"
        predictions.append(pred)
    logger.info("Request finished")
    return predictions

def select_best_retrival(question, predictions):
    # 生成查询和检索结果的向量
    query_embedding = retrival_model.encode(question, normalize_embeddings=True)
    retrieval_embeddings = retrival_model.encode(predictions, normalize_embeddings=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, retrieval_embeddings)

    # 将相似度分数转换为 numpy 数组
    cosine_scores = cosine_scores.numpy()

    # 输出每个检索结果的相似度分数
    for i, score in enumerate(cosine_scores[0]):
        logger.info(f"Retrieval结果 {i + 1}: '{i+1}' - Similarity score: {score}")

    # 返回相似度分数最高的答案
    index = np.argmax(cosine_scores[0])
    logger.info(f"Retrieval结果: {predictions[index]}")
    return predictions[index]


def extract_passage(question, context):
    try:
        preds = retry_n_times_batch(question, context, 3)
    except Exception as e:
        logger.error(f'错误信息：{e}, {traceback.format_exc()}')
        return [{"start": 0, "end": len(context) - 1}]

    pred = select_best_retrival(question, preds)
    splitters = r"[，；。！\n]"
    answer_parts = re.split(splitters, pred)
    answer_parts = [each for each in answer_parts if each]
    answer_emb = bert_model.encode(answer_parts)

    context_parts = re.split(splitters, context)
    context_parts = [each for each in context_parts if each]
    context_emb = bert_model.encode(context_parts)

    res = []
    for i, part in enumerate(answer_parts):
        if part in context:
            start = context.find(part)
            res.append([start, min(start + len(part) + 1, len(context) - 1)])
        else:
            logger.info('未找到原文，使用embedding匹配')
            candidate_score = np.dot(answer_emb[i], context_emb.T)
            target_start = context.find(context_parts[np.argmax(candidate_score)])
            res.append([target_start,
                        min(target_start + len(context_parts[np.argmax(candidate_score)]) + 1, len(context) - 1)])

    res = sorted(res, key=lambda x: x[0])
    merged_res = []
    for each in res:
        if not merged_res or merged_res[-1][1] + 8 < each[0]:
            merged_res.append(each)
        else:
            merged_res[-1][1] = max(merged_res[-1][1], each[1])

    res = [{"start": each[0], "end": each[1]} for each in merged_res]
    res_text = [context[each[0]:each[1]] for each in merged_res]
    logger.info(f'最终回答片段：{res_text}')

    if not res:
        logger.warning('未找到匹配语句')
        res = [{"start": 0, "end": len(context) - 1}]

    return res


@app.route("/predict", methods=["POST"])
def predict_batch():
    logger.info("Predict begin")
    start_time = time.time()
    data = request.get_json()
    question = data['query']
    context = data['segment']
    response = extract_passage(question, context)
    logger.info(f'总用时：{time.time() - start_time:.2f}s')
    return {"data": response}


@app.get('/ready')
def check_model_ready() -> Dict[str, bool]:
    if app_ready:
        return {"ready": True}
    else:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="App is not ready yet.")

@app.route("/predict", methods=["POST"])
def predict():
    logger.info("predict begin")
    start_time = time.time()
    question = request.get_json()['query']
    context = request.get_json()['segment']
    response = extract_passage(question, context)

    result = response
    logger.info(f'总用时：{time.time() - start_time:.2f}s')
    return {"data": result}


if __name__ == '__main__':
    PORT = 80
    app_ready = True
    app.run("0.0.0.0", PORT)