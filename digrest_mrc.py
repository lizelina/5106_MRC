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


def inference(system_prompt, prompt):
    client = OpenAI(api_key="sk-proj-cIdy9MVXhObTWS4UETvAT3BlbkFJS9SOWQLeLtp2Hr1fIDHS")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024,
        top_p=0.5,
        stop=None,
    )

    pred = res.choices[0].message.content
    logger.info(f"--------------------inference finished--------------------")
    return pred


def retry_n_times(question, context, n):
    system_prompt = SYSTEM_PROMPT
    prompt = f'文章：{context}\n\n问题：{question}\n\n'
    # 重复n次返回结果
    prompt += POST_PROMPT
    pred = ''
    predictions = []
    count = 0
    while count < n:
        try:
            count += 1
            pred_mid = inference(system_prompt, prompt).strip()
        except Exception as e:
            logger.error(f'错误信息：{e}, {traceback.format_exc()}')
            continue

        json_pattern = r"{.*?}"
        pred_mid = re.findall(json_pattern, pred_mid, re.DOTALL)
        # if args.debug:
        logger.info(pred_mid)

        for answer in pred_mid:
            seg = parse_answer(answer)
            pred += f"{seg}\n"
        predictions.append(pred)
    logger.info(f"request finished")

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
    prompt = '文章：{context}\n\n问题：{question}\n\n'.format(context=context, question=question)
    prompt += POST_PROMPT
    try:
        preds = retry_n_times(question, context, 3)
    except Exception as e:
        logger.error(f'错误信息：{e}, {traceback.format_exc()}')
        ans = [{"start": 0, "end": len(context) - 1}]
        return ans
    pred = select_best_retrival(question, preds)
    splitters = r"[，；。！\n]"  # ，；。！\n
    answer_parts = re.split(splitters, pred)
    # if args.debug: logger.info(f"匹配原文为：{pred}，分割后为：{answer_parts}")
    answer_parts = [each for each in answer_parts if len(each) != 0]
    answer_emb = bert_model.encode(answer_parts)

    context_parts = re.split(splitters, context)
    context_parts = [each for each in context_parts if len(each) != 0]
    # if args.debug: logger.info(context_parts)
    context_emb = bert_model.encode(context_parts)
    norm2 = np.linalg.norm(context_emb, axis=1)

    res = []
    for i in range(len(answer_parts)):
        if answer_parts[i] in context:
            start = context.find(answer_parts[i])
            res.append([start, min(start + len(answer_parts[i]) + 1, len(context) - 1)])
        else:
            logger.info(f'未找到原文，使用embedding匹配')
            candidate_score = np.dot(answer_emb[i], context_emb.T)
            norm1 = np.linalg.norm(answer_emb[i])
            candidate_score = candidate_score / (norm1 * norm2)
            target_start = context.find(context_parts[np.argmax(candidate_score)])
            res.append([target_start,
                        min(target_start + len(context_parts[np.argmax(candidate_score)]) + 1, len(context) - 1)])
    res = sorted(res, key=lambda x: x[0])

    merged_res = []
    for each in res:
        if len(merged_res) == 0 or merged_res[-1][1] + 8 < each[0]:
            merged_res.append(each)
        else:
            merged_res[-1][1] = max(merged_res[-1][1], each[1])

    res = [{"start": each[0], "end": each[1]} for each in merged_res]
    # if args.debug:
    res_text = [context[each[0]:each[1]] for each in merged_res]
    logger.info(f'最终回答片段：{res_text}')

    logger.info(f"res finished")

    ans = res

    if len(ans) == 0:
        logger.warning('未找到匹配语句')
        ans = [{"start": 0, "end": len(context) - 1}]

    return ans

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


def predict2():
    logger.info("predict begin")
    start_time = time.time()
    examples = {
        "query": "什么地区的人皮肤容易长痘？",
        "segment": "青藏高原之后，紧接着是云贵高原、内蒙古高原以及西北内陆地区太阳辐射量较高；而华东、华南地区辐照度较低，最低者便是文章开头提到的四川盆地。\n\n\n\n下面是按照民族和地区对人群肤色进行的分析：普遍规律是：\n\n生活在辐射高地区的人群肤色深\n\n生活在辐射低地区的人群肤色浅\n\n\n\n如果吸收太多紫外线辐射，会让皮肤皱纹增加，变得粗糙、松垂、缺乏弹性。过强紫外线照射还会导致皮肤不均匀地增厚和变薄，常见的阳光引起的皮肤色素变化是雀斑和晒斑，这两者都是长时间暴露在日光下导致的。\n\n\n\n像是海南三亚的太阳紫外线辐射是沈阳的两倍，但三亚男性皮肤老化的风险是沈阳男性的约6倍，女性则达到了惊人的11倍。\n\n\n\n\n\n\n\n\n\n当然了地理君温馨提醒：\n\n以上说的都是普遍性，假如你生活在四川盆地皮肤却很黑，那只能说明你：没做好防晒！\n\n\n气候湿润地区的高颜值“密码”\n\n\n\n除了皮肤颜色外，令人羡慕的优质皮肤还需满足两个条件：一是皮肤的含水量要足够多；二是皮肤出油率要保持较低水平。\n\n\n\n皮肤的含水量跟湿度密切相关，湿度过低，人体皮肤因缺少水分而变得粗糙甚至开裂，人体的免疫系统也会受到伤害，对疾病的抵抗力大大降低甚至丧失；\n\n\n\n而湿度过高时，并且如果长久呆在潮湿空气中，人体皮肤会出现少量细菌感染或者皮肤发痒等现象。所以长时间呆在湿度过高或者过低的环境里都是不利于皮肤的发育。\n\n\n\n\n通常情况下，人体皮肤所需的舒适气温在10~22℃之间，这相当于皮肤所需的舒适湿度在25%~45%之间。\n\n\n\n1月是冬季的典型代表，皮肤最喜欢的相对湿度45%等值线大体沿25°N分布，也就是南岭以南的华南地区。\n\n\n\n\n5月是春季的代表，此时也是全国湿度的舒适范围达到最大的月份，除了东北北部和青藏高原腹地外，全国都在舒适带范围内。\n\n\n\n7月是夏季的代表，“热”字当头，除了东北北部和青藏高原外，全国其它地区都不在舒适带范围内。\n\n\n\n\n以9月、10月为主的秋季，全国舒适带的范围和春季的分布相差无几。\n\n\n\n那么哪个地区相对而言湿度要更大一些呢？\n\n\n\n答案是：川渝地区仍然榜上有名。\n\n\n\n总体上，根据2021年《中国统计年鉴》的数据，西南地区的平均相对湿度更舒适一些，这样的湿度更利于形成或维持水润的皮肤。\n\n\n\n\n\n\n\n\n\n地理君再次温馨提醒：\n\n皮肤适宜的舒适湿度同样具有普遍性，假如你生活在川渝地区皮肤却很干或者皱纹很多，那说明你：有可能是吸烟人群。\n\n图片\n\n\n\n\n\n\n除此以外，由于地域辽阔，自然环境特征复杂多样，普遍性中往往有很多的个例。\n\n\n\n例如，长春的冬季漫长、寒冷干燥，气候条件比很多南方城市还要恶劣，但正是因为冬季严寒，当地人们反而在户外活动的时间变少，防寒保暖的措施更加牢固，因此长春人皮肤的水分状况反而要比江苏、湖南等地的人们的皮肤好。\n\n\n\n吸烟有害健康大家都知道，但是吸烟对皮肤的伤害也很大，甚至超过了阳光带给皮肤的伤害。\n\n\n\n吸烟会导致皮肤的新陈代谢功能降低，易于发生皮肤老化。还会降低人体胶原合成的减少，易于皮肤产生皱纹。\n\n\n\n\n当今社会，中国的平均吸烟率在25%左右，但是男女吸烟率差距极大，男性吸烟率接近50%，女性吸烟率仅3.1%左右。但是15岁以上人群中吸烟者在2018年便超过了总吸烟人数的四分之一。\n\n\n\n一篇发表在《柳叶刀》的文章研究称：我国最喜欢吸烟的人群主要分布在西北地区，比如宁夏银川人吸烟率就高达49.8%。\n\n\n\n而南方一些省市的吸烟率，例如贵州省超过了40%，江浙沪、云南四川等地对烟草的依赖则最低，总体上是北方多于南方。\n\n\n\n\n除了吸烟会影响皮肤状况外，皮肤出油率高、经常爆痘也是其重要影响因素。\n\n\n\n时不时长个痘，难受不？该不该挤呢？\n\n\n\n实话实说，痘痘的发病率是有明显的地域差别。跟人们普遍认知不同的是，南方的爆痘率要明显高于北方。\n\n\n\n华南地区更是长痘的重灾区。这是因为湿度过高、气温炎热的广州、三亚等地，人们的皮脂分泌通常也较旺盛，皮肤相对油腻、容易爆痘。\n\n\n\n那么，全国最不长痘的地方在哪里呢？\n\n\n\n依旧是川渝所在的西南地区，不愧“天府之国”啊！\n\n\n\n不止在爆痘率上，出油率南方也是高于北方地区。北方气候比较冷，空气干燥。皮肤毛孔收缩，保持皮肤的含水量，皮脂腺分泌降低，皮肤内的水分也用来保持皮肤湿润，因此皮肤油脂少。\n\n\n\n而南方空气相较于北方要暖和且空气湿润，因此皮肤新陈代谢快，皮脂腺分泌旺盛，就导致了皮肤容易出油。\n\n\n\n其实出油并不代表皮肤水分充足，恰恰相反，正是由于人体皮肤干燥缺水，导致肌肤受损，人体自我修复和保护系统才会分泌油脂来保护肌肤！所以即使在南方也要做到皮肤及时补水。\n\n\n\n一个地区的人的皮肤状况和自然地理环境有一定联系，而且这种联系也是潜移默化的，但是随着人口流动的增加，导致基因多元化，所以二者联系越来越小"
    }
    question = examples['query']
    context = examples['segment']
    response = extract_passage(question, context)

    result = response
    logger.info(f'总用时：{time.time() - start_time:.2f}s')
    return {"data": result}

if __name__ == '__main__':
    PORT = 80
    app_ready = True
    app.run("0.0.0.0", PORT)
    #     res = predict2()
    #     print(res)
