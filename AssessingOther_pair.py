# -- coding: utf-8 --
import json
from datetime import time
from model_api import *
import time
from argparse import ArgumentParser
from colorama import Fore, Style
from sentence_transformers import SentenceTransformer, util
import re


def retry(attempts=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    print(Fore.RED +"The analysis upon is wrong\n" + Style.RESET_ALL)
            return None
        return wrapper
    return decorator

# 得到模型回答
@retry(3)
def getresult(model, prompt, is_assessed_model=False):
    if is_assessed_model:
        print(Fore.MAGENTA + f"被测模型 {model} 的回答是:\n" + Style.RESET_ALL)
    else:
        print(Fore.BLUE + f"评价模型 {model} 的回答是:\n" + Style.RESET_ALL)
    func = globals().get(model)
    output = func(prompt)
    return output

# 得到模型对比结果
@retry(3)
def getscore(model, instruction, answer_A, answer_B):
    print(Fore.GREEN + f"评价模型 {model} 的分析是:\n" + Style.RESET_ALL)
    prompt =  ("""请作为一个公正的评判者，评估人工智能助手对用户指令的回复质量。
评估步骤：
1.确保你的最终输出为 JSON 格式，仅包含键 “分析” 用于初始响应分析和 “判断” 用于你的最终判断。
2.注意。确保每个键下的内容不包含任何嵌套的 JSON 结构。
3.首先对提供的回复进行详细分析。在 “分析” 部分记录你的全面观察和见解。
4.在分析之后，进入判断阶段，在这个阶段你将根据进行的分析做出明智的决定或结论。在 “判断” 部分给出你的最终判断。
评估方法：
你将看到两个不同助手对相同用户指令的回复。
你的任务是根据它们对用户原始指令的有效遵循程度以及对用户询问的恰当处理程度来评估和比较这些回复。
在键 “判断” 中表明你的决定，仅当助手 A 占优时使用 “[[A]]”，助手 B 占优时使用 “[[B]]”，平局时使用 “[[C]]”，并且不要有任何额外的单词。请仅当助手 A 占优时使用 “[[A]]”，助手 B 占优时使用 “[[B]]”，平局时使用 “[[C]]”，并且不要有任何额外的单词。
注意：
你的评估应该确定助手是否有效地遵循了用户的指令并解决了用户的询问。
在你的评估中，权衡相关性、准确性、全面性、创造性和回复的粒度等因素。
不要让回复的长度影响你的评估。
不要偏袒某些助手的名称或位置。尽可能客观。
以下是输入：
""" +
f"""
[用户指令开始]
{instruction}
[用户指令结束]
[助手 A 的回答开始]
{answer_A}
[助手 A 的回答结束]
[助手 B 的回答开始]
{answer_B}
[助手 B 的回答结束]
""")

    func = globals().get(model)
    output = func(prompt)

    match = re.search(r'判断[\s\S]*?([A-C])', output)
    if match:
        score = match.group(1)
    else:
        raise ValueError("未找到匹配内容")

    if "A" in score:
        return 0
    else:
        return 1

# 得到最佳回答
def getbestanswer(models, answers, problem):
    best_dict = {} # 最佳模型字典

    # 各个模型分别对各个模型的回答两两对比，得到各个模型觉得的最佳回答
    for j in range(len(models)):
        best = 0
        for i in range(len(models) - 1):

            answer_A = answers[best]
            answer_B = answers[i + 1]

            if getscore(models[j], problem, answer_A, answer_B) == 1: # 得到对比结果(pair为0代表A胜出，为1代表B胜出，平局算B胜出)
                best = i + 1

        # 输出模型觉得的最佳回答
        print(Fore.YELLOW + f"模型 \"{models[j]}\" 认为回答得最好的评价模型是： \"{models[best]}\".\n该模型的回答为 : \"{answers[best]}\"\n" + Style.RESET_ALL)

        # 累计各个模型觉得最佳的回答，最后再投票得出最好的回答
        if models[best] in best_dict.keys():
            best_dict[models[best]] += 1
        else:
            best_dict[models[best]] = 1

    # 输出投票结果
    print(f" {len(models)} 个评估模型的投票结果为: \"{str(best_dict).replace("{","").replace("}","")}\"\n")

    # 最优模型的回答
    best_model = [key for key,value in best_dict.items() if value == max(best_dict.values())][0] # 得到最佳模型
    index = models.index(best_model) # 得到最佳模型的下标
    best_answer = answers[index] # 得到最佳模型的回答

    return best_answer, best_model

# 得到相似度（句子相似度）
# @retry()
def getsimilarity_sentence(best_answer, assess_output):
    embeddings1 = sentence_model.encode(best_answer, convert_to_tensor=True)
    embeddings2 = sentence_model.encode(assess_output, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    # 根据相似度设定评价分数范围
    score = cosine_scores * 100  # 假设将相似度映射到 0-100 的分数范围
    return score

def main(models, assess_models, problem, correct_answer):

    time_start = time.time()  # 开始时间

    # 被评估模型的输出
    assess_answers = []
    for assess_model in assess_models:
        assess_output = getresult(assess_model, problem, is_assessed_model=True) # 被评估模型的输出
        assess_answers.append(assess_output)

    # 所有评估模型的输出，回答记录在answers
    answers = [] # 答案列表
    for model in models:
        output = getresult(model, problem) # 评估模型的输出
        answers.append(output)

    # 得到最佳模型的回答
    best_answer, best_model = getbestanswer(models, answers, problem)

    # 得到被评估模型的评分
    score_of_answers = []
    for assess_answer in assess_answers:

        score_of_answer = getsimilarity_sentence(best_answer, assess_answer)
        score_of_answers.append(score_of_answer)

        print(Fore.CYAN + f"根据在{len(models)}个评价模型投票得到的最优回答 \"{best_answer}\"\n被评估模型 {assess_model} 的回答 \"{assess_answer}\" 评分为 {score_of_answer}" + Style.RESET_ALL)

    time_end = time.time() # 结束时间

    spend_time = time_end - time_start # 计算花费时间

    intput_json = {"problem": problem, "answers": assess_answers, "score": score_of_answers, "correct": correct_answer, "time": spend_time, "standard_model": best_model, "standard_answer": best_answer}

    print(Fore.LIGHTBLUE_EX + f"\n花费时间：{spend_time}\n" + Style.RESET_ALL)

    with open('pair_output.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(intput_json, ensure_ascii=False) + '\n')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--models", type=str, default=None, nargs='+', help="评估模型列表")
    parser.add_argument("--assess_models", type=str, default=None, nargs='+',help="被评估模型")
    parser.add_argument("--problem", type=str, default=None, help="问题")
    args = parser.parse_args()

    # 假设记录位置的文件名为 position.txt
    position_file = 'pair_position.txt'
    sentence_model = SentenceTransformer('./paraphrase-MiniLM-L6-v2')

    # 读取上次记录的位置
    try:
        with open(position_file, 'r') as f:
            start_line = int(f.read())
    except ValueError:
        start_line = 0

    with open('dataset.jsonl', 'r', encoding='utf-8') as f:

        # 跳过已经处理过的行
        for _ in range(start_line):
            next(f)

        # 记录当前行的位置
        line_number = start_line

        for line in f:

            line_number += 1

            json_obj = json.loads(line)

            args.problem = json_obj['input_zh'] + json_obj['instruction_zh']

            try:
                main(args.models, args.assess_models, args.problem, correct_answer=json_obj['output_zh'])
            except Exception as e:
                pass

            # 写入已经处理了多少行
            with open(position_file, 'w') as pos_f:
                pos_f.write(str(line_number))