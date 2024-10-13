# -- coding: utf-8 --
import json
import re
import numpy as np
from model_api import *
from argparse import ArgumentParser
from colorama import Fore, Style
import time


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
def getresult(model, prompt):
    print(Fore.BLUE + f"被评价模型 {model} 的回答是:\n" + Style.RESET_ALL)
    func = globals().get(model)
    output = func(prompt)
    return output

# 得到模型对比结果
@retry(3)
def getscore(model, instruction, assess_output):
    print(Fore.GREEN + f"评价模型 {model} 的分析是:\n" + Style.RESET_ALL)
    prompt =  ("""请作为一个公正的评判者，评估人工智能助手对用户指令。
评估步骤：
1.确保你的最终输出为 JSON 格式，仅包含键 “分析” 用于初始响应分析和 “打分” 用于你的最终判断，不要出现除了JSON以外的其他文字。
2.注意。确保每个键下的内容不包含任何嵌套的 JSON 结构。
3.首先对提供的回复进行详细分析。在 “分析” 部分记录你的全面观察和见解。
4.在分析之后，进入判断阶段，在这个阶段你将根据进行的分析做出明智的决定或结论。在 “打分” 部分给出你的最终判断。
评估方法：
你将看到助手对用户指令的回复。
你的任务是根据它对用户原始指令的有效遵循程度以及对用户询问的恰当处理程度来评估它的回复。
在键 “打分” 中表明你的决定，分数范围为[0-100].输出格式为[[0-100]]，“打分”键对应的值应该是一个整数。
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

[助手回答开始]
{assess_output}
[助手回答结束]
""")

    func = globals().get(model)
    output = func(prompt)

    match = re.search(r'打分[\s\S]*?(\d+)', output)
    if match:
        score = match.group(1)
        return score
    else:
        raise ValueError("未找到匹配内容")


def main(models, assess_models, problem, correct_answer):

    time_start = time.time() # 开始时间

    # 得到被评估模型的评分
    assess_answers = []
    average_scores = []
    each_scores = []
    for assess_model in assess_models:
        assess_output = getresult(assess_model, problem) # 被评估模型的输出
        assess_answers.append(assess_output)

        # 所有评估模型的输出，回答记录在answers
        answers = [] # 答案列表
        for model in models:
            output = getscore(model, problem, assess_output) # 评估模型的输出
            answers.append(int(output))
        each_scores.append(answers)
        average_score = np.mean(answers)
        average_scores.append(average_score)

        for i, model in enumerate(models):
            print(Fore.CYAN + f"模型{models[i]}对评估模型的打分为：{answers[i]}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"\n综上，评估模型的平均分数为：{average_score}" + Style.RESET_ALL)

    time_end = time.time() # 结束时间

    spend_time = time_end - time_start # 计算花费时间

    intput_json = {"problem": problem, "answer": assess_answers, "average_score": average_scores, "correct": correct_answer, "time": spend_time, "models": models, "model_score": each_scores}

    print(Fore.LIGHTBLUE_EX + f"\n花费时间：{spend_time}\n" + Style.RESET_ALL)


    with open('score_output.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(intput_json, ensure_ascii=False) + '\n')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--models", type=str, default=None, nargs='+', help="评估模型列表")
    parser.add_argument("--assess_models", type=str, default=None, nargs='+', help="被评估模型")
    parser.add_argument("--problem", type=str, default=None, help="问题")
    args = parser.parse_args()

    # 假设记录位置的文件名为 position.txt
    position_file = 'score_position.txt'

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




