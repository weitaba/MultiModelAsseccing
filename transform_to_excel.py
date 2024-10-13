# -- coding: utf-8 --
import json
import pandas as pd

data_list1_1 = []
data_list2_1 = []
data_list3_1 = []
data_list1 = []
data_list2 = []
data_list3 = []
with open('pair_output.jsonl', 'r', encoding='utf-8') as f1:
    with open('score_output.jsonl', 'r', encoding='utf-8') as f2:
        with open('batch_output.jsonl', 'r', encoding='utf-8') as f3:
            for line1, line2, line3 in zip(f1, f2, f3):
                data1 = json.loads(line1)
                data2 = json.loads(line2)
                data3 = json.loads(line3)
                data_list1_1.append(data1)
                data_list2_1.append(data2)
                data_list3_1.append(data3)

problem_values1 = [d['problem'] for d in data_list1_1]
problem_values2 = [d['problem'] for d in data_list2_1]
problem_values3 = [d['problem'] for d in data_list3_1]

same_items = [item for item in problem_values1 if item in problem_values2 and item in problem_values3]
for item in same_items:
    print(item)

for data1, data2, data3 in zip(data_list1_1, data_list2_1, data_list3_1):
    if data1['problem'] in same_items:
        data_list1.append(data1)
    if data2['problem'] in same_items:
        data_list2.append(data2)
    if data3['problem'] in same_items:
        data_list3.append(data3)

df1 = pd.DataFrame(data_list1)
df2 = pd.DataFrame(data_list2)
df3 = pd.DataFrame(data_list3)
df1.to_excel('pair_output.xlsx', index=False)
df2.to_excel('score_output.xlsx', index=False)
df3.to_excel('batchoutput.xlsx', index=False)