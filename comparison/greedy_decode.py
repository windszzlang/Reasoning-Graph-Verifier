import json
from sklearn.metrics import accuracy_score



dataset = 'GSM8K'
# dataset = 'SVAMP'
# dataset = 'ASDiv-a'


pred_answers = []
gold_answers = []
with open('data/' + dataset + '.jsonl') as f:
    for line in f.readlines():
        obj = json.loads(line)
        pred_answer = list(obj['gpt_solutions'].keys())[0]
        gold_answer = obj['answer']
        pred_answers.append(pred_answer)
        gold_answers.append(gold_answer)


right = 0
all_num = len(gold_answers)
for pred, gold in zip(pred_answers, gold_answers):
    if pred == gold:
        right += 1

acc = right / all_num
print(acc)