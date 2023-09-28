import json
from sklearn.metrics import accuracy_score



# dataset = 'GSM8K'
# dataset = 'SVAMP'
dataset = 'ASDiv-a'


# def vote(answers, topk):
#     ballot = {}
#     for a in answers:
#         if a not in ballot:
#             ballot[a] = 1
#         else:
#             ballot[a] += 1
#     rank = sorted(ballot.items(), key = lambda kv:(kv[1], kv[0]))
#     final_answer = [key for key, value in rank[-topk:]]
#     return final_answer


pred_answers = []
gold_answers = []
with open('data/' + dataset + '.jsonl') as f:
    for line in f.readlines():
        obj = json.loads(line)

        max_ballot = 0
        pred_answer = ''
        for gpt_answer in obj['gpt_solutions'].keys():
            cur_ballot = len(obj['gpt_solutions'][gpt_answer])
            if cur_ballot > max_ballot:
                pred_answer = gpt_answer
                max_ballot = cur_ballot
        
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