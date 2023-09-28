import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

import numpy as np
import json
from tqdm import tqdm
import random
import os
import re


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(123)

pretrained_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
bert_model = AutoModel.from_pretrained(pretrained_model_name)


class SolutionVerifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(SolutionVerifier, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(base_model.config.hidden_size, num_classes)
        self.fc_step = nn.Linear(base_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, step_positions=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        x = self.fc(cls_token)
        if step_positions == None:
            return x
        # step_tokens = []
        # for i in range(last_hidden_state.size()[0]):
            # step_tokens.append(last_hidden_state[i, step_positions[i], :])
        step_tokens = [last_hidden_state[0, step_positions[0], :]]
        step_tokens = torch.stack(step_tokens, dim=0)
        step_output = self.fc_step(step_tokens)
        return x, step_output

num_classes = 1
model = SolutionVerifier(bert_model, num_classes)


class TextDataset(Dataset):
    def __init__(self, texts, labels, step_labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.step_labels = step_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].replace('\n', '[SEP]') # \n -> [SEP]
        label = self.labels[idx]
        step_labels = [[l] for l in self.step_labels[idx]]
        # step_labels = self.step_labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        offsets = encoding['offset_mapping'][0]

        last_token_positions = []
        for i in range(len(tokens)):
            if tokens[i] == '[SEP]':
                # last_token_positions.append(offsets[i][1]) # char
                last_token_positions.append(i)

        return {
            "input_ids": encoding["input_ids"].flatten(), # a.flatten() here = a[0]
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor([label]),
            "step_positions": torch.tensor(last_token_positions),
            "step_labels": torch.tensor(step_labels),
        }


def load_train_data(dataset='train'):
    texts, labels, step_labels = [], [], []
    with open('../data/' + dataset + '.jsonl') as f:
        for i, line in enumerate(f.readlines()):
            # if i > 10:
                # break
            D = json.loads(line)
            right_step = re.findall("<<([^>]*)>>", D['solution']) # right-step in right solution
            for gpt_answer in D['gpt_solutions'].keys():
                for gpt_solution in D['gpt_solutions'][gpt_answer]:
                    texts.append(gpt_solution)
                    if gpt_answer == D['answer']:
                        labels.append(1.)
                    else:
                        labels.append(0.)
                    # step-aware
                    tmp_step_labels = []
                    false_direction = False
                    for step in gpt_solution.split('\n'):
                        step = re.findall("<<([^>]*)>>", step)
                        if len(step) == 0:
                            tmp_step_labels.append(0.)
                            continue
                        step = step[0]
                        if false_direction or step not in right_step:
                            tmp_step_labels.append(0.)
                            false_direction = True
                        else:
                            tmp_step_labels.append(1.)
                    step_labels.append(tmp_step_labels)

    return texts, labels, step_labels



def load_test_data(dataset):
    gpt_solutions, gpt_answers, gold_answers = [], [], []
    with open('../data/' + dataset + '.jsonl') as f:
        for line in f.readlines():
            D = json.loads(line)
            solutions = []
            answers = []
            for gpt_answer in D['gpt_solutions'].keys():
                for gpt_solution in D['gpt_solutions'][gpt_answer]:
                    solutions.append(gpt_solution)
                    answers.append(gpt_answer)
            gpt_solutions.append(solutions)
            gpt_answers.append(answers)
            gold_answers.append(D['answer'])
    return gpt_solutions, gpt_answers, gold_answers


train_texts, train_labels, train_step_labels = load_train_data()
GSM8K_data = load_test_data('GSM8K')
ASDiv_a_data = load_test_data('ASDiv-a')
SVAMP_data = load_test_data('SVAMP')


max_length = 512
train_dataset = TextDataset(train_texts, train_labels, train_step_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

alpha = 0.1
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


def compute_accuracy_score(gold, pred):
    crt = 0.
    for g, p in zip(gold, pred):
        if g == p:
            crt += 1
    return crt / len(gold)


### Train


epochs = 2
patience_cnt = 0
best_acc = 0
best_epoch = 0
print('Start training............')
for epoch in range(1, epochs + 1):
    # train
    model.train()
    train_loss = 0
    cnt_train = 0
    train_bar = tqdm(train_dataloader, position=0, leave=True)
    for batch_data in train_bar:
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        labels = batch_data["label"].to(device)
        step_positions = batch_data["step_positions"]
        step_labels = batch_data["step_labels"].to(device)


        optimizer.zero_grad()
        outputs, step_outputs = model(input_ids, attention_mask, step_positions)
        loss = criterion(outputs, labels) + alpha * criterion(step_outputs, step_labels[:, :step_outputs.size(1), :])
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        cnt_train += 1
        train_bar.set_description(f'epoch {epoch}')
        train_bar.set_postfix(loss=train_loss / cnt_train)
    
    ## save model
    # save_path = '../models/solution_verifier.pt'
    # torch.save(model, save_path)

    ## valid
    # test_data = gpt_solutions, gpt_answers, gold_answers
    for test_data in [GSM8K_data, ASDiv_a_data, SVAMP_data]:
        gold = []
        verify_pred = []
        voting_verify_pred = []
        model.eval()
        with torch.no_grad():
            gpt_solutions, gpt_answers, gold_answers = test_data
            valid_bar = tqdm(zip(gpt_solutions, gpt_answers, gold_answers), total=len(gpt_solutions), position=0, leave=True)
            # gpt_solution, gpt_answer, gold_answer
            for gpt_solution, gpt_answer, gold_answer in valid_bar:
                answer_score = {}
                max_score = 0.
                best_answer = ''
                for solution, answer in zip(gpt_solution, gpt_answer):
                    test_text = solution
                    encoding = tokenizer(
                        test_text,
                        add_special_tokens=True,
                        max_length=max_length,
                        return_token_type_ids=False,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                    input_ids = encoding["input_ids"].to(device)
                    attention_mask = encoding["attention_mask"].to(device)

                    output = model(input_ids, attention_mask)
                    score = prediction = output.sigmoid().squeeze(-1).item()

                    if answer in answer_score.keys():
                        answer_score[answer] += score
                    else:
                        answer_score[answer] = score
                    if score > max_score:
                        max_score = score
                        best_answer = answer
                
                gold.append(gold_answer)
                verify_pred.append(best_answer)
                final_answer = max(answer_score, key=lambda k: answer_score[k])
                voting_verify_pred.append(final_answer)

            valid_bar.set_description(f'epoch {epoch}')


        verify_acc = compute_accuracy_score(gold, verify_pred)
        voting_verify_acc = compute_accuracy_score(gold, voting_verify_pred)
        print(f'Epoch [{epoch}/{epochs}], verify_acc: {verify_acc}, voting_verify_acc: {voting_verify_acc}')

        # if final_acc > best_acc:
        #     patience_cnt = 0
        #     best_acc = final_acc
        #     best_epoch = epoch
        #     # checkpoint = {
        #     #     'net': model.state_dict(),
        #     #     'optimizer':optimizer.state_dict(),
        #     #     'epoch': epoch
        #     # }
        #     # torch.save(checkpoint, save_path)
        #     # torch.save(model, save_path)
        #     print('***** new score *****')
        #     print(f'The best epoch is: {best_epoch}, with the best acc is: {best_acc}')
        #     print('********************')
        # elif patience_cnt >= patience: # early stop
        #     print(f'Early Stop with best epoch {best_epoch}, with the best acc is: {best_acc}')
        #     break
        # patience_cnt += 1