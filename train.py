from utils import *
from dataloader import get_dataloader
# from network import Network, SolutionVerifier
## ablation
# from ablation.network_wo_graph import Network, SolutionVerifier
# from ablation.network_wo_score import Network, SolutionVerifier
from ablation.network_wo_solution import Network, SolutionVerifier


from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# PLM = 'microsoft/deberta-v3-base'
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
PLM = 'bert-base-uncased'
# PLM = 'roberta-base'
# dataset='GSM8K'

seed = 123


if seed != None:
    seed_everything(seed)


def train(epochs=20, patience=1, lr=2e-5, batch_size=16, max_len=512, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(PLM)
    bert = AutoModel.from_pretrained(PLM)
    bert = bert.to(device)

    print('Prepare data......')
    train_data = load_data('data/train.jsonl')
    # train_data = load_data('data/train.jsonl', 2)
    # test_data = load_data('data/train.jsonl', 2)
    # test_data = load_data('data/' + dataset + '.jsonl')
    GSM8K_data = load_data('data/GSM8K.jsonl')
    SVAMP_data = load_data('data/SVAMP.jsonl')
    ASDiv_a_data = load_data('data/ASDiv-a.jsonl')

    train_data_generator = get_dataloader(train_data, batch_size, None, None, None, is_shuffle=True)
    GSM8K_data_generator = get_dataloader(GSM8K_data, 1, None, None, None)
    SVAMP_data_generator = get_dataloader(SVAMP_data, 1, None, None, None)
    ASDiv_a_data_generator = get_dataloader(ASDiv_a_data, 1, None, None, None)

    model = Network(bert, tokenizer, device, max_seq_len=max_len).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.cls.parameters(), 'lr': 4e-2}, # 1e-2
        {'params': model.solution_verifier.parameters(), 'lr': 4e-5},
        # {'params': model.bert.parameters(), 'lr': 1e-4},
    ], lr=4e-3) # 1e-3
    # optimizer = torch.o # ptim.AdamW(model.parameters(), lr=lr)

    # mark best epoch
    # patience_cnt = 0
    # best_acc = 0
    # best_epoch = 0
    best_acc = [0, 0, 0]
    print('Start training............')
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0
        cnt_train = 0
        train_bar = tqdm(train_data_generator, position=0, leave=True)
        for batch_data in train_bar:
            optimizer.zero_grad()
            loss = model.compute_loss(batch_data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            cnt_train += 1
            train_bar.set_description(f'epoch {epoch}')
            train_bar.set_postfix(loss=train_loss / cnt_train)

        # print(f'Epoch [{epoch}/{epochs}]: loss: {train_loss / cnt_train}')
        
        ## valid
        for dataset_i, test_data_generator in enumerate([GSM8K_data_generator, SVAMP_data_generator, ASDiv_a_data_generator]):
            gold = [] # final answer acc
            pred = []
            gold_j = [] # graph judge acc
            pred_j = []
            model.eval()
            cnt = 0
            with torch.no_grad():
                valid_bar = tqdm(test_data_generator, position=0, leave=True)
                for batch_data in valid_bar:
                    cnt += 1

                    single_data = batch_data[0]
                    pred_answer, gold_answer, pred_judge, gold_judge = model.predict(single_data)
                    gold.append(gold_answer)
                    pred.append(pred_answer)
                    gold_j.extend(gold_judge)
                    pred_j.extend(pred_judge)
                    valid_bar.set_description(f'epoch {epoch}')

                    # if cnt % 10 == 0:
                    #     acc, f1, precision, recall = compute_graph_judge_score(gold_j, pred_j)
                    #     answer_acc = compute_accuracy_score(gold, pred)
                    #     final_acc = answer_acc
                    #     print(f'Epoch [{epoch}/{epochs}], Final Acc: {final_acc}')
                    #     print(f'Graph Judge, acc: {acc}, f1: {f1}, pre: {precision}, rec:{recall}') 
            
                # print(pred_j)
                # print(gold_j)
            acc, f1, precision, recall = compute_graph_judge_score(gold_j, pred_j)
            answer_acc = compute_accuracy_score(gold, pred)
            final_acc = answer_acc
            if dataset_i == 0:
                dataset_name = 'GSM8K'
            elif dataset_i == 1:
                dataset_name = 'SVAMP'
            elif dataset_i == 2:
                dataset_name = 'ASDiv-a'
            print(f'Epoch [{epoch}/{epochs}], {dataset_name}, Final Acc: {final_acc}')
            print(f'Graph Judge, acc: {acc}, f1: {f1}, pre: {precision}, rec:{recall}') 

            if final_acc > best_acc[dataset_i]:
                best_acc[dataset_i] = final_acc

        print(f'***** Current best accuracy *****')
        print(f'GSM8K: {best_acc[0]}, SVAMP: {best_acc[1]}, ASDiv-a: {best_acc[2]}')
        print(f'*********************************')
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
    

    return model



if __name__ == '__main__':
    model = train(epochs=10, patience=10, lr=2e-5, batch_size=2, max_len=512, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
