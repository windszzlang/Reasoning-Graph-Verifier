import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import dgl
from dgl.nn import GINConv, GINEConv, GATConv, GraphConv
import re
import random



class SolutionVerifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(SolutionVerifier, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(base_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        x = self.fc(cls_token)
        return x
    

class Network(nn.Module):
    def __init__(self, plm, tokenizer, device, max_seq_len=512):
        super(Network, self).__init__()
        self.bert = plm
        self.bert_config = self.bert.config
        self.tokenizer = tokenizer
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.device = device
        self.max_seq_len = max_seq_len

        self.dropout = nn.Dropout(0.2)

        self.graph_hidden_dim = 512
        self.gnn_1 = GINConv(nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim))
        # self.gnn_1 = GraphConv(self.bert_config.hidden_size, hidden_dim, allow_zero_in_degree=True)
        self.gnn_2 = GINConv(nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim))
        # self.gnn_2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.gnn_3 = GINConv(nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim))
        # self.gnn_3 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.activate_fn = nn.ReLU()

        self.solution_verifier = torch.load('models/solution_verifier.pt')
        # self.solution_verifier = SolutionVerifier(self.bert, 1)
        self.cls = nn.Linear(self.graph_hidden_dim, 1)
        # self.cls = nn.Linear(self.graph_hidden_dim, 1)

        self.criterion = nn.BCEWithLogitsLoss()


    def forward(self, g, node_features, edge_features, batch_answer_scores):
        h = self.gnn_1(g, node_features)
        # h = self.gnn_1(g, node_features, edge_features)
        h = self.activate_fn(h)
        h = self.gnn_2(g, h)
        # h = self.gnn_2(g, h, edge_features)
        h = self.activate_fn(h)
        h = self.gnn_3(g, h)
        # h = self.gnn_3(g, h, edge_features)
        h = self.activate_fn(h)

        with g.local_scope(): # only effective in this scope
            g.ndata['h'] = h
            # g.ndata['h'] = h + node_features
            g_readout = dgl.sum_nodes(g, 'h')
            # g_readout = dgl.mean_nodes(g, 'h')
            # g_readout = dgl.max_nodes(g, 'h')

            # graphs = dgl.unbatch(g)
            # answers_emb = []
            # for graph, node in zip(graphs, answer_nodes):
            #     answer = graph.ndata['v'][node]
            #     answers_emb.append(answer)
            # g_readout = torch.stack(answers_emb, dim=0)
        # print(g_readout)
        # cls_input = torch.cat([g_readout, batch_answer_scores.unsqueeze(-1)], dim=1)
        cls_input = g_readout
        logits = self.cls(cls_input)
        # logits = self.cls(g_readout)
        return logits


    def compute_loss(self, batch_data):
        batch_graphs, batch_labels, _, batch_answer_scores = self.build_batch_graph_data(batch_data)
        logits = self(batch_graphs, batch_graphs.ndata['v'], batch_graphs.edata['v'], batch_answer_scores)
        loss = self.criterion(logits, batch_labels.unsqueeze(-1))
        # self.compute_balance(batch_labels)
        return loss


    def compute_balance(self, batch_labels):
        cnt = 0.
        for l in batch_labels:
            if l == 1:
                cnt += 1
        print(cnt / len(batch_labels))


    def predict(self, single_data):
        self.eval()
        gold_answer = single_data['answer']
        graphs, labels, gpt_answers, answer_scores = self.build_graph_data(single_data, is_test=True)
        
        graphs = dgl.batch(graphs).to(self.device)
        answer_scores = torch.tensor(answer_scores).to(self.device)
        logits = self(graphs, graphs.ndata['v'], graphs.edata['v'], answer_scores).sigmoid().squeeze(-1)

        # print(logits)
        max_id = logits.argmax()
        # max_id = logits[:, 1].argmax()
        # max_id = (torch.exp(logits[:, 1]) / (torch.exp(logits[:, 0]) + torch.exp(logits[:, 1]))).argmax()
        pred_answer = gpt_answers[max_id]
        
        # print('logits:', logits)
        # print('labels:', labels)
        pred_judge = logits * 0.
        pred_judge[max_id] = 1.
        # pred_judge = logits > 0.5
        pred_judge = pred_judge.cpu()
        gold_judge = labels
        self.train()
        # print(logits)
        return pred_answer, gold_answer, pred_judge, gold_judge

    
    # utility functions
    def build_batch_graph_data(self, batch_data):
        batch_graphs = []
        batch_labels = []
        batch_gpt_answers = []
        batch_answer_scores = []
        for d in batch_data:
            graphs, labels, gpt_answers, answer_scores = self.build_graph_data(d)
            batch_graphs.extend(graphs)
            batch_labels.extend(labels)
            batch_gpt_answers.extend(gpt_answers)
            batch_answer_scores.extend(answer_scores)
        batch_graphs = dgl.batch(batch_graphs).to(self.device)
        batch_labels = torch.tensor(batch_labels).to(self.device)
        batch_answer_scores = torch.tensor(batch_answer_scores).to(self.device)
        return batch_graphs, batch_labels, batch_gpt_answers, batch_answer_scores


    def build_graph_data(self, data, is_test=False):
        question = data['question']
        true_solution = data['solution']
        answer = data['answer']
        gpt_solutions = data['gpt_solutions']

        graphs = []
        labels = []
        gpt_answers = []
        answer_scores = []

        # print(gpt_solutions.keys())
        for gpt_answer in gpt_solutions.keys():
            solutions_with_same_answer = gpt_solutions[gpt_answer]

            ## get text embeddings
            # text_embs = self.tokenize_and_embed_texts(solutions_with_same_answer)# [num, hidden_size]

            # get solution score
            solution_scores = self.compute_solution_scores(solutions_with_same_answer)
            answer_scores.append(solution_scores.sum()) # sum of all scores

            # get reason path
            reason_paths = []
            for solution in solutions_with_same_answer:
                reason_path = self.get_reason_path(solution, gpt_answer)
                reason_paths.append(reason_path)

            # create nodes
            node_num = 1
            node2id = {question: 0}
            # question_node_feature = text_embs[0][0]
            # question_node_feature = text_embs[0]
            # question_node_feature = torch.zeros(1).to(self.device) 
            # node_features_semantic = [[question_node_feature]]
            node_features = [[]]
            for i in range(len(reason_paths)):
                reason_path = reason_paths[i]
                for j in range(len(reason_path)):

                    # node_feature = torch.ones(self.bert_config.hidden_size).to(self.device)
                    # node_feature = text_embs[i][j]
                    # node_feature_semantic = text_embs[i]
                    node_feature = torch.ones(self.graph_hidden_dim).to(self.device)
                    # node_feature = solution_scores[i].to(self.device)

                    if j == 0: # question
                        node_features[0].append(node_feature)
                    elif reason_path[j] in node2id.keys(): # same step-solution
                        node_id = node2id[reason_path[j]]
                        node_features[node_id].append(node_feature)
                        # node_features_semantic[node_id].append(node_feature_semantic)
                    else: # new step-solution
                        node2id[reason_path[j]] = node_num
                        node_features.append([node_feature])
                        # node_features_semantic.append([node_feature_semantic])
                        node_num += 1
            # shape: [node_num, same_num, hidden_size] -> [node_num, hidden_size]
            # feat.shape=[same_num, hidden_size]
            node_features = torch.tensor([torch.mean(torch.stack(feat, dim=0), dim=0).tolist() for feat in node_features]).to(self.device) # mean of text embeddings with same step-solution
            # node_features_mean = torch.tensor([torch.mean(torch.stack(feat, dim=0), dim=0).tolist() for feat in node_features]).to(self.device) # mean of text embeddings with same step-solution
            # node_features_max = torch.tensor([torch.max(torch.stack(feat, dim=0), dim=0)[0].tolist() for feat in node_features]).to(self.device)
            # node_features_min = torch.tensor([torch.min(torch.stack(feat, dim=0), dim=0)[0].tolist() for feat in node_features]).to(self.device)
            # node_features_num = torch.tensor([[len(feat)] for feat in node_features]).to(self.device)
            # node_features_semantic = torch.tensor([torch.max(torch.stack(feat, dim=0), dim=0).tolist() for feat in node_features_semantic]).to(self.device)

            # node_features = torch.cat([node_features_mean, node_features_max, node_features_min, node_features_num], dim=1)

            # create edges
            edges_num = dict() # {(start_node, end_node): occur num...}
            for i in range(len(reason_paths)):
                reason_path = reason_paths[i]
                for j in range(len(reason_path) - 1): # n-1 edges
                    start_node = node2id[reason_path[j]]
                    end_node = node2id[reason_path[j + 1]]
                    if (start_node, end_node) in edges_num.keys():
                        edges_num[(start_node, end_node)] += 1.
                    else:
                        edges_num[(start_node, end_node)] = 1.
            
            edges = [[key[0] for key in edges_num.keys()], [key[1] for key in edges_num.keys()]]
            edge_features = [[v] * self.bert_config.hidden_size for v in edges_num.values()]

            # create a graph
            g = dgl.graph((edges[0], edges[1])).to(self.device)
            
            # node_features_dgree = g.in_degrees().unsqueeze(1)
            g.ndata['v'] = node_features
            # g.ndata['v'] = torch.cat([node_features_mean, node_features_max, node_features_min, node_features_num, node_features_dgree], dim=1)
            # g.ndata['v'] = torch.tensor(node_features).to(self.device)
            g.edata['v'] = torch.tensor(edge_features).to(self.device)
            # add self loop
            # g = dgl.add_self_loop(g, fill_data=0)

            # visualize graph
            # self.visualize_graph(g)

            # assgien graph label
            if gpt_answer == answer:
                label = 1.
            else:
                label = 0.
                # r = random.random()
                # if not is_test and r < self.negative_sampling_rate:
                #     continue

            graphs.append(g)
            labels.append(label)
            gpt_answers.append(gpt_answer)
        return graphs, labels, gpt_answers, answer_scores


    def tokenize_and_embed_texts(self, texts):
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
            max_length=self.max_seq_len,
            truncation=True
        ).to(self.device)
        bert_out = self.dropout(self.bert(**encodings).last_hidden_state)
        # cls
        cls_embedding = bert_out[:, 0, :]
        
        return cls_embedding # [num, hidden_size]

    def compute_solution_scores(self, solutions):
        encodings = self.bert_tokenizer(
            solutions,
            add_special_tokens=True,
            padding='longest',
            return_token_type_ids=False,
            return_tensors='pt',
            max_length=self.max_seq_len,
            truncation=True
        ).to(self.device)
        solution_scores = self.solution_verifier(**encodings).sigmoid()
        return solution_scores


    def get_reason_path(self, solution, answer):
        reason_path = []
        split_text = solution.strip().split('\n')
        for i in range(len(split_text)):
            if i == 0:
                reason_path.append(split_text[0]) # question
            elif i == len(split_text) - 1: # final answer
                reason_path.append(answer)
            else:
                exp = re.findall(r'<<.*>>', split_text[i])
                if exp == []:
                    reason_path.append(split_text[i])
                else:
                    reason_path.append(exp[0])
        return reason_path


    def visualize_graph(self, graph):
        import networkx as nx
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        nx.draw(graph.to_networkx(), ax=ax)
        plt.show()