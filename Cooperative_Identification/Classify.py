import os
import json

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os, random
import numpy as np
import pandas as pd

MAX_LENGTH = 256
BATCH_SIZE = 8

DATA_DIR = r"mydata"
# 如果是csv文件
df_train = pd.read_csv(os.path.join(DATA_DIR, r"train.csv"), encoding='utf-8', sep=',', header=None)
df_train.columns = ['input', 'output']

df_test = pd.read_csv(os.path.join(DATA_DIR, r"test.csv"), encoding='utf-8', sep=',', header=None)
df_test.columns = ['input', 'output']

# df_valid = pd.read_csv(os.path.join(DATA_DIR, r"valid.txt"), encoding='utf-8', sep='\t', header=None)
# df_valid.columns = ['input', 'output']
# 设置随机种子 ++++++++++
import torch
seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import re
def handle(content):
    content = content.strip()
    content = re.sub(r"https?://\S+", "", content)
    content = re.sub(r"what's", "what is", content)
    content = re.sub(r"Won't", "will not", content)
    content = re.sub(r"can't", "can not", content)
    content = re.sub(r"\'s", " ", content)
    content = re.sub(r"\'ve", " have", content)
    content = re.sub(r"n't", " not", content)
    content = re.sub(r"i'm", "i am", content)
    content = re.sub(r"\'re", " are", content)
    content = re.sub(r"\'d", " would", content)
    content = re.sub(r"\'ll", " will", content)
    content = re.sub(r"e - mail", "email", content)
    content = re.sub("\d+ ", "NUM", content)
    content = re.sub(r"<br />", '', content)
    content = re.sub(r'[\u0000-\u0019\u0021-\u0040\u007a-\uffff]', '', content)
    return content



df_train['input'] = df_train.apply(lambda x : handle(str(x['input']).lower()), axis=1)
df_train['output'] = df_train.apply(lambda x : str(x['output']).strip(), axis=1)
df_test['output'] = df_test.apply(lambda x : str(x['output']).strip(), axis=1)
# df_valid['output'] = df_valid.apply(lambda x : str(x['output']).strip(), axis=1)



train_pairs = [[x, y] for x, y in zip(list(df_train['input']), list(df_train['output']))]
test_pairs = [[x, y] for x, y in zip(list(df_test['input']), list(df_test['output']))]
# valid_pairs = [[x, y] for x, y in zip(list(df_valid['input']), list(df_valid['output']))]


# 类别处理
import pickle
from sklearn.preprocessing import LabelEncoder
# 类别处理
label_encoder_file = os.path.join(DATA_DIR, 'label_encoder.pickle')
if os.path.exists(label_encoder_file):
    with open(label_encoder_file, 'rb') as fr:
        label_encoder = pickle.load(fr)
    label_train = label_encoder.transform(list(df_train['output']))
else:
    label_encoder = LabelEncoder()
    label_train = label_encoder.fit_transform(list(df_train['output']))
    with open(label_encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)

list_target_names = label_encoder.classes_
list_target = label_encoder.transform(label_encoder.classes_)
dict_label_map = {list_target_names[i]:list_target[i] for i in range(len(list_target_names))}
num_labels = len(list_target_names)
class PairsLoader():
    def __init__(self, pairs, word2index, tokenizer, batch_size, max_length):
        self.word2index = word2index
        self.pairs = pairs
        self.batch_size = batch_size
        self.max_length = max_length
        self.position = 0
        self.tokenizer = tokenizer

    def load_single_pair(self):
        if self.position >= len(self.pairs):
            # random.shuffle(self.pairs)
            self.position = 0
        single_pair = self.pairs[self.position]
        self.position += 1
        return single_pair

    def load(self):
        while True:
            input_batch = []
            output_batch = []
            mask_batch = []
            for i in range(self.batch_size):
                pair = self.load_single_pair()
                input_indexes, output_indexes, attn_masks = self.tokenizer(pair[0], pair[1], 'train')
                input_batch.append(input_indexes)
                output_batch.append(output_indexes)
                mask_batch.append(attn_masks)
            yield input_batch, mask_batch, output_batch
import torch
from transformers import BertTokenizer, BertModel


MODEL_DIR = r"/home/niuhaojia/分类1/model_bert_classification"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

init_model = r"/home/niuhaojia/分类1/bert-base-uncased" # 英文

tokenizer_ori = BertTokenizer.from_pretrained(init_model)
def tokenizer(x, y, type='train'):
    encodings_dict = tokenizer_ori(str(x), padding='max_length', truncation=True, max_length=MAX_LENGTH)
    input_indexes = encodings_dict['input_ids']
    attn_masks = encodings_dict['attention_mask']
    output_indexes = dict_label_map.get(y, '')
    return input_indexes, output_indexes, attn_masks


class bertcls(torch.nn.Module):
    def __init__(self, config):
        super(bertcls, self).__init__()
        self.bert = BertModel.from_pretrained(config['init_model'])
        self.classifier = torch.nn.Linear(config['hidden_size'], config['num_classes'])
        self.dropout = torch.nn.Dropout(p=config['dropout'])

    def forward(self, input_variable, attention_mask=None):
        '''
        :param x:[input_ids,seq_len,mask]
        :return:
        '''
        outputs = self.bert(input_variable, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {
    'init_model': init_model,
    'hidden_size':  768,
    'num_classes': num_labels,
    'dropout': 0.5,
}

model = bertcls(config)

model.to(device)
import torch
import os
import pandas as pd
from sklearn import metrics
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

# 设定类别2的置信度阈值
THRESHOLD_FOR_CLASS_2 = 0.8

init_model = "model_bert_classification/8"

with open(os.path.join(init_model, 'label_encoder.pickle'), 'rb') as fr:
    label_encoder = pickle.load(fr)

tokenizer_ori = BertTokenizer.from_pretrained(r"bert-base-uncased")

# 假设bertcls是之前定义的模型类
model = bertcls(config)
model.load_state_dict(torch.load(os.path.join(init_model, "model.pth"), map_location=device))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def evaluate(prompt, model):
    prompt = handle(prompt).lower()
    prompt = tokenizer_ori.encode(str(prompt), truncation=True, max_length=MAX_LENGTH)
    prompt = torch.tensor(prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(prompt)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    max_prob, y_pred_label = torch.max(probabilities, dim=1)
    # 应用置信度阈值逻辑
    if y_pred_label.item() == 2 and max_prob.item() < THRESHOLD_FOR_CLASS_2:
        y_pred_label = torch.tensor([0]).to(device)
    pred_name = label_encoder.inverse_transform([y_pred_label.item()])
    return y_pred_label.item(), pred_name[0]

list_result = []
for pair in test_pairs:
    pred, pred_name = evaluate(pair[0], model)
    list_result.append([pair[0], pair[1], pred_name])

df = pd.DataFrame(list_result, columns=['input', 'output', 'pred'])
df['output'] = df.apply(lambda x : label_encoder.transform([x['output']])[0], axis=1)
df['pred'] = df.apply(lambda x : label_encoder.transform([x['pred']])[0], axis=1)

df.to_excel(os.path.join(init_model, "result.xlsx"))

score = metrics.accuracy_score(list(df['output']), list(df['pred']))
print("accuracy:   %0.3f" % score)
print('_' * 80)

report = metrics.classification_report(
    list(df['output']), list(df['pred']), target_names=[str(x) for x in label_encoder.classes_], labels=label_encoder.transform(label_encoder.classes_))
print("classification report:")
print(report)

# ---------------------------------------

# 主目录路径
main_folder = "/home/niuhaojia/file/data"

# 遍历主目录下的每个子目录
for subdir in os.listdir(main_folder):
    subdir_path = os.path.join(main_folder, subdir)
    if os.path.isdir(subdir_path):  # 确保是一个目录
        txt_folder = os.path.join(subdir_path, "1_1")
        output_folder = os.path.join(subdir_path, "1_3")
        new_output_folder = os.path.join(subdir_path, "1_4")  # 新的输出文件夹路径
        
        # 确保新的输出目录存在
        if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)
        
        # 获取txt_folder下的所有文本文件
        txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
        
        # 遍历每个文本文件
        for txt_file in txt_files:
            # 构建完整的文件路径
            txt_file_path = os.path.join(txt_folder, txt_file)
            json_file_path = os.path.join(output_folder, txt_file.replace('_1.txt', '_3.json'))
            new_json_file_path = os.path.join(new_output_folder, txt_file.replace('_1.txt', '_4.json'))
            
            # 读取文本文件中的每一行，并跳过第一行
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()[1:]
            
            # 对每一行文本进行分类
            labeled_lines = []
            for line in lines:
                pred, pred_name = evaluate(line.strip(), model)  # 假设model参数在这里不需要
                if pred == 1:
                    label = "Explicit Cooperation"
                elif pred == 0:
                    label = "Non-cooperation"
                elif pred == 2:
                    label = "Implicit Cooperation"
                else:
                    label = "Unknown"
                labeled_lines.append(label)
            
            # 读取原始的JSON数据
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)
            
            # 将分类结果添加到JSON数据中
            for i, label in enumerate(labeled_lines):
                if i < len(json_data):
                    json_data[i].append(labeled_lines[i])
            
            # 将更新后的数据写入到新的JSON文件
            with open(new_json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)


