import os
import json
import re

# 设置基础文件夹路径
base_folder_path = '/home/niuhaojia/file/data'
# 初始化数据存储列表
data_records = []

# 编译正则表达式以匹配有效的日期格式
timestamp_pattern = re.compile(r'^(\d{4})-(0?[1-9]|1[0-2]|MM)$')

# 遍历基础文件夹中的每个子文件夹
for folder_name in os.listdir(base_folder_path):
    folder_path = os.path.join(base_folder_path, folder_name)
    # 构造1_4文件夹的完整路径
    json_folder_path = os.path.join(folder_path, '1_4')
    # 构造1_1文件夹的完整路径
    txt_folder_path = os.path.join(folder_path, '1_1')
    
    # 确保1_4文件夹和1_1文件夹都存在
    if os.path.isdir(json_folder_path) and os.path.isdir(txt_folder_path):
        # 遍历1_4文件夹中的每个JSON文件
        for file_name in os.listdir(json_folder_path):
            if file_name.endswith('.json'):
                with open(os.path.join(json_folder_path, file_name), 'r') as json_file:
                    json_data = json.load(json_file)
                    # 筛选日期在1700-2020之间的记录索引和年份
                    valid_indices = []
                    years = []
                    cooperation = []  # 用于存储record[6]的内容
                    for i, record in enumerate(json_data):
                        match = timestamp_pattern.match(record[5])
                        if match:
                            year_str = match.group(1)
                            year = int(year_str)
                            if 1700 <= year <= 2020 and record[6] != "Non-cooperation":
                                valid_indices.append(i)
                                years.append(year)  # 存储对应的年份
                                cooperation.append(record[6])  # 存储record[6]的内容
                    # 根据索引，从1_1文件夹中的对应TXT文件读取数据行
                    txt_file_name = file_name.replace('_4.json', '_1.txt')
                    txt_file_path = os.path.join(txt_folder_path, txt_file_name)
                    if os.path.isfile(txt_file_path):
                        with open(txt_file_path, 'r') as txt_file:
                            lines = txt_file.readlines()
                            # 只添加数量与有效JSON记录匹配的TXT行和年份
                            for index, year, info in zip(valid_indices, years, cooperation):
                                if 0 < index + 1 < len(lines):
                                    record_tuple = (lines[index+1].strip(), year, info)
                                    data_records.append(record_tuple)

import json
import os
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# 确保已经下载了以下资源
nltk.download('wordnet')
nltk.download('stopwords')

# 初始化 NLP 工具
wnl = WordNetLemmatizer()

# 假设您已经有了一个停用词列表
stoplist = set(nltk.corpus.stopwords.words('english'))

custom_stop_words = ["husband", "wife", "brother", "sister", "daughter", "son", "father", "mother", "friend", "colleague","aunt","uncle","nephew","niece","granddaughter","grandson","grandfather","grandmother","father-in-law","mother-in-law","son-in-law","daughter-in-law","cousin"]
stoplist.update(custom_stop_words)

# 假设您有一个结果文件路径
result_file_path = '/home/niuhaojia/hezuo/LDA/data.json'

def preprocess_text(text):
    """去标点, 去数字, 分割成单词, 词形还原"""
    text = text.lower()
    text = re.sub(r'[{}]+'.format(string.punctuation + string.digits), ' ', text)
    words = nltk.word_tokenize(text)
    words = [wnl.lemmatize(word) for word in words if word not in stoplist and len(word) >= 3 and wordnet.synsets(word)]
    return ' '.join(words)

def add_data_to_json(file_path, content):
    """添加处理后的文本到 JSON 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append({'content': content})

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

# 预处理 data_records 中的每个文本，并将结果添加到 JSON 文件
for record in data_records:
    text = record[0]  # 获取文本内容
    preprocessed_text = preprocess_text(text)
    add_data_to_json(result_file_path, preprocessed_text)

# 提取词汇表
def extract_vocab_from_json(json_file_path):
    vocab = set()
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    for item in data:
        content = item.get('content', '')
        words = re.findall(r'\b\w+\b', content)
        vocab.update(words)
    return tuple(vocab)

# 创建文档-词频矩阵
def create_document_term_matrix(text_list, vocab):
    vectorizer = CountVectorizer(vocabulary=vocab)
    dtm = vectorizer.fit_transform(text_list)
    return dtm.toarray()

# 使用 JSON 文件中的数据创建词汇表和文档-词频矩阵
vocab = extract_vocab_from_json(result_file_path)
content_list = [item['content'] for item in json.load(open(result_file_path, 'r', encoding='utf-8'))]
dtm = create_document_term_matrix(content_list, vocab)

# 保存词汇表和词频矩阵到文件
np.save('dtm.npy', dtm)
with open('vocab.json', 'w', encoding='utf-8') as vocab_file:
    json.dump(list(vocab), vocab_file, ensure_ascii=False)

print("词频矩阵和词汇表已保存到本地。")

import numpy as np
import lda
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
import warnings
warnings.filterwarnings('ignore')

# 函数：提取DTM中频率最高的N个词
def extract_top_n(N, X, vocab):
    word_frequencies = np.sum(X, axis=0)
    top_words_indices = np.argsort(word_frequencies)[::-1][:N]
    top_words = [(vocab[idx], word_frequencies[idx]) for idx in top_words_indices]
    return top_words

# 函数：训练LDA模型
def LDA_train(X, vocab, n_topics=20, n_iter=40000, alpha=0.1, eta=0.01, random_state=1):
    model = lda.LDA(n_topics=n_topics, n_iter=n_iter, alpha=alpha, eta=eta, random_state=random_state)
    model.fit(X)
    return model

# 假设您有一个来自之前处理的‘dtm.npy’和‘vocab.json’文件
dtm_path = '/home/niuhaojia/hezuo/LDA/dtm.npy'  # 替换为您DTM文件的实际路径
vocab_path = '/home/niuhaojia/hezuo/LDA/vocab.json'  # 替换为您词汇文件的实际路径


# 加载文档-词项矩阵和词汇
dtm = np.load(dtm_path)
with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
    vocab = json.load(vocab_file)

# 训练LDA模型
n_topics = 3
model = LDA_train(dtm, vocab, n_topics=n_topics)

# 获取文档-主题分布
doc_topic_dist = model.doc_topic_
# 保存文档-主题分布到文件
np.save('/home/niuhaojia/hezuo/LDA/doc_topic_dist.npy', doc_topic_dist)
print("文档-主题分布已保存到本地。")

# 将模型保存到磁盘
model_path = '/home/niuhaojia/hezuo/LDA/lda_model.pkl'  # 替换为您想要保存的路径
joblib.dump(model, model_path)

# 提取并保存每个主题的前N个词到TXT文件中
n_top_words = 40
topic_word = model.topic_word_

# 定义要保存主题和关键词的文件路径
topics_file_path = '/home/niuhaojia/hezuo/LDA/topics_keywords.txt'  # 根据需要修改路径

with open(topics_file_path, 'w', encoding='utf-8') as f:
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        topic_info = '主题 {}: {}'.format(i, ' '.join(topic_words))
        f.write(topic_info + '\n')  # 将主题信息写入文件

print("主题及其关键词已保存到 {}".format(topics_file_path))


# 使用PCA在3D图中可视化主题
doc_topic = model.doc_topic_
pca = PCA(n_components=3)
reduced_similarity = pca.fit_transform(doc_topic.T)

# 3D 可视化
fig = plt.figure(figsize=(16, 8))

# 3D 图
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(reduced_similarity[:, 0], reduced_similarity[:, 1], reduced_similarity[:, 2])
ax1.set_title('PCA 3D')

# 2D 图
ax2 = fig.add_subplot(122)
ax2.scatter(reduced_similarity[:, 0], reduced_similarity[:, 1])
ax2.set_title('PCA 2D')

# 设置要保存的图片的路径和名称
image_path = '/home/niuhaojia/hezuo/LDA/image.png'  # 替换为您想要保存图片的路径和名称
plt.savefig(image_path)  # 在显示之前保存图片

plt.show()  # 显示图片

# 使用PCA在3D图中可视化主题
doc_topic = model.doc_topic_
pca = PCA(n_components=3)
reduced_similarity = pca.fit_transform(doc_topic.T)

# 3D 可视化
fig = plt.figure(figsize=(16, 8))

# 3D 图
ax1 = fig.add_subplot(121, projection='3d')
scatter = ax1.scatter(reduced_similarity[:, 0], reduced_similarity[:, 1], reduced_similarity[:, 2])

# 为每个点添加标签
for i in range(reduced_similarity.shape[0]):
    ax1.text(reduced_similarity[i, 0], reduced_similarity[i, 1], reduced_similarity[i, 2], str(i), color='red')

ax1.set_title('PCA 3D')

# 2D 图
ax2 = fig.add_subplot(122)
scatter = ax2.scatter(reduced_similarity[:, 0], reduced_similarity[:, 1])

# 为每个点添加标签
for i in range(reduced_similarity.shape[0]):
    ax2.text(reduced_similarity[i, 0], reduced_similarity[i, 1], str(i), color='red')

ax2.set_title('PCA 2D')

# 设置要保存的图片的路径和名称
image_path = '/home/niuhaojia/hezuo/LDA/image2.png'  # 替换为您想要保存图片的路径和名称
plt.savefig(image_path)  # 在显示之前保存图片

plt.show()  # 显示图片