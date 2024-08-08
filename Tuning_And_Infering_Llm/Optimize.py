from tqdm import tqdm
import re
import ast
import os
import json
from transformers import set_seed
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import  FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import warnings
warnings.filterwarnings("ignore")

# 定义文件夹路径
json_folder = "/home/niuhaojia/file/data/father/1_2"
txt_folder = "/home/niuhaojia/file/data/father/1"
txt_folder_2 = "/home/niuhaojia/file/data/father/1_1"
if not os.path.exists("/home/niuhaojia/file/data/father/faiss"):
    os.makedirs("/home/niuhaojia/file/data/father/faiss")
faiss_folder = "/home/niuhaojia/file/data/father/faiss"
if not os.path.exists("/home/niuhaojia/file/data/father/1_3"):
    os.makedirs("/home/niuhaojia/file/data/father/1_3")
output_folder = "/home/niuhaojia/file/data/father/1_3"

# 获取json文件列表
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
json_files.sort() 

# 创建一个嵌入对象并设置模型的名称
embedding=HuggingFaceEmbeddings(model_name="/home/css/models/bge-large-en-v1.5", model_kwargs={'device': 'cuda'})

llm = ChatOpenAI(api_key="emtpy",base_url="http://127.0.0.1:8005/v1",model_name="llama3-8b-instruct",max_tokens=2048)
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)
# 遍历处理每个json文件
for json_file in tqdm(json_files):
    # 检查是否存在_3.json文件
    output_file = json_file.replace('_2.json', '_3.json')
    if os.path.exists(os.path.join(output_folder, output_file)):
        continue
    # 加载并解析JSON文件
    with open(os.path.join(json_folder, json_file), 'r') as file:
        data = json.load(file)
   
    # 使用你选择的文件路径和loader加载文档
    txt_file = json_file.replace('_2.json', '.txt')  # 根据你的规则来匹配对应的txt文件
    filepath = os.path.join(txt_folder, txt_file)
    loader = TextLoader(file_path=filepath)
    docs = loader.load()

    # 创建一个文本分割器并分割文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)

    # 对每个json文件，都生成一份新的向量库
    vectorstore_path = os.path.join(faiss_folder, json_file.replace('.json', '.faiss'))
    if not os.path.exists(vectorstore_path):
        vector_store = FAISS.from_documents(docs, embedding)
        vector_store.save_local(vectorstore_path)
    else:
        vector_store = FAISS.load_local(vectorstore_path, embeddings=embedding)
    
    # print(output_file)
    
    txt_file_2 = json_file.replace('_2.json', '_1.txt')  
    filepath_2 = os.path.join(txt_folder_2, txt_file_2)
    # 读取filepath_2的内容到一个列表
    with open(filepath_2, 'r') as f:
        lines = f.readlines()
    # 跳过第一行
    lines = lines[1:]

    # 遍历处理数据的每一项
    for i, (d, line) in enumerate(zip(data, lines)):          
        # 从data中获取人名和关系
        person1, gender1, person2, gender2, relationship = d
        if person2 == 'unknown':
            person2 = 'an unknown person'
        # 使用data中的信息构建查询字符串
        query = f"Based on this sentence, we can infer the following quintuple relationships: '[\"{person1}\",\"{gender1}\",\"{person2}\",\"{gender2}\",\"{relationship}\"]'. This implies that {person1} is a {gender1}, {person2} is a {gender2}, and their relationship is {relationship}. I want to ask, first, are there any extraction errors for the two people in this sentence who fit the {relationship} relationship? If there is an error, please find two people who fit the {relationship} relationship from this sentence. Secondly, if there is an \"unknown\", please try to find out what \"unknown\" represents, unless it cannot be found. Thirdly, answer with the full names of the individuals. Please always provide answers in this quintuple format: '[\"Person's Name\",\"gender\",\"Person's Name\",\"gender\",\"relationship\"]'."
        # 清除行两端的空格并进行相似性搜索
        prompt = hub.pull("rlm/rag-prompt")
        docs = vector_store.similarity_search(line.strip())  
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 10})
        rag_chain = (
            {"context": retriever|format_docs , "question": RunnablePassthrough()}
            | prompt
        )
        msg=rag_chain.invoke(query).messages[0].content
        response = llm.invoke(msg).content
        print(response)

        query2 = f'Please analyze the following sentence and extract the earliest explicit time-related information. If multiple timestamps are present, choose the earliest one. If only the year is mentioned, output it in the format \'YYYY-MM\', with \'MM\' representing the missing month. If only the month is mentioned, output it in the format \'YYYY-MM\', with \'YYYY\' representing the missing year. If both the year and the month are mentioned, output the time information in the format \'YYYY-MM\'. If there is no explicit time-related information that includes both the year and the month, output \'YYYY-MM\' to indicate that both the year and the month are unknown.Always remember to answer in the format \'YYYY-MM\'.\nSentence: "{line.strip()}".'
        timestamp = llm.invoke(query2).content
        # print(timestamp)
        # 检查timestamp格式并提取YYYY-MM部分
        timestamp_pattern = re.compile(r'^(\d{4}-(0?[1-9]|1[0-2]|MM))(-\d{1,2}|-\d{1,2})?$')
        match = timestamp_pattern.match(timestamp)

        try:
            response_list = json.loads(response)
        except Exception:
            should_process = False
        else:
            should_process = True

        if should_process:
            # 检查response_list是否是一个5元组
            if len(response_list) != 5:
                should_process = False
            # 检查response_list中的关系是否与data[i]中的关系相同
            elif response_list[4] != d[4]:
                should_process = False
            # 检查response_list中的1和3位置是否是性别
            elif response_list[1] not in ['male', 'female', 'unknown'] or response_list[3] not in ['male', 'female', 'unknown']:
                should_process = False

            if should_process:
                data[i] = response_list


        if match:
            # 提取YYYY-MM部分
            year_month = match.group(1)
            data[i].append(year_month)
            # print("提取到时间" + str(data[i]))
        else:
            data[i].append('YYYY-MM')
            # print("没有提取到--------------时间" + str(data[i]))

        

    # 将结果保存到新的文件
    output_file = json_file.replace('_2.json', '_3.json')
    with open(os.path.join(output_folder, output_file), 'w') as file:
        json.dump(data, file, indent=4)