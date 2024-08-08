import os
import re
import json
import spacy
import neuralcoref
from multiprocessing import Pool

# 加载spaCy模型和NeuralCoref
nlp = spacy.load("en_core_web_lg")
neuralcoref.add_to_pipe(nlp)

# 处理每个文件的函数
def process_file(args):
    folder_path, filename, folder_1, folder_2 = args
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        name = lines[1].strip()
        text = ''.join(lines[3:])
    
    doc_2 = nlp(name)
    if any(ent.label_ == "PERSON" for ent in doc_2.ents):
        # 使用spaCy进行文本处理
        doc = nlp(text)

        # 获取共指关系的结果
        she_count = sum(1 for token in doc if token.text.lower() == 'she')
        he_count = sum(1 for token in doc if token.text.lower() == 'he')

        # 合并第二段代码
        if she_count >= he_count:
            sex = "female"
            pronoun_list = ["her", name]
        else:
            sex = "male" 
            pronoun_list = ["his", name]

        # 关键词
        keywords = ["husband", "wife", "brother", "sister", "daughter", "son", "father", "mother", "friend", "colleague","aunt","uncle","nephew","niece","granddaughter","grandson","grandfather","grandmother","father-in-law","mother-in-law","son-in-law","daughter-in-law","cousin"]

        # 识别命名实体
        ents = [ent for ent in doc.ents]

        # 遍历句子
        # sentence_list = []
        data_list = []
        for sent in doc.sents:
            person_entities = [ent for ent in sent.ents if ent.label_ == 'PERSON']
            pronoun_count = sum(token.text.lower() in ['he', 'she'] for token in sent)
            flag = False
            sent_tokens = [token for token in sent]
            for i, token in enumerate(sent_tokens):
                keyword_count = sum(token_2.text.lower() in keywords for token_2 in sent)
                if keyword_count == 1 and token.text.lower() in keywords:
                    # 检查关键词前一个词
                    # if i > 0 and sent_tokens[i - 1].text.endswith("'s") and name.find(sent_tokens[i - 2].text)== -1:
                    flag2 = False
                    if i >= 2 and sent_tokens[i - 1].text.endswith("'s") :
                        person_entity = sent_tokens[i - 2].text
                        flag2 = True
                    elif flag2 == False:
                        for ent in ents:
                                if ((ent.start - token.i) == 2) and (ent.label_ == "PERSON") and i > 2 and (sent_tokens[i - 2].text + ' ' + sent_tokens[i - 1].text == "and his" or sent_tokens[i - 2].text + ' ' + sent_tokens[i - 1].text == "and her"):
                                    person_entity = ent.text
                                    flag2 = True
                                    break
                    if flag2 == False:
                        person_entity = name
                    # 检查关键词后面近距离的实体
                    for ent in ents:
                        if (ent.start - token.i) <= 2 and (ent.start - token.i) >= 1 and ent.label_ == "PERSON" and "of" not in [t.text for t in doc[token.i : ent.start]] and len(person_entities)+pronoun_count >= 2 :
                            if token.text == "husband":
                                relation = (person_entity, "female", ent.text, "male", "wife-husband")
                                flag = True
                            elif token.text == "wife":
                                relation = (person_entity, "male", ent.text, "female", "husband-wife")
                                flag = True
                            elif token.text == "friend":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "unknown", "friend-friend")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "unknown", "friend-friend")
                                    flag = True
                            elif token.text == "colleague":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "unknown", "colleague-colleague")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "unknown", "colleague-colleague")
                                    flag = True
                            elif token.text == "brother":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "male", "brother-brother")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "male", "sister-brother")
                                    flag = True
                            elif token.text == "sister":
                                if sex == "female":
                                    relation = (person_entity, "female", ent.text, "female", "sister-sister")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", ent.text, "female", "brother-sister")
                                    flag = True
                            elif token.text == "daughter":
                                if sex == "female":
                                    relation = (person_entity, "female", ent.text, "female", "mother-daughter")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", ent.text, "female", "father-daughter")
                                    flag = True
                            elif token.text == "son":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "male", "father-son")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "male", "mother-son")
                                    flag = True
                            elif token.text == "father":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "male", "son-father")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "male", "daughter-father")
                                    flag = True
                            elif token.text == "mother":
                                if sex == "female":
                                    relation = (person_entity, "female", ent.text, "female", "daughter-mother")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", ent.text, "female", "son-mother")
                                    flag = True
                            elif token.text == "aunt":
                                if sex == "female":
                                    relation = (person_entity, "female", ent.text, "female", "niece-aunt")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", ent.text, "female", "nephew-aunt")
                                    flag = True
                            elif token.text == "uncle":
                                if sex == "female":
                                    relation = (person_entity, "female", ent.text, "male", "niece-uncle")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", ent.text, "male", "nephew-uncle")
                                    flag = True
                            elif token.text == "nephew":
                                if sex == "female":
                                    relation = (person_entity, "female", ent.text, "male", "aunt-nephew")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", ent.text, "male", "uncle-nephew")
                                    flag = True
                            elif token.text == "niece":
                                if sex == "female":
                                    relation = (person_entity, "female", ent.text, "female", "aunt-niece")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", ent.text, "female", "uncle-niece")
                                    flag = True
                            elif token.text == "granddaughter":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "female", "grandfather-granddaughter")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "female", "grandmother-granddaughter")
                                    flag = True
                            elif token.text == "grandson":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "male", "grandfather-grandson")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "male", "grandmother-grandson")
                                    flag = True
                            elif token.text == "grandfather":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "male", "grandson-grandfather")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "male", "granddaughter-grandfather")
                                    flag = True
                            elif token.text == "grandmother":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "female", "grandson-grandmother")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "female", "granddaughter-grandmother")
                                    flag = True
                            elif token.text == "father-in-law":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "male", "son-in-law-father-in-law")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "male", "daughter-in-law-father-in-law")
                                    flag = True
                            elif token.text == "mother-in-law":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "female", "son-in-law-mother-in-law")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "female", "daughter-in-law-mother-in-law")
                                    flag = True
                            elif token.text == "son-in-law":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "male", "father-in-law-son-in-law")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "male", "mother-in-law-son-in-law")
                                    flag = True
                            elif token.text == "daughter-in-law":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "female", "father-in-law-daughter-in-law")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "female", "mother-in-law-daughter-in-law")
                                    flag = True
                            elif token.text == "cousin":
                                if sex == "male":
                                    relation = (person_entity, "male", ent.text, "unknown", "cousin-cousin")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", ent.text, "unknown", "cousin-cousin")
                                    flag = True


                    if flag == False and len(person_entities)+pronoun_count >= 2 and (i + 1 >= len(sent_tokens) or sent_tokens[i + 1].text != "of"):
                            if token.text == "husband":
                                relation = (person_entity, "female", "unknown", "male", "wife-husband")
                                flag = True
                            elif token.text == "wife":
                                relation = (person_entity, "male", "unknown", "female", "husband-wife")
                                flag = True
                            elif token.text == "friend":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "unknown", "friend-friend")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "unknown", "friend-friend")
                                    flag = True
                            elif token.text == "colleague":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "unknown", "colleague-colleague")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "unknown", "colleague-colleague")
                                    flag = True
                            elif token.text == "brother":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "male", "brother-brother")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "male", "sister-brother")
                                    flag = True
                            elif token.text == "sister":
                                if sex == "female":
                                    relation = (person_entity, "female", "unknown", "female", "sister-sister")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", "unknown", "female", "brother-sister")
                                    flag = True
                            elif token.text == "daughter":
                                if sex == "female":
                                    relation = (person_entity, "female", "unknown", "female", "mother-daughter")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", "unknown", "female", "father-daughter")
                                    flag = True
                            elif token.text == "son":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "male", "father-son")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "male", "mother-son")
                                    flag = True
                            elif token.text == "father":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "male", "son-father")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "male", "daughter-father")
                                    flag = True
                            elif token.text == "mother":
                                if sex == "female":
                                    relation = (person_entity, "female", "unknown", "female", "daughter-mother")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", "unknown", "female", "son-mother")
                                    flag = True
                            elif token.text == "aunt":
                                if sex == "female":
                                    relation = (person_entity, "female", "unknown", "female", "niece-aunt")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", "unknown", "female", "nephew-aunt")
                                    flag = True
                            elif token.text == "uncle":
                                if sex == "female":
                                    relation = (person_entity, "female", "unknown", "male", "niece-uncle")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", "unknown", "male", "nephew-uncle")
                                    flag = True
                            elif token.text == "nephew":
                                if sex == "female":
                                    relation = (person_entity, "female", "unknown", "male", "aunt-nephew")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", "unknown", "male", "uncle-nephew")
                                    flag = True
                            elif token.text == "niece":
                                if sex == "female":
                                    relation = (person_entity, "female", "unknown", "female", "aunt-niece")
                                    flag = True
                                else :
                                    relation = (person_entity, "male", "unknown", "female", "uncle-niece")
                                    flag = True
                            elif token.text == "granddaughter":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "female", "grandfather-granddaughter")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "female", "grandmother-granddaughter")
                                    flag = True
                            elif token.text == "grandson":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "male", "grandfather-grandson")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "male", "grandmother-grandson")
                                    flag = True
                            elif token.text == "grandfather":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "male", "grandson-grandfather")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "male", "granddaughter-grandfather")
                                    flag = True
                            elif token.text == "grandmother":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "female", "grandson-grandmother")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "female", "granddaughter-grandmother")
                                    flag = True
                            elif token.text == "father-in-law":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "male", "son-in-law-father-in-law")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "male", "daughter-in-law-father-in-law")
                                    flag = True
                            elif token.text == "mother-in-law":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "female", "son-in-law-mother-in-law")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "female", "daughter-in-law-mother-in-law")
                                    flag = True
                            elif token.text == "son-in-law":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "male", "father-in-law-son-in-law")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "male", "mother-in-law-son-in-law")
                                    flag = True
                            elif token.text == "daughter-in-law":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "female", "father-in-law-daughter-in-law")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "female", "mother-in-law-daughter-in-law")
                                    flag = True
                            elif token.text == "cousin":
                                if sex == "male":
                                    relation = (person_entity, "male", "unknown", "unknown", "cousin-cousin")
                                    flag = True
                                else :
                                    relation = (person_entity, "female", "unknown", "unknown", "cousin-cousin")
                                    flag = True

                    if flag and str(sent) not in [data["sentence"] for data in data_list]:
                        data = {"sentence": str(sent).replace('\n', ' ').strip(), "relation": relation}
                        data_list.append(data)
                        break
                    # elif flag == False and len(person_entities) ==2 and pronoun_count ==1:
                    #     relation = (person_entities[0].text, "unknown", person_entities[1].text, "unknown", "unknown-unknown")
                    #     data = {"sentence": str(sent), "relation": relation}
                    #     data_list.append(data)
                    #     break
        
        # 遍历data_list中的每个元素
        for data in data_list:
            # 将relation字段的元组转换为列表
            relation_list = list(data['relation'])
            # 遍历relation字段的位置0和2的元素
            for i in [0, 2]:
                # 遍历ents列表
                for ent in ents:
                    # 如果ent是一个人名，并且全名包含了人名的任何部分
                    if ent.label_ == 'PERSON' and all(part.lower() in ent.text.lower() for part in relation_list[i].split()):
                        # 用全名替换人名
                        full_name = ent.text
                        if full_name.endswith("'s"):
                            full_name = full_name[:-2]
                        if relation_list[i].endswith("'s"):
                            relation_list[i] = relation_list[i][:-2]
                        if len(full_name) > len(relation_list[i]):
                            relation_list[i] = full_name
                        break
            # 将relation字段的列表转换回元组
            data['relation'] = tuple(relation_list)

        # 将处理结果写入文件
        if data_list:
            with open(os.path.join(folder_1, filename.replace('.txt', '_1.txt')), 'w') as f1, open(os.path.join(folder_2, filename.replace('.txt', '_2.json')), 'w') as f2:
                f1.write(name + "\n")
                json_data = []
                for data in data_list:
                    f1.write(data["sentence"] + "\n")
                    json_data.append(data["relation"])
                json.dump(json_data, f2, indent=4)

# 主函数
def main(base_path, processes):
    args_list = []
    for folder in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, folder)) and folder.isdigit():
            folder_path = os.path.join(base_path, folder)
            folder_1 = folder_path + "_1"
            folder_2 = folder_path + "_2"

            if not os.path.exists(folder_1):
                os.makedirs(folder_1)
            if not os.path.exists(folder_2):
                os.makedirs(folder_2)

            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    args_list.append((folder_path, filename, folder_1, folder_2))

    # 使用进程池创建进程
    with Pool(processes=processes) as pool:
        pool.map(process_file, args_list)

if __name__ == "__main__":
    base_path = r"/data/niuhaojia/wikipedia"
    processes = 2  # 设置进程数，根据你的CPU核心数来定
    main(base_path, processes)