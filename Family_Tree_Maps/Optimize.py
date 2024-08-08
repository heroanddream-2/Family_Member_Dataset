import os
import json
import re
from langchain_openai import ChatOpenAI

def validate_response(response_content):
    try:
        cleaned_response_content = response_content.replace('\\"', "'")
        response_list = json.loads(cleaned_response_content)
    except json.JSONDecodeError:
        return False

    for entry in response_list:
        if len(entry) != 5:
            return False
        if not isinstance(entry[0], str) or not isinstance(entry[2], str):
            return False
        if entry[1] not in ['male', 'female', 'unknown'] or entry[3] not in ['male', 'female', 'unknown']:
            return False
        if not re.match(r'^[a-zA-Z]+-[a-zA-Z]+$', entry[4]):
            return False

    return True

def construct_prompt(data):
    list_content = json.dumps(data, indent=4, ensure_ascii=False)
    prompt = f"""
The data format for each entry is: ["first person", "first person's gender", "second person", "second person's gender", "relationship1-relationship2 "].

Explanation:
- "first person": The name of the first person in the relationship.
- "first person's gender": The gender of the first person (male or female).
- "second person": The name of the second person in the relationship.
- "second person's gender": The gender of the second person (male or female).
- "relationship1-relationship2": The type of relationship between the first and second person, where "relationship1" describes the first person's role, and "relationship2" describes the second person's role. For example, "father-son" means the first person is the father and the second person is the son.

Please use your extensive knowledge base to assess whether the provided information is correct. If you find any inaccuracies, please correct them.

Attention:
Examine each entry for any inaccuracies in the names (ensure they are specific and correct names, do not include anything else), genders (ensure they are correct), and the "relationship1-relationship2" field (ensure the relationship is correct and the directionality of the relationship is accurate).

Here is the list:
{list_content}

Give the modified result.
"""
    return prompt

def process_files(folder_path):
    llm = ChatOpenAI(api_key="your_api_key", base_url="http://127.0.0.1:8005/v1", model_name="llama3-8b-instruct", max_tokens=2048, temperature=0.0)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                updated_data = []
                  
                for entry_group in data:
                    prompt = construct_prompt(entry_group)
                    response = llm.invoke(prompt)
                    if validate_response(response.content):
                        updated_data.append(json.loads(response.content))
                    else:
                        updated_data.append(entry_group)
                
                new_filename = filename.replace('.json', 'fine.json')
                new_file_path = os.path.join(folder_path, new_filename)
                with open(new_file_path, 'w', encoding='utf-8') as out_file:
                    json.dump(updated_data, out_file, indent=4, ensure_ascii=False)

# 调用函数，处理指定文件夹下的文件
process_files('/home/niuhaojia/file3')
