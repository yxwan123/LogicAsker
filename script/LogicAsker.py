import importlib
import inference_problems
import logic_inference_lib as lib
import rules
from tqdm.auto import tqdm
import inference_problems as ip
from splits import generate_training_and_test_sets_iid
# import openai
import json
import time
import pandas as pd
import numpy as np
# import requests
import random
import re
# import transformers
# import torch

importlib.reload(inference_problems)
importlib.reload(lib)
importlib.reload(rules)
rules.precompute_rules()

class Bot:
    def __init__(self, keypath=None, token=None, patience=100000000) -> None:
        if token is not None:
            self.key = token
        else:
            with open(keypath, "r") as f:
                self.key = [x.replace("\n", "") for x in f.readlines() if len(x) > 4]
        self.patience = patience

    def ask(self, question):
        succeed = False
        counter = 0
        while not succeed:
            try:
                counter += 1
                if counter > self.patience:
                    break
                ans = self.query(question)
                succeed = True

            except Exception as e:
                timer = counter if counter <= 20 else 1200
                print(f"{e}, wait for {timer} seconds")
                time.sleep(timer)
                continue

        if not succeed:
            raise Exception("OpenAI API failed")
        return ans
    
    def query(self, question):
        raise NotImplementedError
 

############################################
class GPT4(Bot):
    def __init__(self, keypath=None, token=None, patience=20) -> None:
        super().__init__(keypath, token, patience)
        openai.api_key = self.key[0]

    def query(self, question):
        completion = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                    {"role": "user", "content": question},
                                ])
        return completion.choices[0].message["content"]
    

class ChatGPT(Bot):
    def __init__(self, keypath=None, token=None, patience=20) -> None:
        super().__init__(keypath, token, patience)
        openai.api_key = self.key[0]

    def query(self, question):
        completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                    {"role": "user", "content": question},
                                ])
        return completion.choices[0].message["content"]
    

class GPT3(Bot):
    def __init__(self, keypath=None, token=None, patience=20) -> None:
        super().__init__(keypath, token, patience)
        openai.api_key = self.key[0]

    def query(self, question):
        completion = openai.Completion.create(
                                    model="text-curie-001",
                                    prompt=question,
                                    temperature=0.9,
                                    max_tokens=150,
                                    top_p=1,
                                    frequency_penalty=0.0,
                                    presence_penalty=0.6,
                                    )
        return completion.choices[0].text
    

class FakeBot(Bot):
    def __init__(self, keypath=None, token=None, patience=None, error=False, ans="yes") -> None:
        self.patience = 20
        self.error = error
        self.ans = ans

    def query(self, question):
        return self.ans
    
class Guanaco(Bot):
    def __init__(self, keypath=None, token=None, patience=100000000) -> None:
        self.API_URL = "https://api-inference.huggingface.co/models/timdettmers/guanaco-33b-merged"
        with open(keypath, "r") as f:
            self.key = [x.replace("\n", "") for x in f.readlines()]
        self.patience = patience

    def query(self, question):
        payload = {"inputs": question}
        headers = {"Authorization": f"Bearer {random.choice(self.key)}"}
        response = requests.post(self.API_URL, headers=headers, json=payload)
        return response.json()[0]["generated_text"].replace(question + "\n", "")
    
    
class Vicuna(Bot):
    def __init__(self, keypath=None, token=None, patience=1) -> None:
        self.model = "vicuna-13b-v1.3"
        openai.api_key = "EMPTY" 
        openai.api_base = "http://localhost:8000/v1"
        self.patience = patience

    def query(self, question):
        completion = openai.ChatCompletion.create(
                                        model=self.model,
                                        messages=[{"role": "user", "content": question}]
                                    )
        return completion.choices[0].message.content

BOT_DICT = {"gpt3": GPT3, "gpt4": GPT4, "chatgpt": ChatGPT, "guanaco": Guanaco, "fake": FakeBot, "vicuna": Vicuna}

############################################
def get_rules(rule_names):
    return [x for x in lib.ALL_INFERENCE_RULES if x.rule_name in rule_names]


def reformat_query(input_strings, conservative=False):
    "reformat distributive laws in the query to make it more readable"
    have_pattern = lambda pattern, string: len(re.findall(pattern, string, re.IGNORECASE)) > 0
    pattern1 = r'([A-Za-z0-9 _]+) and ([A-Za-z0-9 _]+) or \1 and ([A-Za-z0-9 _]+)'
    pattern2 = r'([A-Za-z0-9 _]+) or ([A-Za-z0-9 _]+) and \1 or ([A-Za-z0-9 _]+)'
    pattern3 = r'([A-Za-z0-9 _]+) and ([A-Za-z0-9 _]+) or ([A-Za-z0-9 _]+)'
    pattern4 = r'([A-Za-z0-9 _]+) or ([A-Za-z0-9 _]+) and ([A-Za-z0-9 _]+)'
    patter_not1 = r'([A-Za-z0-9 _]+) the claim that ([A-Za-z0-9 _]+) cannot both be true'
    patter_not2 = r"neither ([A-Za-z0-9 _]+) nor ([A-Za-z0-9 _]+)"

    def reformat_string(input_string):
        output_string = input_string
        if have_pattern(pattern1, input_string) and not (have_pattern(patter_not1, input_string) or have_pattern(patter_not2, input_string)):
            output_string = re.sub(pattern1, r'either \1 and \2, or \1 and \3', input_string, flags=re.IGNORECASE)
        elif have_pattern(pattern2, input_string) and not (have_pattern(patter_not1, input_string) or have_pattern(patter_not2, input_string)):
            output_string = re.sub(pattern2, r'both \1 or \2 and \1 or \3 are true', input_string, flags=re.IGNORECASE)
        elif have_pattern(pattern3, input_string) and not (have_pattern(patter_not1, input_string) or have_pattern(patter_not2, input_string)):
            output_string = re.sub(pattern3, r'\1, and either \2 or \3', input_string)
        elif have_pattern(pattern4, input_string) and not (have_pattern(patter_not1, input_string) or have_pattern(patter_not2, input_string)):
            output_string = re.sub(pattern4, r'\1, or \2 and \3', input_string)
        if output_string != input_string:
            print("============reformated============")
        return output_string
    
    if not (have_pattern(pattern1, input_strings) or have_pattern(pattern2, input_strings)) and conservative:
        return input_strings
    
    input_strings = input_strings.split('. ')
    for i in range(len(input_strings)):
        input_strings[i] = input_strings[i][0].lower() + input_strings[i][1:]
        input_strings[i] = reformat_string(input_strings[i])
        input_strings[i] = input_strings[i][0].upper() + input_strings[i][1:]
    return ". ".join(input_strings)


def gen_cases(n, type, category, length, rules=None, fallacies=None, explanation=False):
    """
    n: number of cases
    type: 3a (formal) or 3b (natural language)
    category: inference, contradiction, unrelated
    rules: list of rules
    fallacy: list of (fallacy) rules
    return: a list of [question, answer, inference problem expression]
    """
    real_n = 2*n
    len_distribution = [0]*length 
    len_distribution[length - 1] = 1
    result = []
    
    while len(result) < n:
        data = generate_training_and_test_sets_iid(10, 10, real_n, 1, [type,] , len_distribution, inference_types=[category], rules=rules, first_rules=fallacies, name_rule=explanation)
        if type == "3a3b":
            if ip.infer_length(data[0][0].problem, "inference") != length:
                continue
            else:
                result.append([data[0][0], data[0][1], category])

        for d in data[0]:
            if ip.infer_length(d.problem, "inference") != length:
                continue
            if explanation:
                result.append([d, category])
            else:
                result.append([d.inputs, "yes" if category == "inference" and fallacies is None else "no", str(d.problem)])
        
    assert(len(result) >= n)
    result = result[:n]
    for i in range(len(result)):
        result[i][0] = reformat_query(result[i][0])
    return result

def parse_case(case, type, fallacy=None):
    tmp = "We can infer that:" if case[0][-1]=="inference" and not fallacy else "We cannot infer that:"
    if type=="3b":
        prompt = case[0][0].inputs.replace("Can we infer the following from them? Answer yes or no: If we can, name the inference rule being used:", tmp)
    elif type=="3a":
        prompt = case[0][0].inputs.replace("Can we infer", tmp)
        prompt = prompt.replace(" from them? Answer yes or no: If possible, name the inference rules being used at each step", "")
    else:
        prompt1 = parse_case([[case[0][0], case[0][-1]]], "3a", fallacy)
        prompt2 = parse_case([[case[0][1], case[0][-1]]], "3b", fallacy)

        idx = prompt2.find("Because")
        prompt2 = prompt2[:idx] + " This is because the inference can be translate into the following logical form: "
        idx1 = prompt1.find(". Because we can infer")
        idx2 = prompt1.find(" via")
        prompt1 = prompt1[:idx1] + prompt1[idx2:]
        return prompt2 + prompt1

    prompt += " Because "
    if case[0][1]=="contradiction" or (case[0][1]=="inference" and not fallacy):
            prompt += case[0][0].targets[0].lower() + case[0][0].targets[1:]
    elif case[0][1]=="unrelated":
            prompt += "the conclusion is not related to the premises."
    else:
            prompt += "this is a logical fallacy of"
            idx = case[0][0].targets.find("via")
            prompt += case[0][0].targets[idx + 3:].replace(" rule", "")
    idx = prompt.find("Therefore, the answer is")
    if idx == -1:
        idx = prompt.find("Therefore the answer is")
    prompt = prompt[:idx]
    return prompt + "\n\n"

def gen_prompt(type, category, rule, fallacy=None):
    """
    type: 3a (formal) or 3b (natural language)
    category: inference, contradiction, unrelated
    rules: list of rules
    fallacy: list of rules
    return: a list of [question, answer, inference problem expression]
    """
    case = gen_cases(1, type, category, 1, rule, fallacy, explanation=True)
    return parse_case(case, type, fallacy)


################################## version 1 util ################################
def dict_to_df(d, parent_keys=None, result=None):
    if result is None:
        result = []
    if parent_keys is None:
        parent_keys = []

    for key, value in d.items():
        if isinstance(value, dict):
            dict_to_df(value, parent_keys + [key], result)
        else:
            result.append({**{f"level_{i}": k for i, k in enumerate(parent_keys)}, "level_{}".format(len(parent_keys)): key, "value": value})
    return pd.DataFrame(result)

def exp1_json_to_df(data_path='./data/dataframe_exp1.json', have_ans=False):
    # load dataframe_exp1.json and dataframe_exp2.json
    with open(data_path) as f:
        data = json.load(f)["data"]
    
    # exp1 data
    data = dict_to_df(data)

    # Shift the columns as specified
    fallacy_rows = data['level_1'] == 'fallacy'
    data.loc[fallacy_rows, ['level_3', 'level_4', 'level_5']] = data.loc[fallacy_rows, ['level_5', 'level_3', 'level_4']].values

    if have_ans:
        data = data[data['level_5'] != 'acc'].reset_index(drop=True)
        data['value'] = data['value'].apply(lambda x: [np.nan]*10 if not x else x)
        merged_rows = []

        # Iterate through rows in pairs and concatenate their 'value' columns
        for i in range(0, len(data), 2):
            row1 = data.iloc[i]
            row2 = data.iloc[i + 1]
            merged_row = row1.copy()
            merged_row['ans'] = row2['value']
            merged_rows.append(merged_row)

        # Concatenate the merged rows to create the new DataFrame
        data = pd.concat(merged_rows, axis=1).T.reset_index(drop=True)
        data["value"] = data.apply(lambda x: [[x["value"][i], x["ans"][i]] for i in range(len(x["value"]))], axis=1)
        data = data.drop(columns=["ans"]).explode("value").drop(columns=["level_5"])

        data["target"] = data["value"].apply(lambda x: x[0][1])
        data["ans"] = data["value"].apply(lambda x: x[1])
        data["value"] = data["value"].apply(lambda x: x[0][0])

    else:
        data = data[data["level_5"] != "a"].reset_index(drop=True).explode(["value"]).drop(columns=["level_5"])
        # extract the first and second value of the value column
        data["target"] = data["value"].apply(lambda x: x[1])
        data["value"] = data["value"].apply(lambda x: x[0])
        data["ans"] = np.nan

    data.rename(columns={"value": "question", "level_0": "logic", "level_1": "rule_category", "level_2": "rule", "level_3": "problem type", "level_4": "language"}, inplace=True)
    return data

def exp2_json_to_df(data_path='./data/dataframe_exp2.json', have_ans=False):
    # load dataframe_exp1.json and dataframe_exp2.json
    with open(data_path) as f:
        data = json.load(f)["data"]
    
    # exp1 data
    data = dict_to_df(data)

    if have_ans:
        data = data[data['level_3'] != 'acc'].reset_index(drop=True)

        merged_rows = []

        # Iterate through rows in pairs and concatenate their 'value' columns
        for i in range(0, len(data), 2):
            row1 = data.iloc[i]
            row2 = data.iloc[i + 1]
            merged_row = row1.copy()
            merged_row['ans'] = row2['value']
            merged_rows.append(merged_row)

        # Concatenate the merged rows to create the new DataFrame
        data = pd.concat(merged_rows, axis=1).T.reset_index(drop=True)
        data["value"] = data.apply(lambda x: [[x["value"][i], x["ans"][i]] for i in range(len(x["value"]))], axis=1)
        data = data.drop(columns=["ans"]).explode("value").drop(columns=["level_3"])

        data["target"] = data["value"].apply(lambda x: x[0][1])
        data["ans"] = data["value"].apply(lambda x: x[1])
        data["value"] = data["value"].apply(lambda x: x[0][0])

    else:
        data = data[data["level_3"] != "a"].reset_index(drop=True).explode(["value"]).drop(columns=["level_3"])
        # extract the first and second value of the value column
        data["target"] = data["value"].apply(lambda x: x[1])
        data["value"] = data["value"].apply(lambda x: x[0])
        data["ans"] = np.nan

    data.rename(columns={"value": "question", "level_0": "type", "level_1": "length", "level_2": "language"}, inplace=True)
    return data

################################## experiment ################################
def do_experiment(data, bot_name, key_path=None, token=None, patience=20, exp_name="exp1", prompt=None):
        data = data.copy()
        bot = BOT_DICT[bot_name](keypath=key_path, token=token, patience=patience)
        # print(bot.ask("hello"))
        for i in tqdm(range(len(data))):
                if isinstance(data.iloc[i]["ans"], str) and data.iloc[i]["ans"] != "":
                        continue
                question = data.iloc[i]["question"] if prompt is None else prompt + data.iloc[i]["question"]
                ans = bot.ask(question)
                data.loc[i, "ans"] = ans
                if i % 3 == 0:
                        data.to_csv(f"./data/{exp_name}_ans_{bot_name}.csv", index=False)
        data.to_csv(f"./data/{exp_name}_ans_{bot_name}.csv", index=False)


def correct(target, ans):
    if isinstance(ans, float):
        return 0
    ans = ans.replace(".", " ").replace(",", " ").replace("*", " ").lower().replace("there is no", "").replace("there are no", "").replace("there exists no", "").replace("there exist no", "")
    ans = " ".join(ans.split()[:20] + ans.split()[-20:])
    target = target + " " if target == "no" else target
    inverse = {"yes": "no ", "no ": "yes"}
    if target in ans and inverse[target] not in ans:
        return 1
    else:
        return 0
    
def respond(ans):
    if isinstance(ans, float):
        return 0
    ans = ans.replace(".", " ").replace(",", " ").replace("*", " ").lower().replace("there is no", "").replace("there are no", "").replace("there exists no", "").replace("there exist no", "")
    ans = " ".join(ans.split()[:20] + ans.split()[-20:])
    if "yes" in ans and "no " in ans:
        return 0
    elif "yes" in ans or "no " in ans:
        return 1
    elif "please provide" in ans:
        return 1
    else:
        return 0

    
def calculate_acc(data, index):
    return data.groupby(['logic', 'rule_category', 'rule', 'problem type', 'language']).get_group(index).apply(lambda x: correct(x["target"], x["ans"]), axis=1).mean()

def calculate_respond(data, index):
    return data.groupby(['logic', 'rule_category', 'rule', 'problem type', 'language']).get_group(index).apply(lambda x: respond(x["ans"]), axis=1).mean()

def get_stat(data):
    data.replace(np.nan, "nan", inplace=True)
    gp_index = 0
    gp_acc = 0
    gp_respond = 0
    res = []
    for i in range(len(data)):
        if tuple(data.loc[i, ['logic', 'rule_category', 'rule', 'problem type', 'language']].values) == gp_index:
            continue
        else:
            gp_index = tuple(data.loc[i, ['logic', 'rule_category', 'rule', 'problem type', 'language']].values)
            gp_acc = calculate_acc(data, gp_index)
            gp_respond = calculate_respond(data, gp_index)
            if gp_index[4] == "3b":
                res.append(list(gp_index) + [gp_acc, gp_respond]) 

    return pd.DataFrame(res).rename(columns={0: 'logic', 1: 'rule_category', 2: 'rule', 3: 'problem type', 4: 'language', 5: 'acc', 6: 'respond'})

def get_stat_exp0(data):
    return [data.apply(lambda x: correct(x["target"], x["ans"]), axis=1).mean(), data.apply(lambda x: respond(x["ans"]), axis=1).mean()]



    