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
from apis import *
# import transformers
# import torch

importlib.reload(inference_problems)
importlib.reload(lib)
importlib.reload(rules)
rules.precompute_rules()


def get_rules(rule_names):
    return [x for x in lib.ALL_INFERENCE_RULES if x.rule_name in rule_names]


def reformat_query(input_strings, conservative=False):
    "reformat distributive laws in the query to make it more readable, input is a string"
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
        # if output_string != input_string:
        #     print("============reformated============")
        return output_string
    
    def replace_string_pattern(text):
        "associative laws"
        # Define the regex pattern to match the specific string format
        pattern1 = r'(.*): (.+?) or (.+?) or (.+?)\. Can we infer the following from them\? Answer yes or no: (.+?) or (.+?) or (.+?)\.'
        # and instead of or
        pattern2 = r'(.*): (.+?) and (.+?) and (.+?)\. Can we infer the following from them\? Answer yes or no: (.+?) and (.+?) and (.+?)\.'
        
        # Define the replacement pattern
        replacement1 = r'\1: \2, or \3 or \4. Can we infer the following from them? Answer yes or no: \5 or \6, or \7.'
        # and instead of or
        replacement2 = r'\1: \2, and \3 and \4. Can we infer the following from them? Answer yes or no: \5 and \6, and \7.'
        # return re.match(pattern1, text) or re.match(pattern2, text)
        if re.match(pattern1, text) or re.match(pattern2, text):
            # print(text)
            new_text = re.sub(pattern1, replacement1, text)
            new_text = re.sub(pattern2, replacement2, new_text)
            return new_text
        else:
            return text
    
    if not (have_pattern(pattern1, input_strings) or have_pattern(pattern2, input_strings)) and conservative:
        return replace_string_pattern(input_strings)
    
    input_strings = input_strings.split('. ')
    for i in range(len(input_strings)):
        input_strings[i] = input_strings[i][0].lower() + input_strings[i][1:]
        input_strings[i] = reformat_string(input_strings[i])
        input_strings[i] = input_strings[i][0].upper() + input_strings[i][1:]
    return replace_string_pattern(". ".join(input_strings))


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

    if not explanation:
        for i in range(len(result)):
            # print(result[i][0])
            result[i][0] = reformat_query(result[i][0])
        return result
    else:
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

def gen_prompt(category, rule, type="3b", fallacy=None):
    """
    type: 3a (formal) or 3b (natural language)
    category: inference, contradiction, unrelated
    rules: list of rules
    fallacy: list of rules
    return: a list of [question, answer, inference problem expression]
    """
    case = gen_cases(1, type, category, 1, rule, fallacy, explanation=True)
    case = parse_case(case, type, fallacy)
    return reformat_query(case)


################################## experiment ################################
def do_experiment(data, bot_name, key_path=None, patience=20, exp_name="exp1", prompt=None):
        data = data.copy()
        bot = BOT_DICT[bot_name](keypath=key_path, patience=patience)
        # print(bot.ask("hello"))
        for i in tqdm(range(len(data))):
                if isinstance(data.iloc[i][f"ans_{exp_name}"], str) and data.iloc[i][f"ans_{exp_name}"] != "":
                        continue
                question = data.iloc[i]["question"] if prompt is None else prompt + data.iloc[i]["question"]
                ans = bot.ask(question)
                data.loc[i, f"ans_{exp_name}"] = ans
                if i % 3 == 0:
                        data.to_csv(f"./data/{exp_name}_ans_{bot_name}.csv", index=False)
        data.to_csv(f"./data/{exp_name}_ans_{bot_name}.csv", index=False)


# def correct(target, ans):
#     if isinstance(ans, float):
#         return 0
#     ans = ans.replace(".", " ").replace(",", " ").replace("*", " ").lower().replace("there is no", "").replace("there are no", "").replace("there exists no", "").replace("there exist no", "")
#     ans = " ".join(ans.split()[:20] + ans.split()[-20:])
#     target = target + " " if target == "no" else target
#     inverse = {"yes": "no ", "no ": "yes"}
#     if target in ans and inverse[target] not in ans:
#         return 1
#     else:
#         return 0
    
def correct(gtruth, ans):
    gtruth = gtruth.lower().replace(",", "").replace(".", "")
    ans = ans.lower().replace(",", "").replace(".", "")
    # if "yes " in ans and "no " in ans:
    #     return 0
    # if "==yes==" in ans and "==no==" in ans:
    #      return 0
    # if (gtruth == "yes" and "==yes==" in ans) or (gtruth == "no" and "==no==" in ans):
    #     return 1
    # elif (gtruth == "yes" and "==no==" in ans) or (gtruth == "no" and "==yes==" in ans):
    #     return 0
    if gtruth in ans or ans in gtruth:
        return 1
    return 0

# def respond(ans):
#     if isinstance(ans, float):
#         return 0
#     ans = ans.replace(".", " ").replace(",", " ").replace("*", " ").lower().replace("there is no", "").replace("there are no", "").replace("there exists no", "").replace("there exist no", "")
#     ans = " ".join(ans.split()[:20] + ans.split()[-20:])
#     if "yes" in ans and "no " in ans:
#         return 0
#     elif "yes" in ans or "no " in ans:
#         return 1
#     elif "please provide" in ans:
#         return 1
#     else:
#         return 0

    
def calculate_acc(data, index, exp_name):
    return data.groupby(['logic', 'rule_category', 'rule', 'problem']).get_group(index).apply(lambda x: correct(x["answer"], x[f"ans_{exp_name}"]), axis=1).mean()

# def calculate_respond(data, index):
#     return data.groupby(['logic', 'rule_category', 'rule', 'problem type', 'language']).get_group(index).apply(lambda x: respond(x["ans"]), axis=1).mean()

def get_stat(data):
    data.replace(np.nan, "nan", inplace=True)
    gp_index = 0
    gp_acc = 0
    res = []
    # for i in range(1, len(data) + 1):
    for i in range(len(data)):
        if tuple(data.loc[i, ['logic', 'rule_category', 'rule', 'problem']].values) == gp_index:
            continue
        else:
            gp_index = tuple(data.loc[i, ['logic', 'rule_category', 'rule', 'problem']].values)
            gp_acc = [calculate_acc(data, gp_index, exp) for exp in ["ft_weak", "ft_all"]]
            # gp_acc = [calculate_acc(data, gp_index, exp) for exp in ["zero_shot", "zero_shot_cot", "random_icl", "weak"]]
            res.append(list(gp_index) + gp_acc) 

    # return pd.DataFrame(res).rename(columns={0: 'logic', 1: 'rule_category', 2: 'rule', 3: 'problem', 4: 'acc_zero_shot', 5: 'acc_zero_shot_cot', 6: 'acc_random_icl', 7: 'acc_weak'})
    return pd.DataFrame(res).rename(columns={0: 'logic', 1: 'rule_category', 2: 'rule', 3: 'problem', 4: 'acc_ft_weak', 5: 'acc_ft_all'})
# def get_stat_exp0(data):
#     return [data.apply(lambda x: correct(x["target"], x["ans"]), axis=1).mean(), data.apply(lambda x: respond(x["ans"]), axis=1).mean()]



    