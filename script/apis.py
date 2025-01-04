from openai import OpenAI
import google.generativeai as genai
import time

class Bot:
    def __init__(self, keypath=None, patience=10) -> None:
        with open(keypath, "r") as f:
            self.key = f.read().replace("\n", "")
        self.patience = patience

    def ask(self, question, verbose=False):
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
                timer = counter * 3
                print(f"{e}, wait for {timer} seconds")
                time.sleep(timer)
                continue
        if verbose:
            print(f"Bot: {ans}")
        if not succeed:
            raise Exception("OpenAI API failed")
        return ans
    
    def query(self, question):
        raise NotImplementedError
 

############################################
class Gemini(Bot):
    def __init__(self, key_path, patience=1) -> None:
        super().__init__(key_path, patience)
        GOOGLE_API_KEY= self.key
        genai.configure(api_key=GOOGLE_API_KEY)
        self.name = "Gemini"
        self.file_count = 0
        
    def query(self, question):
        model = genai.GenerativeModel('gemini-1.5-flash-latest')    
        response = model.generate_content(question, stream=False, request_options={"timeout": 3000})
        response.resolve()
        return response.text


class GPT4(Bot):
    def __init__(self, key_path, patience=20, model="gpt-4-turbo") -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key)
        self.name="gpt4"
        self.model = model
        
    def query(self, question):
        
        response = self.client.chat.completions.create(
        model=self.model,
        messages=[
         {"role": "user", "content": question}
        ],
        max_tokens=500,
        temperature=0,
        )
        response = response.choices[0].message.content
        return response
    
class GPT4o(Bot):
    def __init__(self, key_path, patience=20, model="gpt-4o") -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key)
        self.name="gpt4o"
        self.model = model
        
    def query(self, question):
        
        response = self.client.chat.completions.create(
        model=self.model,
        messages=[
         {"role": "user", "content": question}
        ],
        max_tokens=500,
        temperature=0,
        )
        response = response.choices[0].message.content
        return response
    
class GPT4oMini(GPT4o):
    def __init__(self, key_path, patience=20, model="gpt-4o-mini") -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key)
        self.name="gpt4o-mini"
        self.model = model


class ChatGPT(GPT4):
    def __init__(self, key_path, patience=20, model="gpt-3.5-turbo-0125") -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key)
        self.name="chatgpt"
        self.model = model

class ChatGPTFTALL(GPT4):
    def __init__(self, key_path, patience=20, model="ft:gpt-3.5-turbo-0125:abc:ft1:9TvpAdvI") -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key)
        self.name="chatgpt"
        self.model = model
    
class ChatGPTFTWEAK(GPT4):
    def __init__(self, key_path, patience=20, model="ft:gpt-3.5-turbo-0125:abc:ft1p:9TvrN4ZX") -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key)
        self.name="chatgpt"
        self.model = model

class GPT3(GPT4):
    def __init__(self, key_path, patience=20, model="davinci-002") -> None:
        super().__init__(key_path, patience)
        self.client = OpenAI(api_key=self.key)
        self.name = "gpt3"
        self.model = model

    def query(self, question):
        response = self.client.completions.create(
        model=self.model,
        prompt=question,
        temperature=0.0,
        max_tokens=300,
        )
        response = response.choices[0].text
        return response
    

class FakeBot(Bot):
    def __init__(self, keypath=None, token=None, patience=None, error=False, ans="yes") -> None:
        self.patience = 20
        self.error = error
        self.ans = ans

    def query(self, question):
        return self.ans


# class Guanaco(Bot):
#     def __init__(self, keypath=None, token=None, patience=100000000) -> None:
#         self.API_URL = "https://api-inference.huggingface.co/models/timdettmers/guanaco-33b-merged"
#         with open(keypath, "r") as f:
#             self.key = [x.replace("\n", "") for x in f.readlines()]
#         self.patience = patience

#     def query(self, question):
#         payload = {"inputs": question}
#         headers = {"Authorization": f"Bearer {random.choice(self.key)}"}
#         response = requests.post(self.API_URL, headers=headers, json=payload)
#         return response.json()[0]["generated_text"].replace(question + "\n", "")
    
    
# class Vicuna(Bot):
#     def __init__(self, keypath=None, token=None, patience=1) -> None:
#         self.model = "vicuna-13b-v1.3"
#         openai.api_key = "EMPTY" 
#         openai.api_base = "http://localhost:8000/v1"
#         self.patience = patience

#     def query(self, question):
#         completion = openai.ChatCompletion.create(
#                                         model=self.model,
#                                         messages=[{"role": "user", "content": question}]
#                                     )
#         return completion.choices[0].message.content
BOT_DICT = {"gpt3": GPT3, "gpt4": GPT4, "chatgpt": ChatGPT, "fake": FakeBot, "gemini": Gemini, "chatgptftall": ChatGPTFTALL, "chatgptftweak": ChatGPTFTWEAK, "gpt4o": GPT4o}



ZERO_SHOT = "(Put the final answer in the form of ==yes== or ==no==) "

ZERO_SHOT_COT = """Please think step-by-step and answer the following question (put the final answer in the form of ==yes== or ==no==): """

RANDOM_ICL_PROMPT = """Q: Consider the following premises: If it snows, then Joseph is a politician. Can we infer the following from them? Answer yes or no: It is snowing if and only if Joseph is a politician. 

A: Let A be the claim that "it snows", B be the claim that "Joseph is a politician", then the premises are "if A then B, if B then A", which is equivalent to "A if and only if B" by the biconditional introduction rule. Therefore, we can infer that It is snowing if and only if Joseph is a politician. The answer is ==yes==.

Q: Consider the following premises: It is late and it is windy. Can we infer the following from them? Answer yes or no: It is windy.

A: Let A be the claim that "it is late", B be the claim that "it is windy", then the premises are "A and B". By the simplification rule, we can infer B. Therefore, we can infer that it is windy. The answer is ==yes==.

Q: Consider the following premises: Jessica is not running. Jessica is running or it is raining. Can we infer the following from them? Answer yes or no: Sarah is not happy.

A: Let A be the claim that "Jessica is running", B be the claim that "it is raining", C be the claim that "Sarah is happy", then the premises are "not A, A or B". We cannot infer C from the premises. The answer is ==no==.

Q: Consider the following premises: It is not raining. It is raining or it is late. Can we infer the following from them? Answer yes or no: It is not late.

A: Let A be the claim that "it is raining", B be the claim that "it is late", then the premises are "not A, A or B". We can infer "B", which is "it is late". Therefore, we cannot infer "it is not late". The answer is ==no==.

Q: """


WEAK_PROMPT = {
"claude3": """Q: Consider the following premises: Barbara is not bored or Susan is an artist. Can we infer the following from them? Answer yes or no: If Barbara were bored, then Susan is an artist.
    A: Let A be the claim that "Barbara is bored", B be the claim that "Susan is an artist", then the premises are "not A or B". By the conditional laws rule, we can infer "if A then B". Therefore, we can infer that If Barbara were bored, then Susan is an artist. The answer is ==yes==.

    Q: Consider the following premises: If David were happy, then Jessica is rich. If Jessica were rich, then David is happy. Can we infer the following from them? Answer yes or no: Jessica is rich if and only if David is happy.
    A: Let A be the claim that "David is happy", B be the claim that "Jessica is rich", then the premises are "if A then B, if B then A". By the biconditional introduction rule, we can infer "A if and only if B". Therefore, we can infer that Jessica is rich if and only if David is happy. The answer is ==yes==.

    Q: Consider the following premises: Richard is not a poet or William is a mechanic. Can we infer the following from them? Answer yes or no: The fact that Richard were a poet does not imply that William is a mechanic.
    A: Let A be the claim that "Richard is a poet", B be the claim that "William is a mechanic", then the premises are "not A or B". We cannot infer "not (A implies B)" from the premises, because that contradicts the premises. . The answer is ==no==.

    Q: Consider the following premises: If Richard plays tennis, then it is windy. If it is windy, then Richard is playing tennis. Can we infer the following from them? Answer yes or no: Richard is playing tennis.
    A: Let A be the claim that "Richard plays tennis", B be the claim that "it is windy", then the premises are "if A then B, if B then A". We cannot infer A from the premises, because the conclusion is not related to the premises. The answer is ==no==.

    Q: """,

"llama3": """Q: Consider the following premises: Jessica is not driving a car. Jessica is driving a car or it is overcast. Can we infer the following from them? Answer yes or no: It is overcast.
    A: Let A be the claim that "Jessica is driving a car", B be the claim that "it is overcast", then the premises are "not A, A or B". By the disjunctive syllogism rule, we can infer B. Therefore, we can infer that it is overcast. The answer is ==yes==.

    Q: Consider the following premises: Charles is not playing a game or James is playing squash. Can we infer the following from them? Answer yes or no: If Charles plays a game, then James is playing squash.
    A: Let A be the claim that "Charles plays a game", B be the claim that "James is playing squash", then the premises are "not A or B". By the conditional laws rule, we can infer "if A then B". Therefore, we can infer that If Charles plays a game, then James is playing squash. The answer is ==yes==.

    Q: Consider the following premises: If it is cloudy, then Linda is excited. If Richard listens to a song, then it is cloudy. Can we infer the following from them? Answer yes or no: The fact that Richard listens to a song does not imply that Linda is excited.
    A: Let A be the claim that "it is cloudy", B be the claim that "Linda is excited", C be the claim that "Richard listens to a song", then the premises are "if A then B, if C then A". We cannot infer "not (C implies B)" from the premises, because that contradicts the premises. The answer is ==no==.

    Q: Consider the following premises: If Charles reads a book, then it is overcast. Can we infer the following from them? Answer yes or no: Neither Charles is not reading a book nor it is overcast.
    A: Let A be the claim that "Charles reads a book", B be the claim that "it is overcast", then the premises are "if A then B". We cannot infer "not (not A or B)" from the premises, because that contradicts the premises. The answer is ==no==.

    Q: """,

"mixtral": """Q: Consider the following premises: If Patricia were a server, then Patricia is drinking water. Can we infer the following from them? Answer yes or no: Patricia is not a server or Patricia is drinking water.
A: Let A be the claim that "Patricia is a server", B be the claim that "Patricia is drinking water", then the premises are "if A then B". By the conditional laws rule, we can infer "not A or B". Therefore, we can infer that Patricia is not a server or Patricia is drinking water. The answer is ==yes==.

    Q: Consider the following premises: If John sleeps, then Charles is reading a book. John is sleeping. Can we infer the following from them? Answer yes or no: Charles is reading a book.
    A: Let A be the claim that "John sleeps", B be the claim that "Charles is reading a book", then the premises are "if A then B, A". By the modus ponens rule, we can infer B. Therefore, we can infer that Charles is reading a book. The answer is ==yes==.

    Q: Consider the following premises: If Robert drinks water, then Charles is listening to a song. If Charles listens to a song, then Robert is drinking water. Can we infer the following from them? Answer yes or no: It is not true that Robert is drinking water if and only if Charles is listening to a song.
    A: Let A be the claim that "Robert drinks water", B be the claim that "Charles is listening to a song", then the premises are "if A then B, if B then A". We cannot infer "not (A if and only if B)" from the premises, because that contradicts the premises. The answer is ==no==.

    Q: Consider the following premises: Patricia is climbing a mountain if and only if Richard is cooking. Can we infer the following from them? Answer yes or no: The claim that if Patricia climbs a mountain, then Richard is cooking and the claim that if Richard cooks, then Patricia is climbing a mountain cannot both be true.
    A: Let A be the claim that "Patricia climbs a mountain", B be the claim that "Richard is cooking", then the premises are "A if and only if B". We cannot infer "not (if A then B, and if B then A)" from the premises, because that contradicts the premises. The answer is ==no==.
    
    Q: """,

"gpt3": """Q: Consider the following premises: If it is overcast, then it is cloudy. If it is cloudy, then it is overcast. Can we infer the following from them? Answer yes or no: It is overcast if and only if it is cloudy.
    A: Let A be the claim that "it is overcast", B be the claim that "it is cloudy", then the premises are "if A then B, if B then A". By the biconditional introduction rule, we can infer "A if and only if B". Therefore, we can infer that It is overcast if and only if it is cloudy. The answer is ==yes==.

    Q: Consider the following premises: If it is overcast, then Robert is drinking water. If Patricia were a police officer, then it is overcast. Can we infer the following from them? Answer yes or no: If Patricia were a police officer, then Robert is drinking water.
    A: Let A be the claim that "it is overcast", B be the claim that "Robert is drinking water", C be the claim that "Patricia is a police officer", then the premises are "if A then B, if C then A". By the transitivity rule, we can infer "if C then B". Therefore, we can infer that If Patricia were a police officer, then Robert is drinking water. The answer is ==yes==.

    Q: Consider the following premises: Elisabeth is cooking, and Robert is a police officer and it is raining. Can we infer the following from them? Answer yes or no: The claim that Elisabeth is cooking and Robert is a police officer and the claim that it is raining cannot both be true.
    A: Let A be the claim that "Elisabeth is cooking", B be the claim that "Robert is a police officer", C be the claim that "it is raining", then the premises are "A and (B and C)". We cannot infer "not ((A and B) and C)" from the premises, because that contradicts the premises. The answer is ==no==.

    Q: Consider the following premises: It is overcast or it is cloudy. Can we infer the following from them? Answer yes or no: Neither it is cloudy nor it is overcast.
    A: Let A be the claim that "it is overcast", B be the claim that "it is cloudy", then the premises are "A or B". We cannot infer "not (B or A)" from the premises, because that contradicts the premises. The answer is ==no==.
    
    Q: """,

"gpt4": """Q: Consider the following premises: It is not cloudy or Joseph is rich. It is cloudy or Jennifer is playing a game. Can we infer the following from them? Answer yes or no: Jennifer is playing a game or Joseph is rich.
    A: Let A be the claim that "it is cloudy", B be the claim that "Joseph is rich", C be the claim that "Jennifer is playing a game", then the premises are "not A or B, A or C". By the resolution rule, we can infer "B or C". Therefore, we can infer that Jennifer is playing a game or Joseph is rich. The answer is ==yes==.

    Q: Consider the following premises: If Robert plays a game, then Jessica is a politician. Jessica is not a politician. Can we infer the following from them? Answer yes or no: Robert is not playing a game.
    A: Let A be the claim that "Robert plays a game", B be the claim that "Jessica is a politician", then the premises are "if A then B, not B". By the modus tollens rule, we can infer "not A". Therefore, we can infer that Robert is not playing a game. The answer is ==yes==.

    Q: Consider the following premises: The claim that it is snowing and the claim that John is an astronaut cannot both be true. Can we infer the following from them? Answer yes or no: Neither it is not snowing nor John is not an astronaut.
    A: Let A be the claim that "it is snowing", B be the claim that "John is an astronaut", then the premises are "not (A and B)". We cannot infer "not (not A and not B)" from the premises, because that contradicts the premises. The answer is ==no==.

    Q: Consider the following premises: It is raining and it is windy. Can we infer the following from them? Answer yes or no: Robert is not sad.
    A: Let A be the claim that "it is raining", B be the claim that "it is windy", C be the claim that "Robert is sad", then the premises are "A and B". We cannot infer "not C" from the premises, because the conclusion is not related to the premises. The answer is ==no==.
    
    Q: """,


"gpt4o": """Q: Consider the following premises: Thomas is not climbing a mountain or David is a scientist. Thomas is climbing a mountain or Joseph is happy. Can we infer the following from them? Answer yes or no: Joseph is happy or David is a scientist.
    A: Let A be the claim that "Thomas is climbing a mountain", B be the claim that "David is a scientist", C be the claim that "Joseph is happy", then the premises are "not A or B, A or C". By the resolution rule, we can infer "C or B". Therefore, we can infer that Joseph is happy or David is a scientist. The answer is ==yes==.

    Q: Consider the following premises: If Richard were a nurse, then Karen is an office clerk. Richard is a nurse or John is a sailor. If John were a sailor, then Karen is an office clerk. Can we infer the following from them? Answer yes or no: Karen is an office clerk.
    A: Let A be the claim that "Richard is a nurse", B be the claim that "Karen is an office clerk", C be the claim that "John is a sailor", then the premises are "if A then B, A or C, if C then B". By the disjunction elimination rule, we can infer B. Therefore, we can infer that Karen is an office clerk. The answer is ==yes==.

    Q: Consider the following premises: If Barbara were a cashier, then John is an office clerk. If John were an office clerk, then Barbara is a cashier. Can we infer the following from them? Answer yes or no: Barbara is a cashier.
    A: Let A be the claim that "Barbara is a cashier", B be the claim that "John is an office clerk", then the premises are "if A then B, if B then A". We cannot infer A from the premises, because the conclusion is not related to the premises. The answer is ==no==.

    Q: Consider the following premises: If it is cold, then it is cloudy. Can we infer the following from them? Answer yes or no: Neither it is not cold nor it is cloudy.
    A: Let A be the claim that "it is cold", B be the claim that "it is cloudy", then the premises are "if A then B". We cannot infer "not (not A or B)" from the premises, because that contradicts the premises. The answer is ==no==.
    
    Q: """,



"chatgpt": """Q: Consider the following premises: If Mary takes a plane, then it is sunny. Can we infer the following from them? Answer yes or no: Mary is not taking a plane or it is sunny.
    A: Let A be the claim that "Mary takes a plane", B be the claim that "it is sunny", then the premises are "if A then B". By the conditional laws rule, we can infer "not A or B". Therefore, we can infer that Mary is not taking a plane or it is sunny. The answer is ==yes==.

    Q: Consider the following premises: If William were an electrician, then it is sunny. If it is sunny, then Patricia is a janitor. Can we infer the following from them? Answer yes or no: If William were an electrician, then Patricia is a janitor.
    A: Let A be the claim that "William is an electrician", B be the claim that "it is sunny", C be the claim that "Patricia is a janitor", then the premises are "if A then B, if B then C". By the transitivity rule, we can infer "if A then C". Therefore, we can infer that If William were an electrician, then Patricia is a janitor. The answer is ==yes==.

    Q: Consider the following premises: Neither Patricia is sleeping nor David is climbing a mountain. Can we infer the following from them? Answer yes or no: The claim that Patricia is not sleeping and the claim that David is not climbing a mountain cannot both be true.
    A: Let A be the claim that "Patricia is sleeping", B be the claim that "David is climbing a mountain", then the premises are "not A and not B". We cannot infer "not (not A and not B)" from the premises, because that contradicts the premises. The answer is ==no==.

    Q: Consider the following premises: If Susan were a musician, then David is drinking water. If David drinks water, then Susan is a musician. Can we infer the following from them? Answer yes or no: David is drinking water.
    A: Let A be the claim that "Susan is a musician", B be the claim that "David is drinking water", then the premises are "if A then B, if B then A". We cannot infer B from the premises, because the conclusion is not related to the premises. The answer is ==no==.

    Q: """,


"gemini": """Q: Consider the following premises: James is not driving a car or William is drinking water. Can we infer the following from them? Answer yes or no: If James drives a car, then William is drinking water.
    A: Let A be the claim that "James drives a car", B be the claim that "William is drinking water", then the premises are "not A or B". By the conditional laws rule, we can infer "if A then B". Therefore, we can infer that If James drives a car, then William is drinking water. The answer is ==yes==.

    Q: Consider the following premises: Jessica is playing tennis or Robert is a taxi driver. Jessica is not playing tennis. Can we infer the following from them? Answer yes or no: Robert is a taxi driver.
    A: Let A be the claim that "Jessica is playing tennis", B be the claim that "Robert is a taxi driver", then the premises are "A or B, not A". By the disjunctive syllogism rule, we can infer B. Therefore, we can infer that Robert is a taxi driver. The answer is ==yes==.

    Q: Consider the following premises: If Linda drives a car, then it is overcast. Can we infer the following from them? Answer yes or no: Neither Linda is not driving a car nor it is overcast.
    A: Let A be the claim that "Linda drives a car", B be the claim that "it is overcast", then the premises are "if A then B". We cannot infer "not (not A or B)" from the premises, because that contradicts the premises. The answer is ==no==.

    Q: Consider the following premises: Sarah is writing a letter or it is late. Sarah is not writing a letter. Can we infer the following from them? Answer yes or no: It is not late.
    A: Let A be the claim that "Sarah is writing a letter", B be the claim that "it is late", then the premises are "A or B, not A". We cannot infer "not B" from the premises, because that contradicts the premises. The answer is ==no==.

    Q: """,
}


prompt_dict = {
    "zero_shot": ZERO_SHOT,
    "zero_shot_cot": ZERO_SHOT_COT,
    "random_icl": RANDOM_ICL_PROMPT,
    "weak": WEAK_PROMPT,
}


