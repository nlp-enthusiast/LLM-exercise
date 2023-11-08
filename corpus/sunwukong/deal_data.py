import json

train_data = []
instruction = "你是孙悟空，你的特征描述是: 生性聪明、活泼、忠诚、嫉恶如仇，代表了机智、勇敢。常用口头禅是：“俺老孙”、“我齐天大圣”、“我美猴王”。 现在请你回答我的一些问题，以准确展现你的人格特征!你的说话风格要全面模仿被赋子的人格角色!请不要暴露你是人工智能模型或者语言模型，你要时刻记住你只被赋予的一个人格角色。说话不要啰嗦，也不要太过于正式或礼貌。"
with open("role-specific-孙悟空.txt") as f:
    data = f.readlines()
    for item in data:
        temp = {}
        item = json.loads(item)
        temp["instruction"] = instruction
        temp["input"] = item["instruction"]
        temp["output"] = item["answer"]
        train_data.append(temp)

with open("train-general.txt") as f:
    data = f.readlines()
    for item in data:
        item = json.loads(item)
        if item["role"]!="孙悟空":
            continue
        for ans in item["generated"]:
            temp = {}
            temp["instruction"] =instruction
            temp["input"] = item["question"]
            temp["output"] = ans
            train_data.append(temp)dsdads

with open("train-base.txt") as f:
    data = f.readlines()
    for item in data:
        item = json.loads(item)
        if item["role"]!="孙悟空":
            continue
        for ans in item["generated"]:
            temp = {}
            temp["instruction"] = instruction
            temp["input"] = item["question"]
            temp["output"] = ans
            train_data.append(temp)

with open("./train.txt","w",encoding="utf8") as f:
    for item in train_data:
        f.write(json.dumps(item,ensure_ascii=False)+"\n")