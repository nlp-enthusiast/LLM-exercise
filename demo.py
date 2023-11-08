# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: merge_lora
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/8/6 16:13
"""
    文件说明：
            
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import torch
from model import ModelMode
import argparse
from peft import PeftModel


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_model_dir', default="/home/fg21/llm-master/llm-models/chatglm2-6b", type=str, help='')
    parser.add_argument('--model_dir', default="/home/fg21/llm-master/save/epoch-3-step-594", type=str, help='')
    parser.add_argument('--mode', default="glm2", type=str, help='')

    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    model = ModelMode[args.mode]["model"].from_pretrained(args.ori_model_dir)
    tokenizer = ModelMode[args.mode]["tokenizer"].from_pretrained(args.ori_model_dir)
    # lora_model = PeftModel.from_pretrained(base_model, args.model_dir, torch_dtype=torch.float16)
    # lora_model.to("cpu")
    # model = lora_model.merge_and_unload()
    model.cuda()
    # instruction = "你是孙悟空，你的特征描述是: 生性聪明、活泼、忠诚、嫉恶如仇，代表了机智、勇敢。常用口头禅是：“俺老孙”、“我齐天大圣”、“我美猴王”。 现在请你回答我的一些问题，以准确展现你的人格特征!你的说话风格要全面模仿被赋子的人格角色!请不要暴露你是人工智能模型或者语言模型，你要时刻记住你只被赋予的一个人格角色。说话不要啰嗦，也不要太过于正式或礼貌。"
    instruction=""
    past_key_values, history = None, []
    while True:
        query = input("\n用户：")
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, instruction+query, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):

                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")
