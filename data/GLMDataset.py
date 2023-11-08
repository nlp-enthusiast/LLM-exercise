import json
from torch.utils.data import Dataset

# class GLMPromptDataSet(Dataset):
#     def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
#         self.all_data = []
#         skip_data_number = 0
#         with open(data_path, "r", encoding="utf-8") as fh:
#             data = fh.readlines()
#             print(len(data))
#             for i, line in enumerate(data):
#
#                 sample = json.loads(line.strip())
#                 skip_flag = False
#                 src_tokens = tokenizer.tokenize(
#                     "[Round {}]\n\n问：{}\n\n答：".format(1, sample["instruction"] + sample["input"]))
#
#                 if len(src_tokens) > max_src_len:
#                     # 当输入内容超长时，随向后截断，但保留“\n\n答：”内容
#                     src_tokens = src_tokens[:max_src_len - 4] + src_tokens[-4:]
#                     skip_flag = True
#
#                 max_tgt_len = max_len - 3 - len(src_tokens)
#                 tgt_tokens = tokenizer.tokenize(sample["output"])
#
#                 if len(tgt_tokens) > max_tgt_len:
#                     tgt_tokens = tgt_tokens[:max_tgt_len]
#                     skip_flag = True
#
#                 tokens = src_tokens + tgt_tokens + ["</s>"]
#                 assert len(tokens) <= max_len
#                 # ChatGLM2需要增加[gMASK]、sop两个标记
#                 input_ids = [tokenizer.get_command("[gMASK]"),
#                              tokenizer.get_command("sop")] + tokenizer.convert_tokens_to_ids(tokens)
#                 context_length = len(src_tokens) + 2
#                 labels = [-100] * context_length + input_ids[context_length:]
#
#                 assert len(input_ids) == len(labels)
#                 assert len(input_ids) <= max_len
#                 if is_skip and skip_flag:
#                     skip_data_number += 1
#                     continue
#                 self.all_data.append({"input_ids": input_ids, "labels": labels})
#         print("the number of skipping data is {}".format(skip_data_number))
#
#     def __len__(self):
#         return len(self.all_data)
#
#     def __getitem__(self, item):
#         instance = self.all_data[item]
#         return instance
class GLMPromptDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        self.all_data = []
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            data = fh.readlines()
            for i, line in enumerate(data):

                sample = json.loads(line.strip())
                skip_flag = False
                src_tokens = tokenizer.build_single_message("user", "", sample["instruction"] + sample["input"], )
                src_tokens.extend([tokenizer.get_command("<|assistant|>")])
                if len(src_tokens) > max_src_len:
                    # 当输入内容超长时，随向后截断，但保留“\n\n答：”内容
                    src_tokens = src_tokens[:max_src_len - 4] + src_tokens[-4:]
                    skip_flag = True

                max_tgt_len = max_len - 3 - len(src_tokens)
                tgt_tokens = tokenizer.tokenize("\n"+sample["output"])

                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                    skip_flag = True

                input_ids = src_tokens + tokenizer.convert_tokens_to_ids(tgt_tokens) + [
                    tokenizer.eos_token_id]
                assert len(input_ids) <= max_len
                input_ids = tokenizer.get_prefix_tokens()+input_ids
                context_length = len(src_tokens) + 2
                labels = [-100] * context_length + input_ids[context_length:]

                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids": input_ids, "labels": labels})
        print("the number of skipping data is {}".format(skip_data_number))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance
