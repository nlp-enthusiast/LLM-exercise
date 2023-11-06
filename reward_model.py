from typing import List
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import nn
from torch.optim import Adam
from transformers import BertModel, BertTokenizer

def compute_rank_list_loss(rank_rewards_list: List[List[torch.tensor]], device='cpu') -> torch.Tensor:
    """
    通过给定的有序（从高到低）的ranklist的reward列表，计算rank loss。
    所有排序高的句子的得分减去排序低的句子的得分差的总和，并取负。

    Args:
        rank_rewards_list (torch.tensor): 有序（从高到低）排序句子的reward列表，e.g. ->
                                        [
                                            [torch.tensor([0.3588]), torch.tensor([0.2481]), ...],
                                            [torch.tensor([0.5343]), torch.tensor([0.2442]), ...],
                                            ...
                                        ]
        device (str): 使用设备

    Returns:
        loss (torch.tensor): tensor([0.4891], grad_fn=<DivBackward0>)
    """
    # if type(rank_rewards_list) != list:
    #     raise TypeError(f'@param rank_rewards expected "list", received {type(rank_rewards)}.')

    loss, add_count = torch.tensor([0]).to(device), 0
    for rank_rewards in rank_rewards_list:
        for i in range(len(rank_rewards) - 1):  # 遍历所有前项-后项的得分差
            for j in range(i + 1, len(rank_rewards)):
                diff = F.sigmoid(rank_rewards[i] - rank_rewards[j])  # sigmoid到0~1之间
                loss = loss + diff
                add_count += 1
    loss = loss / add_count
    return -loss


class RewardModel(Module):
    def __init__(self, encoder):
        super(RewardModel, self).__init__()
        self.encoder = encoder
        self.reward_layer = nn.Linear(768, 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        pooler_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )["pooler_output"]  # (batch, hidden_size)
        reward = self.reward_layer(pooler_output)  # (batch, 1)
        return reward


tokenizer = BertTokenizer.from_pretrained("/home/fg21/llm-master/bert-base-chinese")
encoder = BertModel.from_pretrained("/home/fg21/llm-master/bert-base-chinese")
model = RewardModel(encoder)
optimizer = Adam(model.parameters(), lr=1e-4)



if __name__ == '__main__':
    epochs = 10
    device = torch.device("cpu")
    samples = ["买过很多箱这个苹果了，一如既往的好，汁多味甜～", "名不副实。", "拿过来居然屏幕有划痕，顿时就不开心了", "什么手机啊！一台充电很慢，信号不好！退了！又买一台竟然是次品"
        , "一直用沙宣的洗发露！是正品！去屑止痒润发护发面面俱到！", "觉得比外买的稀，好似加了水的", "非常非常不满意，垃圾。", "什么垃圾衣服，买来一星期不到口袋全拖线，最差的一次购物"]
    test_samples = ['买过很多箱这个苹果了，一如既往的好，汁多味甜～', '什么手机啊！一台充电很慢，信号不好！退了！又买一台竟然是次品。。服了。。']
    inputs = tokenizer(samples, padding=True, return_tensors="pt")

    model.to(device)
    print("Start Train!")
    for epoch in range(epochs):
        for step in range(30):
            optimizer.zero_grad()
            outputs = model(inputs["input_ids"].to(device), inputs["token_type_ids"].to(device), inputs["attention_mask"].to(device))
            loss = compute_rank_list_loss([outputs],device=device)
            loss.backward()
            optimizer.step()
            if (step+1)%10==0:
                print(f"epoch:{epoch} step:{step+1} loss:{loss}")
        test_inputs = tokenizer(test_samples, padding=True, return_tensors="pt")
        outputs = model(inputs["input_ids"].to(device), inputs["token_type_ids"].to(device), inputs["attention_mask"].to(device))
        print("Start Eval!")
        for i in range(len(test_samples)):
            print(f"score:{outputs[i].item()} "+test_samples[i])