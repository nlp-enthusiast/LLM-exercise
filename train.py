import argparse
import math
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from model import ModelMode
from data import DateMode, DataCollator
from utils import *
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--global_rank", type=int, default=-1)
    parser.add_argument("--model_name_or_path", type=str, help="", required=True)
    parser.add_argument("--model_mode", type=str, choices=["glm2"], help="", default="glm2")
    # data
    parser.add_argument("--train_path", type=str, default="./corpus/sunwukong/train.txt", help="")
    parser.add_argument("--is_skip", action="store_true", help="如果长度过长跳过该样本")
    parser.add_argument("--max_len", type=int, default=1024, help="")
    parser.add_argument("--max_src_len", type=str, default=256, help="")
    # Train
    parser.add_argument("--train_type", type=str, default="lora", help="")
    parser.add_argument("--epochs", type=int, default=3, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="")
    parser.add_argument("--lr", type=float, default=5e-5, help="")
    parser.add_argument("--save_model_step", type=int, default=1000, help="")
    parser.add_argument("--show_loss_step", type=int, default=10, help="")
    parser.add_argument("--output_dir", type=str, default="./save", help="")
    parser.add_argument("--rank_id", type=str, default="1,2", help="指定gpu编号")
    parser.add_argument("--rank_num", type=int, default=1)
    # LoRA
    parser.add_argument("--lora_dim", type=int, default=8, help="")
    parser.add_argument("--lora_alpha", type=int, default=30, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="")
    parser.add_argument("--lora_module_name", type=str, default="query_key_value", help="")

    return parser.parse_args()


def main():
    set_seed()
    args = parse_args()

    # choose device
    if torch.cuda.is_available():
        if args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.distributed.init_process_group(backend="nccl")
            device_list = [int(item) for item in args.rank_id.split(",")]
            args.local_rank = device_list[args.local_rank]
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            args.global_rank = torch.distributed.get_rank()
            args.rank_num=len(device_list)
    else:
        device = torch.device("cpu")

    # load tokenizer
    tokenizer = ModelMode[args.model_mode]["tokenizer"].from_pretrained(args.model_name_or_path)

    # load model
    model = ModelMode[args.model_mode]["model"].from_pretrained(args.model_name_or_path)
    if args.train_type == "lora":
        lora_module_name = args.lora_module_name.split(",")
        config = LoraConfig(r=args.lora_dim,
                            lora_alpha=args.lora_alpha,
                            target_modules=lora_module_name,
                            lora_dropout=args.lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            inference_mode=False,
                            )
        model = get_peft_model(model, config)
    elif args.train_type == "freeze":
        freeze_module_name = args.freeze_module_name.split(",")
        for name, param in model.named_parameters():
            if not any(nd in name for nd in freeze_module_name):
                param.requires_grad = False
    elif args.train_type == "ptuning":
        config = ModelMode[args.model_mode]["config"].from_pretrained(args.model_name_or_path)
        config.pre_seq_len = args.pre_seq_len
        config.prefix_projection = args.prefix_projection
        model = ModelMode[args.model_mode]["model"].from_pretrained(args.model_name_or_path, config=config)
        for name, param in model.named_parameters():
            if not any(nd in name for nd in ["prefix_encoder"]):
                param.requires_grad = False
    elif args.train_type == "all":
        model = ModelMode[args.mode]["model"].from_pretrained(args.model_name_or_path)
    else:
        raise Exception("train_type无效")

    if args.global_rank <= 0:
        tb_write = SummaryWriter()

    # load data
    train_dataset = DateMode[args.model_mode](args.train_path, tokenizer, args.max_len, args.max_src_len, args.is_skip)
    log_rank_0(f'''input id example:\n{train_dataset[0]["input_ids"]}\n{train_dataset[0]["labels"]}''',
               args.global_rank)
    log_rank_0(
        f'''input example:\n{tokenizer.decode(train_dataset[0]["input_ids"])}\n{tokenizer.decode(train_dataset[0]["labels"])}''',
        args.global_rank)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)

    data_collator = DataCollator(tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    num_training_steps = args.epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)

    log_rank_0("epochs = {}".format(args.epochs), args.global_rank)
    log_rank_0("train samples num = {}".format(len(train_dataset)), args.global_rank)
    log_rank_0("per_device_train_batch_size = {}".format(args.per_device_train_batch_size), args.global_rank)
    log_rank_0("train_batch_size = {}".format(args.per_device_train_batch_size*args.rank_num), args.global_rank)
    log_rank_0("gradient_accumulation_steps = {}".format(args.gradient_accumulation_steps), args.global_rank)
    log_rank_0("num_training_steps = {}".format(num_training_steps), args.global_rank)
    log_rank_0("num_warmup_steps = {}".format(num_warmup_steps), args.global_rank)

    # print model and parameters
    log_rank_0(model, args.global_rank)
    print_trainable_parameters(model)

    # # gradient_checkpointing
    # if args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()
    #     if hasattr(model, "enable_input_require_grads"):
    #         model.enable_input_require_grads()
    #     else:
    #         def make_inputs_require_grad(module, input, output):
    #             output.requires_grad_(True)
    #
    #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    optimizer = Adam(model.parameters(),lr=args.lr)
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    # train
    model.to(device)
    for epoch in range(args.epochs):
        log_rank_0("Beginning of Epoch {}/{}, Total Batches {}".format(epoch + 1, args.epochs,
                                                                               len(train_dataloader)), args.global_rank)
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch"):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            tr_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                # write loss
                if global_step % args.show_loss_step == 0:
                    log_rank_0("Epoch: {}, step: {}, global_step:{}, loss: {}".format(epoch, step + 1, global_step,
                                                                                        (tr_loss - logging_loss) /
                                                                                        (
                                                                                                args.show_loss_step * args.gradient_accumulation_steps)
                                                                                        ),
                                 args.global_rank)
                    log_rank_0("step: {} global_step: {}".format(step + 1, global_step), args.global_rank)
                    if args.global_rank <= 0:
                        tb_write.add_scalar("train_loss", (tr_loss - logging_loss) /
                                            (args.show_loss_step * args.gradient_accumulation_steps), global_step)
                        logging_loss = tr_loss
                # save model
                if args.save_model_step is not None and global_step % args.save_model_step == 0:
                    if args.global_rank <= 0:
                        save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")
                    model.train()

        if args.global_rank <= 0:
            save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")
if __name__ == '__main__':
    main()
