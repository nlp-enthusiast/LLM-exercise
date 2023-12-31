
save_dir=./save/sunwukong
model_mode="glm3"
model_name_or_path="./llm-models/chatglm3-6b"

mkdir -p $save_dir
python -m torch.distributed.launch --nproc_per_node=2 train.py \
--rank_id 2,3 \
--model_mode $model_mode \
--train_path "./corpus/sunwukong/train.txt" \
--model_name_or_path $model_name_or_path  \
--train_type "lora" \
--epochs 3 \
--per_device_train_batch_size 4 \
--lr 5e-5 \
--save_model_step 1000 \
--show_loss_step 10 \
--save_dir $save_dir


