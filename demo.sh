
save_dir=./save/sunwukong
mkdir -p $save_dir

python demo.py \
--ori_model_dir "./llm-models/chatglm3-6b" \
--model_dir "./save/sunwukong/epoch-2-step-396" \
--mode "glm3"



