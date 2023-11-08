
save_dir=./save/sunwukong
mkdir -p $save_dir

python demo.py \
--ori_model_dir "./llm-models/chatglm3-6b" \
--model_dir "./save/sunwukong/epoch-3-step-594" \
--mode "glm3"



