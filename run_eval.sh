python self_eval.py \
    --input_data './results/generation/vicuna15_harmful_adsuffix_output.json'\
    --model './model/vicuna-7b-v1.5'\
    --mode 'input_only' \
    --output_path './results/evaluation/vicuna15_harmful_adsuffix_output_vicuna15_eval_input_only.csv'\
    --template_name 'vicuna' \
    --device 'cuda:0'\
    --num_samples 100

# # mode: 'input_only', 'output_only', 'input_and_output'
