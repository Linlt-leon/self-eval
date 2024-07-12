python generate.py \
    --input_data './data/demo/individual_behavior_controls_vicuna_v15.json'\
    --model './model/vicuna-7b-v1.5'\
    --output_path './results/generation/vicuna15_harmful_adsuffix_output.json'\
    --template_name 'vicuna'\
    --device 'cuda:0'\
    --suffix_type 'adsuffix' \
    --data_type 'harmful'\
    --num_samples 100

# python generate.py \
#     --input_data './data/demo/individual_behavior_controls_llama2.json'\
#     --model '/hy-tmp/llama-2-7b-chat-hf'\
#     --output_path './results/generation/llama2_harmful_adsuffix_output.json'\
#     --template_name 'llama-2'\
#     --device 'cuda:0'\
#     --suffix_type 'adsuffix' \
#     --data_type 'harmful'\
#     --num_samples 100

# # suffix_type: 'nosuffix', 'adsuffix', 'insuffix'