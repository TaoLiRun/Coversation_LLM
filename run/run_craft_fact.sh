export DEEPSEEK_API_KEY=sk-c8a1802f6d0c4e5098a07959bf6d3a37
export LOCALHOST_API_ADDR=http://dgx-34:12890/v1

python ../src/mediQ_benchmark.py \
 --expert_model="llama" --expert_class FixedExpert\
 --patient_model="deepseek" --patient_class FactSelectPatient\
 --data_dir ../data --dev_filename all_craft_md.jsonl \
 --output_filename ../logs/craft_fact_fixed.jsonl \
 --max_questions 5 --log_filename ../logs/craft_fact_fixed_result.log
