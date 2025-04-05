export DEEPSEEK_API_KEY=sk-c8a1802f6d0c4e5098a07959bf6d3a37
export LOCALHOST_API_ADDR=http://dgx-34:12890/v1

python ../src/mediQ_benchmark.py \
 --expert_model="llama" --expert_class BinaryExpert\
 --patient_model="deepseek" --patient_class FactSelectPatient\
 --data_dir ../data --dev_filename all_dev_good.jsonl \
 --output_filename ../logs/dev_fact_binary.jsonl \
 --max_questions 5 --log_filename ../logs/dev_fact_binary_result.log
