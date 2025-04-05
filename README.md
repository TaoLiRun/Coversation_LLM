**# Currently largely based on MediQ: Question-Asking LLMs for Adaptive and Reliable Clinical Reasoning**



- Current Results (based on craft data; SR: success rate)

  |             | Patient (Deepseek)               | Doctor (llama-2-70B)     | SR   | Log                       |
  | ----------- | -------------------------------- | ------------------------ | ---- | ------------------------- |
  | Replication | Randomly Reply                   | Fixed Abstain            | 0.45 | craft_random_fixed.log   |
  | Replication | Reply based on question and fact | Fixed Abstain            | 0.48 | craft_fact_fixed.log  |
  | Replication | Reply based on question and fact | Abstain based on history | 0.432 | craft_fact_binary.log |
  |New|Reply based on question and fact|Unc_Logprobs|0.32|             craft_fact_unc_logprobs.log              |
  |             |                                  |                          |      |                           |
  |             |                                  |                          |      |                           |
  |             |                                  |                          |      |                           |

  
