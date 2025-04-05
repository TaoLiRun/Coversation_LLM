generate_prompt = '''You are a doctor. Here are all the possible diseases that the patient may suffer from:
{items_str}

Please design a question to ask your patient with symptoms about disease and can only be answer by YES or NO. {asked} Then classify the possible disease above based on this question. If the answer is 'YES', put this disease into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO.
Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
You should think about best {n} questions to response. And your answer should be:
Question 1: ...?
YES: aaa, bbb, ... (disease names only)
Count of YES: ...
NO: ccc, ddd, ... (disease names only)
Count of NO: ...
'''

classify_prompt = '''Here are all diseases that the patient may suffer from:
{item_list_str}

{repo}
For each disease under this report, if the patient is possible to have, put this disease into 'YES: ...', otherwise to 'NO: ...'. And your answer should be like:
YES: aaa, bbb, ... (disease names only)
NO: ccc, ddd, ... (disease names only)'''

self_repo_prompt = '''The patient self-reports that: {repo}'''

format_generated_prompt = """Please format the following response according to the required format:
{rsp}

The format should be:
Question 1: [question]
YES: [comma-separated list]
NO: [comma-separated list]

Question 2: [question]
YES: [comma-separated list]
NO: [comma-separated list]

And so on..."""

init_open_set_prompt = """Based on the initial patient information:
{repo}

List {size} possible diseases that could explain these symptoms. Format your response as a Python list of strings."""

renew_open_set_prompt = """Based on our previous conversation and the current list of possible diseases {item_list},
suggest {size} additional diseases that could be relevant but haven't been considered yet.
Format your response as a Python list of strings.""" 