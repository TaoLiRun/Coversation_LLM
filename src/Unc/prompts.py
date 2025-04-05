question_word = "Doctor Question"
answer_word = "Patient Response"

update_template = """A patient comes into the clinic presenting with a symptom as described in the conversation log below:
PATIENT INFORMATION: {}
CONVERSATION LOG:
{}
QUESTION: {}
OPTIONS: {}
YOUR TASK: {}"""

update_task = "Given the information from above, respond with two letter choices and NOTHING ELSE.\nTWO LETTER CHOICES: "

meditron_system_msg = "You are a medical doctor trying to reason through a real-life clinical case. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, respond according to the task specified by the user. Base your response on the current and standard practices referenced in medical guidelines."

atomic_question_improved = "Based on the conversation log, you are considering {} as a potential diagnosis. However, there are missing features that prevent you from making this diagnosis with confidence. Consider which features specific to {} have not yet been asked about in the conversation log. Then, identify the most important missing feature that would help confirm or rule out {}. You can ask about any relevant information such as family history, tests and exams results, or treatments already done that are particularly important for diagnosing {}. Consider what questions doctors commonly ask when evaluating a patient for {}. Ask ONE SPECIFIC ATOMIC QUESTION to address this feature. The question should be bite-sized, and NOT ask for too much at once. Make sure to NOT repeat any questions from the above conversation log. Answer in the following format:\nATOMIC QUESTION: the atomic question and NOTHING ELSE.\nATOMIC QUESTION: "

