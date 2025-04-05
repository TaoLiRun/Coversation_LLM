import numpy as np
import random
from typing import List, Dict, Tuple
from Unc.models import get_response_method
import copy
import re
import time

import importlib 

class UncNode:
    def __init__(self, task_guesser_model, task_set, threshold_p: float, threshold_u: float, lambda_ = 0.1):
        """
        Uncertainty node: Initialize the UncNode with candidates, thresholds, and hyperparameters.

        :param candidates: List of possible answers (e.g., ["fever", "headache"]).
        :param threshold_p: Probability threshold (\hat{p}) to decide when to give an answer.
        :param threshold_u: Uncertainty threshold (\hat{u}) to decide when to give an answer.
        :param lambda_: Hyperparameter (\lambda) to balance probability and uncertainty.
        """
        self.response = get_response_method(task_guesser_model)
        self.items = task_set # a dictionary {'A': 'name', 'B': 'name'}
        self.options_text = f'A: {task_set["A"]}, B: {task_set["B"]}, C: {task_set["C"]}, D: {task_set["D"]}'
        self.target_tokens = list(task_set.keys()) + [' ' + key for key in task_set.keys()]
        
        self.possible_items = []
        self.probabilities = {}  # Uniform initial probabilities
        self.uncertainties = {}  # Initial uncertainties
        self.history_e = []  # List of historical interactions from the examiner's perspective
        self.history_g = []  # List of historical interactions from the guesser's perspective
        self.threshold_p = threshold_p
        self.threshold_u = threshold_u
        self.lambda_ = lambda_

        self.prompts = importlib.import_module("Unc.prompts")
        # self.prompts.Initial_prompt, self.prompts.Rejudge_prompt
        # self.prompts.history_prompt, self.prompts.new_question_prompt
        # self.prompts.targeting_prompt_set, 
    
    def sample_prob(self, prompt_message, num_samples: int = 3):
        '''update self.possible_items, self.probabilities, self.uncertainties based on the prompt_message'''
        # Get multiple rounds of filtering and probability estimation
        self.probabilities = {}  # Uniform initial probabilities
        self.uncertainties = {}  # Initial uncertainties

        all_possible_items = set()
        sampled_probabilities = {}

        # Multiple rounds of filtering
        for _ in range(num_samples):
            chosen_index_probs = self.response(prompt_message, temperature=1, max_tokens=200, target_tokens=self.target_tokens, only_probs=True)
            # chosen_index_probs is a dictionary with the indexes of the items as keys and their probabilities as values
            # Update all_possible_items and sampled_probabilities using chosen_index_probs
            for index, prob in chosen_index_probs.items():
                all_possible_items.add(index)
                if index not in sampled_probabilities:
                    sampled_probabilities[index] = []
                sampled_probabilities[index].append(prob)     
        self.possible_items = list(all_possible_items)

        # Calculate mean probability and uncertainty for each disease
        count = 0
        for item in sampled_probabilities:
            # Filter out None values and zeros before calculating statistics
            valid_probs = [p for p in sampled_probabilities[item] if p is not None and p > 0]
            if len(valid_probs) == 1:
                self.probabilities[item] = valid_probs[0]
                self.uncertainties[item] = 0.5
                continue
            if valid_probs:
                mean_p = np.mean(valid_probs)
                std_u = np.std(valid_probs, ddof=1) if len(valid_probs) > 1 else 0.0
                self.probabilities[item] = mean_p
                self.uncertainties[item] = std_u

        # Normalize probabilities to sum to 1
        total_prob = sum(self.probabilities.values())
        if total_prob > 0:
            for item in self.probabilities:
                self.probabilities[item] /= total_prob

        #print(self.probabilities)
        #print(self.uncertainties)
        #breakpoint()

    def update_probabilities_and_uncertainties(self, kwargs, num_samples: int = 3):
        """
        Update probabilities and uncertainties based on historical interactions.

        :param kwargs: Dictionary containing parameters from the expert system:
            - patient_state: Dictionary containing current patient information
            - inquiry: The current inquiry or question being addressed
            - options_dict: Dictionary of available options/choices
            - messages: List of previous conversation messages
            - independent_modules: Boolean indicating whether to use independent modules
            - model_name: Name of the model to use for generation
            - use_vllm: Boolean indicating whether to use vLLM
            - use_api: Boolean indicating whether to use API
            - temperature: Float controlling response randomness
            - max_tokens: Integer limiting response length
            - top_p: Float for nucleus sampling
            - top_logprobs: Integer for number of logprobs to return
            - api_account: API account information
        :param num_samples: Number of times to sample probabilities for each candidate.
        """
        # Combine history_e and history_g in chronological order
        patient_info = kwargs["patient_state"]["initial_info"]
        conv_log = '\n'.join([f"{self.prompts.question_word}: {qa['question']}\n{self.prompts.answer_word}: {qa['answer']}" for qa in kwargs["patient_state"]["interaction_history"]])
        prompt_update = self.prompts.update_template.format(patient_info, conv_log if conv_log != '' else 'None', kwargs["inquiry"], self.options_text, self.prompts.update_task)
    
        # Format the rejudge prompt
        # Convert probabilities dictionary to a formatted string
        #prob_list_str = ", ".join([f"{k}={v:.3f}" for k, v in self.probabilities.items() if v != 0.0])
        #unc_list_str = ", ".join([f"{k}={v:.3f}" for k, v in self.uncertainties.items() if v != 0.0])
        prompt_message = [
            {"role": "system", "content": self.prompts.meditron_system_msg},
            {"role": "user", "content": prompt_update}
        ]

        self.sample_prob(prompt_message)

    def generate_query(self, kwargs):
        """
        Generate a new query based on the selected candidate and its unasked aspects.

        :param candidate: The candidate to generate a query for.
        :return: A new query (question).
        """
        # Get the list of already asked aspects for the candidate
        query_candidate_index = max(self.probabilities.keys(), 
                            key=lambda x: self.probabilities[x] + self.lambda_ * self.uncertainties[x])
        query_candidate = self.items[query_candidate_index]
        patient_info = kwargs["patient_state"]["initial_info"]
        conv_log = '\n'.join([f"{self.prompts.question_word}: {qa['question']}\n{self.prompts.answer_word}: {qa['answer']}" for qa in kwargs["patient_state"]["interaction_history"]])
        task_prompt = self.prompts.atomic_question_improved.format(query_candidate,query_candidate,query_candidate,query_candidate,query_candidate)
        prompt_query = self.prompts.update_template.format(patient_info, conv_log if conv_log != '' else 'None', kwargs["inquiry"], self.options_text, task_prompt)
    
        prompt_message = [
            {"role": "system", "content": self.prompts.meditron_system_msg},
            {"role": "user", "content": prompt_query}
        ]

        # Get the response from the language model
        rsp = self.response(prompt_message, temperature=0, max_tokens=1000)

        atomic_question = parse_atomic_question(rsp)
        print("[ATOMIC QUESTION RETURN]:", atomic_question, "usage: ", " ")

        return {
            "atomic_question": atomic_question,
            "messages": rsp,
            "usage": "",
        }

    def select_answer(self) -> str:
        """
        Select an answer based on the current probabilities and uncertainties.

        :param ques_id: The ID of the question being answered.
        :return: A tuple of (answer, is_final).
        """
        # Find the candidate with the highest probability
        max_prob = 0
        max_candidate = None
        for candidate, prob in self.probabilities.items():
            if prob > max_prob:
                max_prob = prob
                max_candidate = candidate

        return max_candidate

    def __repr__(self):
        return (f"UncNode(candidates={self.items}, probabilities={self.probabilities}, "
                f"uncertainties={self.uncertainties})")
    
    def renew_node(self):
        """
        Renew the node by clearing its history and resetting its state.
        """
        self.history_e = []
        self.history_g = []
        self.asked_aspect_map = {candidate: [] for candidate in self.items}
        self.probabilities = {}
        self.uncertainties = {}
        self.possible_items = [] 

def parse_atomic_question(response_text):
    questions = []
    for line in response_text.split("\n"):
        if '?' in line:
            questions.append(line.split(":")[-1].strip())
        
    if len(questions) == 0:
        print("can't find question in answer: {}".format(response_text), type="error")
        return None
            
    atomic_question = questions[-1].replace("'", "").replace('"', "").strip()
    return atomic_question