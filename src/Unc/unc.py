import numpy as np
import random
from typing import List, Dict, Tuple
from .models import get_response_method
import copy
import re

import importlib
task_parameter_mapping = {
    "20q": "unc_twenty_question",
    "md": "unc_medical_diagnosis",
    "tb": "unc_troubleshooting",
}
def import_prompts_by_task(task_name):
    parameter = task_parameter_mapping.get(task_name)
    module_name = f"uot.tasks.prompts.{parameter}"
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        raise ImportError(f"Failed to import module: {module_name}")
    

class UncNode:
    def __init__(self, task, threshold_p: float, threshold_u: float, lambda_ = -0.1):
        """
        Uncertainty node: Initialize the UncNode with candidates, thresholds, and hyperparameters.

        :param candidates: List of possible answers (e.g., ["fever", "headache"]).
        :param threshold_p: Probability threshold (\hat{p}) to decide when to give an answer.
        :param threshold_u: Uncertainty threshold (\hat{u}) to decide when to give an answer.
        :param lambda_: Hyperparameter (\lambda) to balance probability and uncertainty.
        """
        self.task = task
        self.response = get_response_method(task.guesser_model)
        self.items = list(map(str.lower, task.set))
        self.possible_items = []
        self.repo = None
        self.probabilities = {}  # Uniform initial probabilities
        self.uncertainties = {}  # Initial uncertainties
        self.history_e = []  # List of historical interactions from the examiner's perspective
        self.history_g = []  # List of historical interactions from the guesser's perspective
        self.asked_aspect_map = {candidate: [] for candidate in self.items}  # Track asked aspects for each candidate
        self.threshold_p = threshold_p
        self.threshold_u = threshold_u
        self.lambda_ = lambda_

        self.prompts = import_prompts_by_task("md") 
        # self.prompts.Initial_prompt, self.prompts.Rejudge_prompt
        # self.prompts.history_prompt, self.prompts.new_question_prompt
        # self.prompts.targeting_prompt_set, 

    def extract_probs(self, rsp: str, sampled_probabilities : Dict) -> Dict[str, float]:
        """
        Extract probabilities for each disease from the response.

        :param rsp: The response from the language model.
        :return: A dictionary mapping each disease to its probability.
        """

        rsp_lower = rsp.lower()
        # Create a regex pattern to match any term followed by '=xxx'
        pattern = r"\*\s*([\w\s,\s(\s)]+)\s*=\s*(\d*\.?\d+)"  # Match '* term = xxx' where term can have multiple words and commas
        # Find all matches in the response
        matches = re.findall(pattern, rsp_lower)
        
        # Convert matches to a dictionary
        for match in matches:
            term, prob = match
            term = term.strip()
            if term not in sampled_probabilities:
                sampled_probabilities[term] = []
            sampled_probabilities[term].append(float(prob))
        
        return sampled_probabilities
        

    def summarize_history(self) -> str:
        """
        Summarize a history into a string by asking the guesser to summarize it.

        :param history: List of historical interactions (either history_e or history_g).
        :return: A summarized string of the history.
        """
        # Extract content from history_e and history_g
        history_e_content = [h['content'] for h in self.history_e]
        history_g_content = [h['content'] for h in self.history_g]
        
        # Format the prompt with the extracted content
        message = [
            {"role": "user", "content": self.prompts.history_prompt.format(
                history_e_content=". ".join(history_e_content),
                history_g_content=". ".join(history_g_content)
            )}
        ]

        # Get the summarized response from the guesser
        summarized_response = self.response(message, model=self.task.guesser_model, max_tokens=400)

        return summarized_response

    def handle_self_repo(self, repo):

        repo = self.task.prompts.self_repo_prompt.format(repo=repo)
        self.repo = repo

        input_to_filter = repo

        # Get multiple rounds of filtering and probability estimation
        num_filter_rounds = 3
        all_possible_items = set()
        sampled_probabilities = {}  # Initialize empty dictionary

        # Multiple rounds of filtering
        for _ in range(num_filter_rounds):
            filter_message = [{"role": "user", "content": self.prompts.Initial_prompt.format(
                    item_list_str=', '.join(self.items),
                    repo=input_to_filter
                    )}]
            filter_response = self.response(filter_message, model=self.task.guesser_model, max_tokens=200)
            filter_response = filter_response.lower()
            # Extract the names of possible diseases from the filter response
            all_possible_items = set()
            try:
                diseases_part = filter_response.split(":", 1)[1]
                diseases_list = diseases_part.split("\n\n", 1)[1].strip().split("\n")[:5]
                for disease in diseases_list:
                    disease_name = disease.split(" ", 1)[1].strip()
                    if disease_name in self.items:
                        all_possible_items.add(disease_name)
            except Exception as e:
                diseases_part = filter_response.split(":", 1)[1]
                diseases_list = diseases_part.split("\n\n", 1)[1].strip().split("\n")[0]
                disease_name_str = diseases_list.split(",",)
                for disease in disease_name_str:
                    disease_name = disease.replace(",", "").strip()
                    if disease_name in self.items:
                        all_possible_items.add(disease_name)
            
            # Get probability estimates for all possible items
            message = [{"role": "user", "content": self.prompts.Initial_prob_prompt.format(
                repo=repo,
                item_list_str=', '.join(list(all_possible_items))
                )}]
            response = self.response(message, model=self.task.guesser_model, temperature=1, max_tokens= min(1000, len(self.items) * (self.task.expected_target_tokens + 50)))
            
            sampled_probabilities = self.extract_probs(response,sampled_probabilities)
            
                        
            all_possible_items.update(sampled_probabilities.keys())
        if sampled_probabilities == {} :
            breakpoint()
        
        self.possible_items = list(all_possible_items)

        # Calculate mean probability and uncertainty for each disease
        self.probabilities = {}  # Initialize empty dictionary
        self.uncertainties = {}  # Initialize empty dictionary
        for item in sampled_probabilities:
            # Filter out None values and zeros before calculating statistics
            valid_probs = [p for p in sampled_probabilities[item] if p is not None and p > 0]
            if valid_probs:
                mean_p = np.mean(valid_probs)
                std_u = np.std(valid_probs, ddof=1) if len(valid_probs) > 1 else 0.0
                self.probabilities[item] = mean_p
                self.uncertainties[item] = std_u

    def update_probabilities_and_uncertainties(self, num_samples: int = 3):
        """
        Update probabilities and uncertainties based on historical interactions.

        :param history_e: List of historical interactions from the examiner's perspective.
        :param history_g: List of historical interactions from the guesser's perspective.
        :param num_samples: Number of times to sample probabilities for each candidate.
        """
        # Combine history_e and history_g in chronological order
        history_str = ""
        i = 0
        while i < len(self.history_e) and i < len(self.history_g):
            history_str += f"Question: {self.history_e[i]['content']}\n"
            history_str += f"Answer: {self.history_g[i]['content']}\n"
            i += 1

        # Format the rejudge prompt
        # Convert probabilities dictionary to a formatted string
        #prob_list_str = ", ".join([f"{k}={v:.3f}" for k, v in self.probabilities.items() if v != 0.0])
        #unc_list_str = ", ".join([f"{k}={v:.3f}" for k, v in self.uncertainties.items() if v != 0.0])

        refilter_message = [
            {"role": "user", "content": self.prompts.Rejudge_prompt.format(
            repo=self.repo,
            history_str=history_str,
            item_list_str = ', '.join(self.items)
            )}]
        refilter_response = self.response(refilter_message, model=self.task.guesser_model, max_tokens=200)
        refilter_response = refilter_response.lower()
        # Extract the names of possible diseases from the filter response
        all_possible_items = set()
        try:
            diseases_part = refilter_response.split(":", 1)[1]
            diseases_list = diseases_part.split("\n\n", 1)[1].strip().split("\n")[:5]
            for disease in diseases_list:
                disease_name = disease.split(" ", 1)[1].strip()
                if disease_name in self.items:
                    all_possible_items.add(disease_name)
        except Exception as e:
            diseases_part = refilter_response.split(":", 1)[1]
            diseases_list = diseases_part.split("\n\n", 1)[1].strip().split("\n")[0]
            disease_name_str = diseases_list.split(",",)
            for disease in disease_name_str:
                disease_name = disease.replace(",", "").strip()
                if disease_name in self.items:
                    all_possible_items.add(disease_name)
        
        # Get probability estimates for all possible items
        message = [{"role": "user", "content": self.prompts.Initial_prob_prompt.format(
            repo=self.repo,
            item_list_str=', '.join(list(all_possible_items))
            )}]
        response = self.response(message, model=self.task.guesser_model, temperature=1, max_tokens= min(1000, len(self.items) * (self.task.expected_target_tokens + 50)))
        
        sampled_probabilities = {}
        sampled_probabilities = self.extract_probs(response,sampled_probabilities)
        
        all_possible_items.update(sampled_probabilities.keys())
        self.possible_items = list(all_possible_items)

        # Calculate mean probability and uncertainty for each disease
        self.probabilities = {}  # Initialize empty dictionary
        self.uncertainties = {}  # Initialize empty dictionary
        for item in sampled_probabilities:
            # Filter out None values and zeros before calculating statistics
            valid_probs = [p for p in sampled_probabilities[item] if p is not None and p > 0]
            if valid_probs:
                mean_p = np.mean(valid_probs)
                std_u = np.std(valid_probs, ddof=1) if len(valid_probs) > 1 else 0.0
                self.probabilities[item] = mean_p
                self.uncertainties[item] = std_u

    def ask_new_aspect(self, candidate: str, asked_ques: List[str] = None) -> Tuple[str, str]:
        """
        Ask for a new aspect to explore for a given candidate.

        :param candidate: The candidate to explore.
        :param asked_ques: List of already asked questions.
        :return: A tuple of (new_question, new_aspect).
        """
        # Format the prompt with the candidate and asked questions
        message = [
            {"role": "user", "content": self.prompts.new_question_prompt.format(
                candidate=candidate,
                asked_ques=", ".join(asked_ques) if asked_ques else "None"
            )}
        ]

        # Get the response from the language model
        rsp = self.response(message, temperature=0, max_tokens=200)

        # Extract the new question and new aspect from the response
        new_question = rsp.split("New Question: ", 1)[1].split("\n", 1)[0].strip()
        new_aspect = rsp.split("New Aspect: ", 1)[1].strip()
        try:
            new_aspect = new_aspect.split("\n\n", 1)[0].strip()
        except:
            pass

        # Add the new aspect to the asked_aspect_map
        if candidate not in self.asked_aspect_map:
            self.asked_aspect_map[candidate] = [new_aspect]
        else:
            self.asked_aspect_map[candidate].append(new_aspect)

        return new_question, new_aspect

    def generate_query(self, candidate: str) -> str:
        """
        Generate a new query based on the selected candidate and its unasked aspects.

        :param candidate: The candidate to generate a query for.
        :return: A new query (question).
        """
        # Get the list of already asked aspects for the candidate
        try:
            asked_ques = self.asked_aspect_map[candidate]
        except KeyError:
            asked_ques = []

        # Generate a new question and aspect using the ask_new_aspect logic
        new_question, _ = self.ask_new_aspect(candidate, asked_ques)

        return new_question

    def select_answer(self, ques_id: int) -> Tuple[str, bool]:
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

        # Check if we should give a final answer
        if max_prob >= self.threshold_p and self.uncertainties[max_candidate] <= self.threshold_u:
            return max_candidate, True

        # If not giving a final answer, generate a new query
        new_query = self.generate_query(max_candidate)
        return new_query, False

    def add_interaction(self, examiner_response: Dict[str, str], guesser_response: Dict[str, str]):
        """
        Add a new interaction to the history.

        :param examiner_response: The examiner's response.
        :param guesser_response: The guesser's response.
        """
        self.history_e.append(examiner_response)
        self.history_g.append(guesser_response)

    def __repr__(self):
        return f"UncNode(threshold_p={self.threshold_p}, threshold_u={self.threshold_u}, lambda_={self.lambda_})"

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