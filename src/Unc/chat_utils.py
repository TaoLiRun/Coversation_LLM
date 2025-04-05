import json
from typing import List, Dict, Any, Optional
import numpy as np

def format_history(history: List[Dict[str, str]]) -> str:
    """
    Format the conversation history into a string.
    
    :param history: List of dictionaries containing 'role' and 'content'
    :return: Formatted history string
    """
    formatted = ""
    for msg in history:
        role = msg['role']
        content = msg['content']
        formatted += f"{role}: {content}\n"
    return formatted

def extract_probabilities(response: str) -> Dict[str, float]:
    """
    Extract probabilities from a model response.
    
    :param response: String response from the model
    :return: Dictionary mapping items to their probabilities
    """
    try:
        # Try to parse as JSON first
        probs = json.loads(response)
        if isinstance(probs, dict):
            return probs
    except:
        pass
    
    # If not JSON, try to extract probabilities from text
    probs = {}
    lines = response.strip().split('\n')
    for line in lines:
        try:
            item, prob = line.split(':')
            probs[item.strip()] = float(prob.strip())
        except:
            continue
    
    return probs

def format_question(question: str, history: List[Dict[str, str]], task: Any) -> List[Dict[str, str]]:
    """
    Format a question with history for the model.
    
    :param question: The question to ask
    :param history: List of previous interactions
    :param task: Task object containing prompts and configuration
    :return: List of message dictionaries
    """
    messages = []
    
    # Add system prompt
    messages.append({
        "role": "system",
        "content": task.prompts.system_prompt
    })
    
    # Add history
    if history:
        messages.append({
            "role": "user",
            "content": format_history(history)
        })
    
    # Add current question
    messages.append({
        "role": "user",
        "content": question
    })
    
    return messages

def process_response(response: str, task: Any) -> Dict[str, Any]:
    """
    Process a model response and extract relevant information.
    
    :param response: String response from the model
    :param task: Task object containing configuration
    :return: Dictionary with processed information
    """
    result = {}
    
    # Extract probabilities
    probs = extract_probabilities(response)
    result['probabilities'] = probs
    
    # Calculate uncertainty if needed
    if task.uncertainty_threshold > 0:
        uncertainties = {}
        for item, prob in probs.items():
            uncertainties[item] = 1 - abs(prob - 0.5) * 2
        result['uncertainties'] = uncertainties
    
    return result

def generate_follow_up(history: List[Dict[str, str]], task: Any) -> str:
    """
    Generate a follow-up question based on history and uncertainties.
    
    :param history: List of previous interactions
    :param task: Task object containing configuration
    :return: Generated follow-up question
    """
    # Get the last response
    last_response = history[-1]['content'] if history else ""
    
    # Process the response
    processed = process_response(last_response, task)
    
    # Find items with high uncertainty
    high_uncertainty = []
    if 'uncertainties' in processed:
        for item, unc in processed['uncertainties'].items():
            if unc > task.uncertainty_threshold:
                high_uncertainty.append(item)
    
    # Generate question based on uncertainties
    if high_uncertainty:
        return f"Let's clarify about {', '.join(high_uncertainty[:3])}. "
    else:
        return "What else would you like to know?" 