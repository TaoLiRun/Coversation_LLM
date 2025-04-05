from typing import Dict, List, Any, Optional
from .expert import Expert
from .Unc.unc import UncNode
from .Unc.prompts import get_prompts
from .Unc.chat_utils import format_question, process_response, generate_follow_up

class UncExpert(Expert):
    """Expert class that uses UNC (Uncertainty-based) approach for medical diagnosis."""
    
    def __init__(self, model_name: str, task_name: str = 'medical_diagnosis', uncertainty_threshold: float = 0.3):
        """
        Initialize the UNC expert.
        
        :param model_name: Name of the model to use (e.g., 'gpt-4')
        :param task_name: Name of the task (default: 'medical_diagnosis')
        :param uncertainty_threshold: Threshold for considering an outcome uncertain
        """
        super().__init__()
        self.model_name = model_name
        self.task_name = task_name
        self.uncertainty_threshold = uncertainty_threshold
        self.prompts = get_prompts(task_name)
        self.node = None
        self.history = []
    
    def initialize(self, patient_state: Dict[str, Any]) -> None:
        """
        Initialize the expert with patient state.
        
        :param patient_state: Dictionary containing patient information
        """
        self.node = UncNode(
            model_name=self.model_name,
            task_name=self.task_name,
            uncertainty_threshold=self.uncertainty_threshold
        )
        self.history = []
        
        # Add initial patient information to history
        if 'initial_info' in patient_state:
            self.history.append({
                'role': 'user',
                'content': f"Patient Information: {patient_state['initial_info']}"
            })
    
    def generate_response(self, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response based on the current patient state.
        
        :param patient_state: Dictionary containing patient information
        :return: Dictionary with response information
        """
        # Update node with latest patient state
        if 'last_answer' in patient_state:
            self.node.update_probabilities(patient_state['last_answer'])
            self.history.append({
                'role': 'assistant',
                'content': patient_state['last_answer']
            })
        
        # Generate next question
        question = self.node.generate_query(self.history)
        
        # Format question with history
        messages = format_question(question, self.history, self.node)
        
        # Get model response
        response = self.node.response_method(messages)
        
        # Process response
        processed = process_response(response, self.node)
        
        # Add to history
        self.history.append({
            'role': 'assistant',
            'content': response
        })
        
        return {
            'question': question,
            'probabilities': processed.get('probabilities', {}),
            'uncertainties': processed.get('uncertainties', {})
        }
    
    def generate_question(self, patient_state: Dict[str, Any]) -> str:
        """
        Generate a follow-up question based on uncertainties.
        
        :param patient_state: Dictionary containing patient information
        :return: Generated question
        """
        return generate_follow_up(self.history, self.node)
    
    def get_diagnosis(self) -> Dict[str, float]:
        """
        Get the current diagnosis probabilities.
        
        :return: Dictionary mapping diagnoses to probabilities
        """
        if self.node is None:
            return {}
        return self.node.probabilities

    # Required properties for UncNode to work
    @property
    def guesser_model(self):
        return "gpt-4"  # or your preferred model
        
    @property
    def max_turn(self):
        return 10  # or your preferred max turns
        
    @property
    def set(self):
        return self.node.items if self.node else [] 