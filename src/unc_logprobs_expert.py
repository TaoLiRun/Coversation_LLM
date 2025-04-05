from typing import Dict, Any, Optional, List
from expert import Expert
from Unc.unc_logprobs import UncNode

class UNCLogProbsExpert(Expert):
    """Expert class that uses UNC LogProbs approach for medical diagnosis."""
    
    def __init__(self, args, inquiry, options, model_name: str = 'llama', task_name: str = 'medical_diagnosis', 
                 threshold_p: float = 0.7, threshold_u: float = 0.2, lambda_: float = 0.1):
        """
        Initialize the UNC LogProbs expert.
        
        :param model_name: Name of the model to use (e.g., 'gpt-4')
        :param task_name: Name of the task (default: 'medical_diagnosis')
        :param threshold_p: Probability threshold to decide when to give an answer
        :param threshold_u: Uncertainty threshold to decide when to give an answer
        :param lambda_: Hyperparameter to balance probability and uncertainty
        :param args: Additional arguments for the base Expert class
        :param inquiry: Initial inquiry for the base Expert class
        :param options: Available options for the base Expert class
        """
        # Initialize base class with required arguments
        super().__init__(
            args=args,
            inquiry=inquiry,
            options=options
        )
        self.task_name = task_name
        self.threshold_p = threshold_p
        self.threshold_u = threshold_u
        self.lambda_ = lambda_
        self.node = UncNode(
            task_guesser_model=model_name,  # The expert class will act as the task
            task_set=options,
            threshold_p=self.threshold_p,
            threshold_u=self.threshold_u,
            lambda_=self.lambda_
        )
        
    
    def respond(self, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle patient interaction and return appropriate response.
        
        :param patient_state: Dictionary containing patient information and last interaction
        :return: Dictionary with response information
        """
        # Handle initial state
        kwargs = self.get_abstain_kwargs(patient_state)

        # Update probabilities and uncertainties based on history
        self.node.update_probabilities_and_uncertainties(kwargs)
        print(self.node)

        max_candidate = self.node.select_answer()
        max_prob = self.node.probabilities[max_candidate]
        if (max_prob >= self.threshold_p and self.node.uncertainties[max_candidate] <= self.threshold_u) or len(patient_state['interaction_history']) == kwargs['max_depth']:
            return {
                "type": "choice",
                "letter_choice": max_candidate,
                "confidence": max_prob,
                "usage": ""
            }
        else:
            new_query = self.node.generate_query(kwargs)
            return {
                "type": "question",
                "question": new_query,
                "letter_choice": max_candidate,
                "confidence": max_prob,
                "usage": ""
            }
    