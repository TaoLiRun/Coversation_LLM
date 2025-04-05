import random
from UoT.uot import UoTNode, expand, select, renew_node_to_root
from UoT.chat_utils import ques_and_cls_given_items, cls_given_repo, initialize_open_set, renew_open_set
from expert import Expert
from UoT.tasks.prompts import medical_diagnosis

class UoTExpert(Expert):
    """
    Expert system that uses the UoT approach for medical diagnosis.
    This expert implements a tree-based decision-making process where each node represents
    a question and its possible answers, helping to narrow down the diagnosis through
    systematic questioning.
    """
    def __init__(self, args, inquiry, options):
        super().__init__(args, inquiry, options)
        # Initialize the root node with all possible diagnoses
        self.root = UoTNode(
            question=inquiry,
            answer=True,
            items=list(options.values()),
            model=args.expert_model
        )
        self.current_node = self.root
        self.history = []
        self.prompts = medical_diagnosis
        self.n_extend_layers = 3
        self.n_potential_actions = args.n_potential_actions if hasattr(args, 'n_potential_actions') else 3
        self.n_pruned_nodes = args.n_pruned_nodes if hasattr(args, 'n_pruned_nodes') else 0

    def respond(self, patient_state):
        """
        Generate a response using the UoT approach. This can either be a question
        to narrow down the diagnosis or a final diagnosis choice.
        """
        # Update the current node based on patient's response if available
        if 'last_answer' in patient_state:
            self.current_node = self.current_node.ans2node(patient_state['last_answer'])
            self.history.append({
                'question': self.current_node.question,
                'answer': patient_state['last_answer']
            })

        # If we have a terminal node (2 or fewer possible diagnoses), make a choice
        if self.current_node.is_terminal:
            # Choose the most likely diagnosis from remaining options
            choice = random.choice(self.current_node.items)
            return {
                "type": "choice",
                "letter_choice": choice,
                "confidence": 1.0,  # UoT doesn't provide confidence scores
                "usage": {"input_tokens": 0, "output_tokens": 0}  # Placeholder for token usage
            }

        # Generate new questions using UoT's question generation
        if not self.current_node.children:
            questions = ques_and_cls_given_items(
                task=self,  # Pass self as task since we implement the required interface
                items=self.current_node.items,
                n=self.n_potential_actions,
                asked_ques=[msg['question'] for msg in self.history]
            )
            if questions:
                # Create child nodes for each question
                for q in questions:
                    yes_node = UoTNode(q["question"], True, q["items_yes"], parent=self.current_node, model=self.args.expert_model)
                    no_node = UoTNode(q["question"], False, q["items_no"], parent=self.current_node, model=self.args.expert_model)
                    self.current_node.children.append((yes_node, no_node))

        # Select the best question using UoT's selection mechanism
        if self.current_node.children:
            best_node = select(self, self.current_node)
            if best_node:
                return {
                    "type": "question",
                    "question": best_node.question,
                    "letter_choice": None,
                    "confidence": 1.0,  # UoT doesn't provide confidence scores
                    "usage": {"input_tokens": 0, "output_tokens": 0}  # Placeholder for token usage
                }

        # Fallback to basic question generation if UoT methods fail
        return super().ask_question(patient_state, self.history)

    def ask_question(self, patient_state, prev_messages):
        """
        Generate a question using the UoT approach. This method is called when
        the expert needs to ask a question to narrow down the diagnosis.
        """
        # Use UoT's question generation if possible
        if self.current_node and not self.current_node.is_terminal:
            questions = ques_and_cls_given_items(
                task=self,
                items=self.current_node.items,
                n=1,
                asked_ques=[msg['question'] for msg in prev_messages]
            )
            if questions:
                return {
                    "type": "question",
                    "atomic_question": questions[0]["question"],
                    "letter_choice": None,
                    "confidence": 1.0,
                    "usage": {"input_tokens": 0, "output_tokens": 0}
                }

        # Fallback to parent class's question generation
        return super().ask_question(patient_state, prev_messages) 
