import torch
from typing import List, Tuple
import os
import importlib
import sys
import numpy as np

def guide(model_output: List[torch.Tensor], guidance_func_file: str, input_to_state = None, score_to_output = None):
    """applies the guidance function to the input joint states and returns the guidance score"""
    #print input dimensions

    if input_to_state is not None:
        states, indices = input_to_state(model_output)
    else:
        states = model_output

    if guidance_func_file is None:
        print ("No guidance function provided")
        return model_output
    
    #check if file exists
    if not os.path.exists(guidance_func_file):
        print ("Guidance function file does not exist")
        return model_output


    # send to cpu
    states = states.cpu()

    #check if dir is in path
    if os.path.dirname(guidance_func_file) not in sys.path:
        sys.path.append(os.path.dirname(guidance_func_file))

    print(guidance_func_file.split('/')[-1].split('.')[0])
    #load guidance function
    guidance_func = importlib.import_module(guidance_func_file.split('/')[-1].split('.')[0]).guidance
    
    print("states: ",states.shape)

    flatten_states = states.view(-1, states.shape[-1])
    guidance_score = torch.tensor([guidance_func(s) for s in flatten_states.cpu().detach().numpy()]).to(states.device)
    # print("guidance_score: ",guidance_score)
    print("guidance_score.shape: ",guidance_score.shape)

    guidance_score = guidance_score.reshape(states.shape[:-1])
    #normalize guidance score
    if torch.max(guidance_score) != torch.min(guidance_score):
        guidance_score = (guidance_score - torch.min(guidance_score))/(torch.max(guidance_score) - torch.min(guidance_score))

    guidance_score = guidance_score.unsqueeze(-1)
    print("guidance_score.shape: ",guidance_score.shape)

    output = score_to_output(model_output, guidance_score, indices)
    #send back to device

    return output


class GuidanceLayer():
    def __init__(self, guidance_func_file: str,
                 input_to_state = None,
                 score_to_output = None,
                 robot_state_ranges: List[Tuple[float]] = None) -> None:
        """
        Args:
            guidance_func_file (str): The file containing the guidance function
            input_to_state (function): A function that takes the model output and returns the robot state
            score_to_output (function): A function that takes the model output, the guidance score and the indices of the joint states and returns the modified model output
            robot_state_ranges (list): A list of tuples containing the min and max values for each joint state
        """
        self.guidance_func_file = guidance_func_file
        self.input_to_state = input_to_state
        self.score_to_output = score_to_output
        self.robot_state_ranges = robot_state_ranges
        if self.guidance_func_file is not None:
            self.guidance_func = self.load_guidance_func(self.guidance_func_file)

        #if DEBUG env variable is set, print debug info
        if os.environ.get("DEBUG"):
            self.debug = True
        else:
            self.debug = False

        self.normalize_guidance = True


    def guide(self, model_output: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Applies the guidance function to the input joint states and returns the guidance score in the format of the model output
        
        Args:
            model_output (list): The model output
            
        Returns:
            list: The modified model output"""
        
        if self.input_to_state is not None:
            states, indices = self.input_to_state(model_output)
        else:
            states = self.model_output

        if self.guidance_func_file is None:
            print ("No guidance function provided")
            return model_output
        
        #check if file exists
        if not os.path.exists(self.guidance_func_file):
            print ("Guidance function file does not exist")
            return model_output
        
        flatten_states = states.view(-1, states.shape[-1]) #flatten states (n_states, n_features)

        #apply guidance function
        guidance_score = torch.tensor([self.guidance_func(s) for s in flatten_states.cpu().detach().numpy()]).to(states.device)

        #reshape guidance score to match input
        guidance_score = guidance_score.reshape(states.shape[:-1])

        #normalize guidance score
        if self.normalize_guidance and torch.max(guidance_score) != torch.min(guidance_score):
            guidance_score = (guidance_score - torch.min(guidance_score))/(torch.max(guidance_score) - torch.min(guidance_score))

        guidance_score = guidance_score.unsqueeze(-1)
        # print("guidance_score.shape: ",guidance_score.shape)

        output = self.score_to_output(model_output, guidance_score, indices)
        return output

    def load_guidance_func(self, guidance_func_file: str) -> None:
        """Loads the guidance function from a file"""

        #check if dir is in path
        if os.path.dirname(guidance_func_file) not in sys.path:
            sys.path.append(os.path.dirname(guidance_func_file))

        print(guidance_func_file.split('/')[-1].split('.')[0])
        #load guidance function
        guidance_func = importlib.import_module(guidance_func_file.split('/')[-1].split('.')[0]).guidance

        print("Successfully loaded guidance function! ")

        self.guidance_func = guidance_func
        self.guidance_func_file = guidance_func_file

    def __call__(self, model_output: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.guide(model_output)
    