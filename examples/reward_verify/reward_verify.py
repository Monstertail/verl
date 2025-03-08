from verl import DataProto
import pandas as pd
import torch
import numpy as np
import argparse
from verl.workers.reward_manager import NaiveRewardManager
from verl.utils.reward_score import _default_compute_score, gsm8k, math
from verl.utils.tokenizer import hf_tokenizer
# adapted from _validate: https://github.com/ocss884/verl/blob/3165d98894ecf97650ebe9f40434a586b54dbc25/verl/trainer/ppo/ray_trainer.py#L598

# - load csv 然后转换到dataproto
# - 把daatproto 丢进 val_reward_fn，这里的case是NaiveRewardManager
# - output score并查看


class CSVVerifier:
    def __init__(self, csv_path, tokenizer, val_reward_fn):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.val_reward_fn = val_reward_fn

    def load_and_process_csv(self):
        '''
        Load CSV and prepare for reward manager: https://github.com/Monstertail/verl/blob/3165d98894ecf97650ebe9f40434a586b54dbc25/docs/preparation/reward_function.rst#rewardmanager
        '''
        df = pd.read_csv(self.csv_path)
        
        # extract prompt,response and tokenize to 'input_ids', 'responses'
        # Extract 'prompt' and 'response' columns
        prompts = df["prompt"].tolist()
        responses = df["response"].tolist()

        # Tokenize both prompts and responses
        tokenized_inputs = self.tokenizer(prompts, return_tensors="pt")
        tokenized_responses = self.tokenizer(responses, return_tensors="pt")
        
        # get ground truth 'ground_truth' and source 'data_source'

        # Convert to dictionary format for DataProto
        data_list = df.to_dict(orient="records")

        # Convert each row into DataProto
        data_proto_list = [DataProto.from_single_dict(data) for data in data_list]

        return data_proto_list

    def evaluate(self):
        data_proto_list = self.load_and_process_csv()
        

# Example usage
# tokenizer = YourTokenizer()
# val_reward_fn = NaiveRewardManager()  # Replace with actual reward function
# verifier = CSVVerifier("your_file.csv", tokenizer, val_reward_fn)
# results = verifier.evaluate()



def main():
    parser = argparse.ArgumentParser(description="Run CSV verifier with reward function.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    args = parser.parse_args()

    # Initialize tokenizer and reward function 
    tokenizer =  hf_tokenizer(name_or_path=args.model_path)
    # _default_compute_score, see here: https://github.com/Monstertail/verl/blob/3165d98894ecf97650ebe9f40434a586b54dbc25/verl/utils/reward_score/__init__.py#L17C28-L17C39 
    val_reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=4000, compute_score=None)  # Replace with actual reward function

    verifier = CSVVerifier(args.csv_path, tokenizer, val_reward_fn)
    results = verifier.evaluate()

if __name__ == "__main__":
    main()