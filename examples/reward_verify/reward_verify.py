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
# - 把dataproto 丢进 val_reward_fn，这里的case是NaiveRewardManager
# - output score并查看

import os

class CSVVerifier:
    def __init__(self, csv_path, tokenizer, val_reward_fn):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_root = os.path.join(self.script_dir, "data")
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.val_reward_fn = val_reward_fn
        self.gsm8k_test_batch = None # DataProto list for gsm8k
        self.math_test_batch = None # DataProto list for math
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
        
        tokenized_responses = self.tokenizer(responses, padding=True, truncation=True, return_tensors="pt") #这里看起来是右填充
        self.tokenizer.padding_side = "left"  # 确保左填充
        tokenized_inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        
        # Divide dataset into gsm8k (0-1318) and math (1319-6318)
           # 注意：这里我们需要构造一个拼接后的 attention_mask，
        # 使其长度等于 prompt + response 的总长度，从而方便 reward_manager 分割
        gsm8k_prompt_ids = tokenized_inputs["input_ids"][:1319]
        gsm8k_response_ids = tokenized_responses["input_ids"][:1319]
        gsm8k_prompt_mask = tokenized_inputs["attention_mask"][:1319]
        gsm8k_response_mask = tokenized_responses["attention_mask"][:1319]
        combined_attention_mask = torch.cat([gsm8k_prompt_mask, gsm8k_response_mask], dim=-1)
        
        # 构造 gsm8k 的 batch 字典，其中 "prompts" 键保存 prompt 的 token_ids
        gsm8k_inputs = {
            "input_ids": gsm8k_prompt_ids,  # 可选，实际使用中可能只用 "prompts"
            "prompts": gsm8k_prompt_ids,
            "attention_mask": combined_attention_mask,
            "responses": gsm8k_response_ids
        }
        
        # 对于 MATH 数据，同样构造拼接后的 attention_mask
        math_prompt_ids = tokenized_inputs["input_ids"][1319:6319]
        math_response_ids = tokenized_responses["input_ids"][1319:6319]
        math_prompt_mask = tokenized_inputs["attention_mask"][1319:6319]
        math_response_mask = tokenized_responses["attention_mask"][1319:6319]
        combined_attention_mask_math = torch.cat([math_prompt_mask, math_response_mask], dim=-1)
        
        math_inputs = {
            "input_ids": math_prompt_ids,
            "prompts": math_prompt_ids,
            "attention_mask": combined_attention_mask_math,
            "responses": math_response_ids
        }
        # get ground truth 'ground_truth' and source 'data_source'
        # 'openai/gsm8k':1319 datapoints(index 0-1318) 'lighteval/MATH': 5000 datapoints(index 1319-6318)
       # Load ground truth and data source
        gsm8k_dataset = pd.read_parquet(self.dataset_root + "/gsm8k/test.parquet")
        gsm8k_ground_truth = gsm8k_dataset["reward_model"].apply(lambda x: x["ground_truth"]).tolist()

        math_dataset = pd.read_parquet(self.dataset_root + "/math/test.parquet")
        math_ground_truth = math_dataset["reward_model"].apply(lambda x: x["ground_truth"]).tolist()
        
        # generate dict for 'openai/gsm8k' and 'lighteval/MATH' with keys of  'input_ids', 'responses', ['reward_model']['ground_truth'] and  'data_source'
        # Construct data dictionaries
        import numpy as np

        gsm8k_reward_model = np.array([{"ground_truth": gt} for gt in gsm8k_ground_truth], dtype=object)
        math_reward_model = np.array([{"ground_truth": gt} for gt in math_ground_truth], dtype=object)
        
        gsm8k_test_data = {
            "batch": gsm8k_inputs,
            "non_tensor_batch": {
                "reward_model": gsm8k_reward_model,
                "data_source": np.array(["openai/gsm8k"] * 1319)
            }
        }
        math_test_data = {
            "batch": math_inputs,
            "non_tensor_batch": {
                "reward_model": math_reward_model,
                "data_source": np.array(["lighteval/MATH"] * 5000)
            }
        }
        
        # print("GSM8K test data dictionary:", gsm8k_test_data)
        # Convert to DataProto using from_dict()
        self.gsm8k_test_batch = DataProto.from_dict(
            tensors=gsm8k_test_data["batch"],
            non_tensors=gsm8k_test_data["non_tensor_batch"]
        )
        self.math_test_batch = DataProto.from_dict(
            tensors=math_test_data["batch"],
            non_tensors=math_test_data["non_tensor_batch"]
        )

        
        

    def prepare_evaluation(self):
        self.load_and_process_csv()
        
    def evaluate_gsm8k(self):
        """
        使用已加载的 GSM8K 数据（self.gsm8k_test_batch）调用奖励函数进行评估，
        并打印出平均分和部分样本得分。
        """
        # 检查 GSM8K 数据是否加载成功
        if self.gsm8k_test_batch is None:
            raise ValueError("GSM8K test batch is not prepared. Please call load_and_process_csv() first.")

        print("Evaluating GSM8K dataset...")
        
        # 计算奖励，假设 val_reward_fn 接受 DataProto 并返回一个 reward tensor
        reward_tensor = self.val_reward_fn(self.gsm8k_test_batch)
        
        # 求和得到每个样本的得分，转到 CPU 并转换为列表
        scores = reward_tensor.sum(-1).cpu().tolist()
        
        # 计算平均得分
        avg_score = np.mean(scores)
        
        # 可选：输出部分样本的得分和对应的输入文本
        input_ids = self.gsm8k_test_batch.batch.get('input_ids', None)
        if input_ids is not None:
            sample_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids[:5]]
        else:
            sample_inputs = []
        
        print("GSM8K Evaluation:")
        print("Average score:", avg_score)
        print("Sample inputs:", sample_inputs)
        print("Sample scores:", scores[:5])
        
        return avg_score
    def evaluate_math(self):
        """
        使用已加载的 MATH 数据（self.math_test_batch）调用奖励函数进行评估，
        并打印出平均分和部分样本得分。
        """
        # 检查 MATH 数据是否已加载
        if self.math_test_batch is None:
            raise ValueError("MATH test batch is not prepared. Please call load_and_process_csv() first.")

        print("Evaluating MATH dataset...")

        # 计算奖励，假设 val_reward_fn 接受 DataProto 并返回一个 reward tensor
        reward_tensor = self.val_reward_fn(self.math_test_batch)

        # 求和得到每个样本的得分，转到 CPU 并转换为列表
        scores = reward_tensor.sum(-1).cpu().tolist()

        # 计算平均得分
        avg_score = np.mean(scores)

        # 可选：输出部分样本的得分和对应的输入文本
        input_ids = self.math_test_batch.batch.get('input_ids', None)
        if input_ids is not None:
            sample_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids[:5]]
        else:
            sample_inputs = []

        print("MATH Evaluation:")
        print("Average score:", avg_score)
        print("Sample inputs:", sample_inputs)
        print("Sample scores:", scores[:5])

        return avg_score
    
    def evaluate(self):
        # 执行验证
        print("Evaluating GSM8K...")
        gsm8k_score = self.evaluate_gsm8k()
        print("Evaluating MATH...")
        math_score = self.evaluate_math()
        
        return {
            "gsm8k_score": gsm8k_score,
            "math_score": math_score
        }
    
        

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
    val_reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=5, compute_score=None)  # Replace with actual reward function

    verifier = CSVVerifier(args.csv_path, tokenizer, val_reward_fn)
    verifier.prepare_evaluation()
    results = verifier.evaluate()
    print(f"Validation Results: {results}")
    
    # verifier.evaluate_gsm8k()
    # verifier.evaluate_math()

if __name__ == "__main__":
    main()