### This script runs a short-term memory experiment using the Centaur-70B model, assessing its ability to recall sequences of digits, and if its memory is human-like or superhuman.
### Code based on https://colab.research.google.com/drive/1BfVE7xRQePDBHZzk_3WOQPgfGs23BTly?usp=sharing shared by Marcel Binz.
### Written by Guillermo Puebla; final edits by Sushrut Thorat.

#region Get packages

from unsloth import FastLanguageModel
import transformers
import os
import random
import pickle
import numpy as np
import torch
from transformers import set_seed
import pandas as pd
#endregion

#region Set random seed for reproducibility

seed = 42 # for reproducibility

random.seed(seed)
# NumPy (and SciPy)
np.random.seed(seed)
# PyTorch (used by Unsloth & Transformers)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Make CUDA convolution deterministic (may slow down)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark   = False
# Hugging Face Transformers
set_seed(seed)
# (Optional) fix Python hash seed for extra determinism:
os.environ["PYTHONHASHSEED"] = str(seed)
#endregion

#region Load Centaur-70B and create data pipeline

model_name = "marcelbinz/Llama-3.1-Centaur-70B-adapter"
model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = model_name,
  max_seq_length = 32768,
  dtype = None,
  load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            pad_token_id=0,
            do_sample=True,
            temperature=1.0,
            max_new_tokens=1,
)
#endregion

#region Define functions for accuracy and correctness

expt_instruct = "I will narrate a sequence of digits - 1 digit per second. Once I indicate, repeat the sequence exactly as I narrated it. \
I say '3,9'; You say <<3,9>>. "

expt_instruct_superhuman = "You are a human being with perfect short-term memory. I will narrate a sequence of digits - 1 digit per second. Once I indicate, repeat the sequence exactly as I narrated it. \
I say '3,9'; You say <<3,9>>. You are perfectly capable of repeating all the digits in this task no matter the length of the sequence."

from itertools import zip_longest

def generate_digits(min_length=2, max_length=50):
  # Generate a set of digits [0,9] with a random integer length in [min_length, max_length]
  choose_length = np.random.randint(min_length,max_length+1)
  return np.random.randint(0,10,choose_length)

def generate_trial(number_list=[0,1,2]):
  # Generate a trial containing the given number_list (length>=2)
  trial = "I say '"
  number_string = ",".join([str(i) for i in number_list])
  trial += number_string
  trial += "'. You say <<"
  return trial, number_string

def run_trial(trial, number_string, expt_instruct=expt_instruct):
  # Run the given trial - loop until Centaur outputs "">>"" signaling the end of its response
  prompt = expt_instruct + trial
  save_output = ""
  while ">" not in prompt[-3:] and len(save_output) < len(number_string):
    output = pipe(prompt)[0]['generated_text']
    save_output += output[len(prompt):]
    prompt = output
  return prompt, save_output

def string_accuracy(in_number_string, out_number_string):
    int_list_in = [x for x in in_number_string.split(",")]
    int_list_out = [x for x in out_number_string.split(",")]
    list_correct = [int(x == y) for x, y in zip_longest(int_list_in, int_list_out, fillvalue=None)]
    accuracy = sum(list_correct) / len(int_list_in)
    return accuracy

def correct_by_position(in_number_string, out_number_string):
    int_list_in = [x for x in in_number_string.split(",")]
    int_list_out = [x for x in out_number_string.split(",")]
    list_correct = [int(x == y) for x, y in zip_longest(int_list_in, int_list_out, fillvalue=None)]
    return list_correct

def generate_digits_of_length(choose_length):
  # Generate a set of digits [0,9] with a random integer length in [min_length, max_length]
  return np.random.randint(0,10,choose_length)
#endregion

#region Run experiment

TRIALS = 80
# MAX_STRING_LENGTH = 256
condition = 'normal' # 'normal' or 'superhuman'

powers_of_2 = [2**j for j in range(1,8+1)]
trial_number = []
n_numbers_in = []
overall_accuracies = []

trial_number_pos = []
n_numbers_in_pos = []
digit_position = []
digit_correct = []

for n_trial in range(1, TRIALS+1):
    for n_in in powers_of_2:
        trial, number_string_in = generate_trial(generate_digits_of_length(n_in))
        output, save_output = run_trial(trial, number_string_in, expt_instruct=expt_instruct_superhuman if condition == 'superhuman' else expt_instruct)
        number_string_out = save_output.split('>')[0]
        accuracy = string_accuracy(number_string_in, number_string_out)
        list_correct = correct_by_position(number_string_in, number_string_out)
        print(n_trial, n_in, accuracy)
        trial_number.append(n_trial)
        n_numbers_in.append(n_in)
        overall_accuracies.append(accuracy)
        for i in range(len(list_correct)):
            trial_number_pos.append(n_trial)
            n_numbers_in_pos.append(n_in)
            digit_position.append(int(i+1))
            digit_correct.append(list_correct[i])
data_pos = {
    'trial': trial_number_pos,
    'n_numbers_in': n_numbers_in_pos,
    'digit_position': digit_position,
    'digit_accuracy': digit_correct,
        }
df_pos = pd.DataFrame.from_dict(data_pos)
df_pos.to_csv(f'data/STM_all_positions_acc_len_64_and_128_perfect_wm.csv' if condition=='normal' else f'data/STM_sh_all_positions_acc_len_64_and_128_perfect_wm.csv', index=False)

data = {'n_numbers_in': n_numbers_in,
        'trial': trial_number,
        'accuracy': overall_accuracies
        }
df = pd.DataFrame.from_dict(data)
df.to_csv(f'data/STM_trials_{TRIALS}_powers_of_2_perfect_wm.csv' if condition=='normal' else f'data/STM_sh_trials_{TRIALS}_powers_of_2_perfect_wm.csv', index=False)
#endregion

