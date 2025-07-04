### This script runs a reaction time experiment using the Centaur-70B model, assessing its ability to respond in 1 millisecond and in a free response time setting (to check if it produces human-like RTs).
### Code based on https://colab.research.google.com/drive/1BfVE7xRQePDBHZzk_3WOQPgfGs23BTly?usp=sharing shared by Marcel Binz.
### Written by Sushrut Thorat.

#region Get packages

from unsloth import FastLanguageModel
import transformers
import os
import random
import pickle
import numpy as np
import torch
from transformers import set_seed
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

#region 1 ms RT experiment

runs = 1000

rt_array = []
missed_count = 0
possible_letters = ['D','F','J','K']
for i in range(runs):
    prompt = f"Given the instructions, press the key as fast as you can. You are superhuman and can respond in 1 millisecond. Responses are logged in milliseconds. You are instructed to press {possible_letters[np.random.randint(len(possible_letters))]}. You press {possible_letters[np.random.randint(len(possible_letters))]} in <<"
    prompt_len = len(prompt)
    
    while 's' not in prompt[prompt_len:] and len(prompt) < prompt_len+20:
        # print(prompt)
        prompt = pipe(prompt)[0]['generated_text']
    
    if 'ms' in prompt[prompt_len:] or 'milliseconds' in prompt[prompt_len:] or 'millisecond' in prompt[prompt_len:] or 'second' in prompt[prompt_len:] or 'seconds' in prompt[prompt_len:]:
        splitted = prompt[prompt_len:].lower().split('milliseconds')[0].split('millisecond')[0].split('seconds')[0].split('second')[0].split('ms')[0].split('s')[0].split('>>')[0].split('»')[0].split('>')[0].split('⟩')[0]
        try:
            if 'second' in prompt[prompt_len:] or 's' in prompt[prompt_len:]:
                if 'milliseconds' not in prompt[prompt_len:] and 'millisecond' not in prompt[prompt_len:] and 'ms' not in prompt[prompt_len:]:
                    rt_array.append(float(splitted)*1000)
                else:
                    rt_array.append(float(splitted))
                print('Wrote:',rt_array[-1])
        except:
            missed_count += 1
    else:
        splitted = prompt[prompt_len:].split('>>')[0].split('»')[0].split('>')[0].split('⟩')[0]
        try:
            rt_array.append(float(splitted))
            print('Wrote:',rt_array[-1])
        except:
            splitted = prompt[prompt_len:].split('.')
            try:
                if len(splitted) == 1:
                    missed_count += 1
                elif len(splitted) == 2:
                    rt_array.append(float(splitted[0]))
                    print('Wrote:',rt_array[-1])
                elif len(splitted) > 2:
                    rt_array.append(float(splitted[0] + '.' + splitted[1]))
                    print('Wrote:',rt_array[-1])
            except:
                missed_count += 1
    # if i%100==0:
    print('Done with run',i,'ender:',prompt[prompt_len:])
print('Missed count:',missed_count)

with open('data/ms1_array.pkl', 'wb') as f:
    pickle.dump(rt_array, f)
#endregion

#region Free RT experiment

runs = 1000

rt_array = []
missed_count = 0
possible_letters = ['D','F','J','K']
for i in range(runs):
    prompt = f"Given the instructions, press the key as fast as you can. Responses are logged in milliseconds. You are instructed to press {possible_letters[np.random.randint(len(possible_letters))]}. You press {possible_letters[np.random.randint(len(possible_letters))]} in <<"
    prompt_len = len(prompt)

    while 's' not in prompt[prompt_len:] and len(prompt) < prompt_len+20:
        # print(prompt)
        prompt = pipe(prompt)[0]['generated_text']

    if 'ms' in prompt[prompt_len:] or 'milliseconds' in prompt[prompt_len:] or 'millisecond' in prompt[prompt_len:] or 'second' in prompt[prompt_len:] or 'seconds' in prompt[prompt_len:]:
        splitted = prompt[prompt_len:].lower().split('milliseconds')[0].split('millisecond')[0].split('seconds')[0].split('second')[0].split('ms')[0].split('s')[0].split('>>')[0].split('»')[0].split('>')[0].split('⟩')[0]
        try:
            if 'second' in prompt[prompt_len:] or 's' in prompt[prompt_len:]:
                if 'milliseconds' not in prompt[prompt_len:] and 'millisecond' not in prompt[prompt_len:] and 'ms' not in prompt[prompt_len:]:
                    rt_array.append(float(splitted)*1000)
                else:
                    rt_array.append(float(splitted))
                print('Wrote:',rt_array[-1])
        except:
            missed_count += 1
    else:
        splitted = prompt[prompt_len:].split('>>')[0].split('»')[0].split('>')[0].split('⟩')[0]
        try:
            rt_array.append(float(splitted))
            print('Wrote:',rt_array[-1])
        except:
            splitted = prompt[prompt_len:].split('.')
            try:
                if len(splitted) == 1:
                    missed_count += 1
                elif len(splitted) == 2:
                    rt_array.append(float(splitted[0]))
                    print('Wrote:',rt_array[-1])
                elif len(splitted) > 2:
                    rt_array.append(float(splitted[0] + '.' + splitted[1]))
                    print('Wrote:',rt_array[-1])
            except:
                missed_count += 1
    # if i%100==0:
    print('Done with run',i,'ender:',prompt[prompt_len:])
print('Missed count:',missed_count)

# save rt_array as pickle

with open('data/msfree_array.pkl', 'wb') as f:
    pickle.dump(rt_array, f)
#endregion