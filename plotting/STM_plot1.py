import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


df = pd.read_csv("STM_trials_80_powers_of_2_perfect_wm.csv")
df['n_numbers_in'] = df['n_numbers_in'].astype('category')
df = df.rename(columns={"n_numbers_in": "Sequence length", "trial": "Trial", "accuracy": "Accuracy"})
print(df)

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
ax = sns.stripplot(
	data=df, 
	x="Sequence length", 
	y="Accuracy", 
	jitter=0.2, 
	ax=ax, 
	order=[2, 4, 8, 16, 32, 64, 128, 256],
	marker="."
	)
ax.axhline(0.1, ls='--', color='grey')
ax.axhline(0.9, ls='--', color='red')
mean_values = df.groupby('Sequence length')['Accuracy'].mean().plot(color='maroon')
print("means:")
print(df.groupby('Sequence length')['Accuracy'].mean())

plt.tight_layout()
plt.savefig("accuracy_by_seq_length_perfect_wm.svg", format='svg')
# plt.show()