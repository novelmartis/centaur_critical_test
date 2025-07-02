import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

condition = 'normal' # normal or superhuman

df = pd.read_csv("STM_all_positions_acc_len_64_and_128_perfect_wm.csv" if condition == 'normal' else "STM_sh_all_positions_acc_len_64_and_128_perfect_wm.csv")
df['n_numbers_in'] = df['n_numbers_in'].astype('category')
df = df.rename(columns={"n_numbers_in": "Sequence length"})
#df = df[df["Sequence length"] == 64]
palette = sns.color_palette('tab10', 8)

ax = sns.relplot(
	data=df, 
	x="digit_position", 
	y="digit_accuracy", 
	kind="line",
	hue="Sequence length",
	aspect=5/3,
	palette=palette
	)
sns.move_legend(ax, "upper right")
ax.set(xlabel='Digit position', ylabel='Accuracy', ylim=(0.0, 1.0))
plt.tight_layout()
plt.savefig("accuracy_by_position_length_all_perfect_wm.svg" if condition == 'normal' else "accuracy_by_position_length_all_perfect_wm_sh.svg", format='svg', dpi=300)
# plt.show()