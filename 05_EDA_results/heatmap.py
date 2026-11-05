import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

df = pd.read_csv('aggregate_by_variable.csv')

# 1) Build the setting label (columns)
df['setting'] = ('L' + df['input_length'].astype(str) + '_H' + df['horizon'].astype(str)
                 + '_' + df['input_scaling'])
setting_order  = ['L96_H96_zscore', 'L96_H192_zscore', 'L96_H336_zscore', 'L96_H720_zscore',
                  'L192_H336_zscore', 'L384_H336_zscore', 'L96_H336_minmax']
setting_labels = ['L96, H96\nz-score', 'L96, H192\nz-score', 'L96, H336\nz-score', 'L96, H720\nz-score',
                  'L192, H336\nz-score', 'L384, H336\nz-score', 'L96, H336\nmin-max']

# 2) Order rows by mean MSE across settings, hardest first
var_order = (df.groupby('variable')['mse_common_zscore_mean'].mean()
             .sort_values(ascending=False).index.tolist())

def heat_matrix(value_col):
    return (df.pivot_table(index='variable', columns='setting', values=value_col)
              .reindex(index=var_order, columns=setting_order))

mse_mat = heat_matrix('mse_common_zscore_mean')
mae_mat = heat_matrix('mae_common_zscore_mean')

# 3) Two panels sharing a log colour scale (errors span ~2 orders of magnitude)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
norm = mcolors.LogNorm(vmin=df['mse_common_zscore_mean'].min()*0.9,
                       vmax=df['mse_common_zscore_mean'].max()*1.1)

for ax, mat, title in zip(axes, [mse_mat, mae_mat], ['MSE', 'MAE']):
    im = ax.imshow(mat.values, aspect='auto', cmap='viridis', norm=norm)
    ax.set_xticks(range(len(setting_order))); ax.set_xticklabels(setting_labels, fontsize=9)
    ax.set_yticks(range(len(var_order)));     ax.set_yticklabels(var_order, fontsize=10)
    ax.set_title(f'TimesNet test error on ETTm2 ({title}, common z-score scale)', fontsize=12)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if norm(v) < 0.6 else 'black')
    ax.set_xlabel('Experimental setting (input length, horizon, external scaling)')

cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label('Error (log colour scale; cell text shows raw value)', fontsize=10)
fig.suptitle('Variable × setting: mean over 3 random seeds (n=3), sorted by difficulty (hardest at top)',
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig('timesnet_variable_error_heatmap_en.png', dpi=200, bbox_inches='tight')
plt.show()