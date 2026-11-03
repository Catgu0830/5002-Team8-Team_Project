# =====================================================================
# Intraday error profile + retrospective stratification, single script
# TimesNet ETTm2, setting L192-H336-zscore, seed 2021
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Configuration ------------------------------
HOUR_FILE  = 'metrics_by_hour(1).csv'      # note: "(1)" in filename
STRAT_FILE = 'stratified_metrics.csv'
VARS = ['HUFL', 'HULL', 'LUFL', 'LULL', 'MUFL', 'MULL', 'OT']
OUT_INTRADAY = 'intraday_error_profile.png'
OUT_STRAT    = 'stratified_error_layers.png'

hour_df  = pd.read_csv(HOUR_FILE)
strat_df = pd.read_csv(STRAT_FILE)

# =====================================================================
# Figure 1: intraday error profile (daily periodicity of error)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharex=True)

axes[0].plot(hour_df['target_hour'], hour_df['mse_common_zscore'],
             '-o', ms=4, color='#4c72b0')
axes[0].set_title('MSE by target hour of day')
axes[0].set_ylabel('MSE (common z-score scale)')

axes[1].plot(hour_df['target_hour'], hour_df['mae_common_zscore'],
             '-s', ms=4, color='#dd8452')
axes[1].set_title('MAE by target hour of day')
axes[1].set_ylabel('MAE (common z-score scale)')

for ax in axes:
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel('Target timestamp hour (00-23, local)')
    ax.grid(alpha=0.3)
    ax.axvspan(13, 18, color='red', alpha=0.07)   # afternoon plateau
axes[0].annotate('afternoon plateau 13:00-18:00',
                 xy=(15.5, hour_df['mse_common_zscore'].max()*0.97),
                 ha='center', fontsize=9, color='darkred')

fig.suptitle('Intraday error profile, TimesNet ETTm2 L192-H336-zscore, seed 2021',
             fontsize=12.5)
fig.tight_layout()
fig.savefig(OUT_INTRADAY, dpi=200, bbox_inches='tight')
plt.show()

# Summary statistics for the report
mse_h = hour_df['mse_common_zscore'].values
print('[Intraday] MSE min = %.4f (hour %d) | max = %.4f (hour %d) | max/min = %.2f'
      % (mse_h.min(), mse_h.argmin(), mse_h.max(), mse_h.argmax(),
         mse_h.max()/mse_h.min()))
print('[Intraday] 4-hour block means:',
      {f'{4*i:02d}-{4*i+3:02d}': round(float(mse_h[4*i:4*i+4].mean()), 4)
       for i in range(6)})

# =====================================================================
# Figure 2: retrospective stratification (volatility / jump layers)
# =====================================================================
piv_vol = (strat_df[strat_df['group'].isin(['high_volatility', 'other_valid_volatility'])]
           .pivot(index='variable', columns='group', values='mse_common_zscore')
           .reindex(VARS))
piv_jump = (strat_df[strat_df['group'].isin(['jump', 'non_jump'])]
            .pivot(index='variable', columns='group', values='mse_common_zscore')
            .reindex(VARS))

x = np.arange(len(VARS)); w = 0.36
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)

# Panel A: high-volatility vs other periods
axes[0].bar(x - w/2, piv_vol['other_valid_volatility'], w,
            label='other (lower-volatility) periods', color='#9ecae1')
axes[0].bar(x + w/2, piv_vol['high_volatility'], w,
            label='high-volatility periods', color='#de2d26')
axes[0].set_yscale('log')
axes[0].set_title('Error stratified by target-time volatility')
for i, v in enumerate(VARS):
    r = piv_vol.loc[v, 'high_volatility'] / piv_vol.loc[v, 'other_valid_volatility']
    axes[0].text(i, piv_vol.loc[v].max()*1.15, f'{r:.2f}x', ha='center', fontsize=8.5)

# Panel B: jump vs non-jump target times
axes[1].bar(x - w/2, piv_jump['non_jump'], w, label='non-jump', color='#c7e9c0')
axes[1].bar(x + w/2, piv_jump['jump'], w, label='jump', color='#238b45')
axes[1].set_title('Error stratified by target-time price jump')
for i, v in enumerate(VARS):
    r = piv_jump.loc[v, 'jump'] / piv_jump.loc[v, 'non_jump']
    axes[1].text(i, piv_jump.loc[v].max()*1.15, f'{r:.2f}x', ha='center', fontsize=8.5)

for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(VARS)
    ax.set_ylabel('MSE (common z-score scale, log)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

fig.suptitle('Retrospective stratification of test error, L192-H336-zscore, seed 2021\n'
             '(labels at target time; descriptive - no causal claim)', fontsize=12)
fig.tight_layout()
fig.savefig(OUT_STRAT, dpi=200, bbox_inches='tight')
plt.show()

# Summary tables for the report
ratio_vol  = (piv_vol['high_volatility'] / piv_vol['other_valid_volatility']).round(2)
ratio_jump = (piv_jump['jump'] / piv_jump['non_jump']).round(2)
vol_share  = (strat_df[strat_df['group'] == 'high_volatility']
              .set_index('variable')['forecast_pairs']
              .div(strat_df[strat_df['group'].isin(['high_volatility', 'other_valid_volatility'])]
                   .groupby('variable')['forecast_pairs'].sum())
              .mul(100).round(1).reindex(VARS))
print('\n[Stratified] MSE ratio high_vol/other:\n', ratio_vol.to_string())
print('\n[Stratified] MSE ratio jump/non_jump:\n', ratio_jump.to_string())
print('\n[Stratified] high-volatility share of forecast pairs (%):\n', vol_share.to_string())