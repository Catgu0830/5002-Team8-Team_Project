import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- Configuration ------------------------------
VARS = ['HUFL', 'HULL', 'LUFL', 'LULL', 'MUFL', 'MULL', 'OT']
LEAD_FILE  = '/Users/songyutong/Desktop/metrics_by_lead.csv'           
ERR_FILE   = '/Users/songyutong/Desktop/fixed_lead_errors.csv'     
HOUR_FILE  = '/Users/songyutong/Desktop/metrics_by_hour.csv'        
STRAT_FILE = '/Users/songyutong/Desktop/stratified_metrics.csv'     
SETTING    = 'L96_H192_zscore, seed 2026'

lead_df  = pd.read_csv(LEAD_FILE)
err_df   = pd.read_csv(ERR_FILE)
hour_df  = pd.read_csv(HOUR_FILE)
strat_df = pd.read_csv(STRAT_FILE)

# =====================================================================
# Figure 1: error vs forecast lead (linear + log panels)
# =====================================================================
mse_piv = lead_df.pivot(index='lead', columns='variable',
                        values='mse_common_zscore')[VARS]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
for ax, scale in zip(axes, ['linear', 'log']):
    for v in VARS:
        ax.plot(mse_piv.index, mse_piv[v], label=v, lw=1.4)
    if scale == 'log':
        ax.set_yscale('log')
    ax.set_title(f'Error vs forecast lead ({scale} y-scale)', fontsize=12)
    ax.set_xlabel('Forecast lead (15-min steps; 96 steps = 1 day)')
    ax.set_ylabel('MSE (common z-score scale)')
    ax.grid(alpha=0.3)
    ax.axvline(96, color='grey', ls='--', lw=0.8)
axes[1].legend(fontsize=8, loc='lower right')
fig.suptitle(f'TimesNet test error on ETTm2 ({SETTING}): MSE by forecast lead, per variable',
             fontsize=12.5)
fig.tight_layout()
fig.savefig('L96_s2026_lead_growth_curves.png', dpi=200, bbox_inches='tight')
plt.show()

# =====================================================================
# Figure 2: residual ACF at fixed leads (never concatenate windows)
# =====================================================================
def acf(x, max_lag):
    x = np.asarray(x, float); x = x - x.mean()
    d = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / d if k else 1.0
                     for k in range(max_lag + 1)])

WANTED = [1, 24, 48, 96, 168, 336]
LEADS  = [L for L in WANTED if L in err_df['lead'].unique()]  # auto-detect
MAX_LAG = 100
acf_store = {}
for L in LEADS:
    sub = err_df[err_df['lead'] == L].sort_values('origin_row')
    assert sub['origin_row'].is_unique, f'duplicate origins at lead {L}'
    for v in VARS:
        acf_store[(L, v)] = acf(sub[f'{v}_signed_error_zscore'].values, MAX_LAG)

fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True, sharey=True)
axes = axes.ravel()
colors = plt.cm.viridis(np.linspace(0, 0.95, len(LEADS)))
for i, v in enumerate(VARS):
    ax = axes[i]
    for j, L in enumerate(LEADS):
        ax.plot(range(MAX_LAG + 1), acf_store[(L, v)],
                color=colors[j], lw=1.4, label=f'lead {L}')
    ax.axvline(96, color='grey', ls='--', lw=0.8)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_title(v, fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.15, 1.02)
axes[7].axis('off')
axes[7].legend(*axes[0].get_legend_handles_labels(), fontsize=11,
               title='Fixed forecast lead', loc='center')
for ax in axes[4:7]:
    ax.set_xlabel('Lag (15-min steps; 96 = 1 day)')
for ax in axes[::4]:
    ax.set_ylabel('Residual ACF')
fig.suptitle(f'Residual autocorrelation across forecast origins, per variable, {SETTING}\n'
             '(errors at fixed forecast lead; overlapping windows never concatenated)',
             fontsize=12.5)
fig.tight_layout()
fig.savefig('L96_s2026_residual_acf.png', dpi=200, bbox_inches='tight')
plt.show()

# =====================================================================
# Figure 3: intraday error profile
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharex=True)
axes[0].plot(hour_df['target_hour'], hour_df['mse_common_zscore'], '-o', ms=4, color='#4c72b0')
axes[0].set_title('MSE by target hour of day')
axes[0].set_ylabel('MSE (common z-score scale)')
axes[1].plot(hour_df['target_hour'], hour_df['mae_common_zscore'], '-s', ms=4, color='#dd8452')
axes[1].set_title('MAE by target hour of day')
axes[1].set_ylabel('MAE (common z-score scale)')
for ax in axes:
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel('Target timestamp hour (00-23, local)')
    ax.grid(alpha=0.3)
    ax.axvspan(13, 18, color='red', alpha=0.07)
axes[0].annotate('afternoon plateau 13:00-18:00',
                 xy=(15.5, hour_df['mse_common_zscore'].max()*0.97),
                 ha='center', fontsize=9, color='darkred')
fig.suptitle(f'Intraday error profile, TimesNet ETTm2 {SETTING}', fontsize=12.5)
fig.tight_layout()
fig.savefig('L96_s2026_intraday_error_profile.png', dpi=200, bbox_inches='tight')
plt.show()

# =====================================================================
# Figure 4: retrospective stratification (volatility / jump)
# =====================================================================
piv_vol = (strat_df[strat_df['group'].isin(['high_volatility', 'other_valid_volatility'])]
           .pivot(index='variable', columns='group', values='mse_common_zscore').reindex(VARS))
piv_jump = (strat_df[strat_df['group'].isin(['jump', 'non_jump'])]
            .pivot(index='variable', columns='group', values='mse_common_zscore').reindex(VARS))

x = np.arange(len(VARS)); w = 0.36
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
axes[0].bar(x - w/2, piv_vol['other_valid_volatility'], w,
            label='other (lower-volatility) periods', color='#9ecae1')
axes[0].bar(x + w/2, piv_vol['high_volatility'], w,
            label='high-volatility periods', color='#de2d26')
axes[0].set_yscale('log')
axes[0].set_title('Error stratified by target-time volatility')
for i, v in enumerate(VARS):
    r = piv_vol.loc[v, 'high_volatility'] / piv_vol.loc[v, 'other_valid_volatility']
    axes[0].text(i, piv_vol.loc[v].max()*1.15, f'{r:.2f}x', ha='center', fontsize=8.5)
axes[1].bar(x - w/2, piv_jump['non_jump'], w, label='non-jump', color='#c7e9c0')
axes[1].bar(x + w/2, piv_jump['jump'], w, label='jump', color='#238b45')
axes[1].set_title('Error stratified by target-time price jump')
for i, v in enumerate(VARS):
    r = piv_jump.loc[v, 'jump'] / piv_jump.loc[v, 'non_jump']
    axes[1].text(i, piv_jump.loc[v].max()*1.15, f'{r:.2f}x', ha='center', fontsize=8.5)
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels(VARS)
    ax.set_ylabel('MSE (common z-score scale, log)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
fig.suptitle(f'Retrospective stratification of test error, {SETTING}\n'
             '(labels at target time; descriptive - no causal claim)', fontsize=12)
fig.tight_layout()
fig.savefig('L96_s2026_stratified_error_layers.png', dpi=200, bbox_inches='tight')
plt.show()

# =====================================================================
# Summary statistics for the report
# =====================================================================
print('[lead growth] MSE increase lead1->lead336 by variable (%):')
print(((mse_piv.loc[336] - mse_piv.loc[1]) / mse_piv.loc[1] * 100).round(1)
      .sort_values(ascending=False).to_string())
print(f'\n[ACF@lag96] available leads = {LEADS}')
print(pd.DataFrame([{'lead': L, **{v: round(acf_store[(L, v)][96], 3) for v in VARS}}
                    for L in LEADS]).set_index('lead').to_string())
mse_h = hour_df['mse_common_zscore'].values
print(f'\n[intraday] MSE min={mse_h.min():.4f} (hour {mse_h.argmin()}) | '
      f'max={mse_h.max():.4f} (hour {mse_h.argmax()}) | max/min={mse_h.max()/mse_h.min():.2f}')
print('[intraday] 4-hour block means:',
      {f'{4*i:02d}-{4*i+3:02d}': round(float(mse_h[4*i:4*i+4].mean()), 4) for i in range(6)})
print('\n[stratified] MSE ratio high_vol/other:')
print((piv_vol['high_volatility']/piv_vol['other_valid_volatility']).round(2).to_string())
print('\n[stratified] MSE ratio jump/non_jump:')
print((piv_jump['jump']/piv_jump['non_jump']).round(2).to_string())