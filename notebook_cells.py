"""
After running classify_contexts.py, use this in your notebook to load the mapping
and run the context-level analysis.

Copy-paste these cells into your notebook.
"""

# --- Cell 1: Load context groups ---
CELL_1 = """
context_groups = pd.read_csv('context_groups.csv')
combined_df = combined_df.merge(context_groups, on='context', how='left')
combined_df['context_group'] = combined_df['context_group'].fillna('Other')
combined_df['context_group'].value_counts()
"""

# --- Cell 2: Falsehood rate by context group ---
CELL_2 = """
combined_df_filtered = combined_df[combined_df['party'] != 'none'].copy()

context_label_counts = combined_df_filtered.groupby(['context_group', 'label']).size().unstack(fill_value=0)
context_totals = context_label_counts.sum(axis=1)

falseish_labels = ['barely-true', 'FALSE', 'pants-fire']
context_false_counts = context_label_counts[falseish_labels].sum(axis=1)
context_false_rate = (context_false_counts / context_totals * 100).sort_values(ascending=False)

context_summary = pd.DataFrame({
    'total_statements': context_totals,
    'false_count': context_false_counts,
    'false_rate_%': context_false_rate
}).sort_values('false_rate_%', ascending=False)

context_summary
"""

# --- Cell 3: Visuals ---
CELL_3 = """
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
context_label_sorted = context_label_counts.loc[context_totals.sort_values(ascending=False).index]
context_label_sorted.plot(kind='bar', stacked=True, ax=ax1,
                          color=['#c0392b', '#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#2ecc71'])
ax1.set_title('Statement Distribution by Context Group', fontsize=14, fontweight='bold')
ax1.set_xlabel('Context Group')
ax1.set_ylabel('Count')
ax1.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.tick_params(axis='x', rotation=45)

ax2 = axes[1]
rate_sorted = context_false_rate.sort_values()
bars = ax2.barh(range(len(rate_sorted)), rate_sorted.values,
                color=['#e74c3c' if x > 45 else '#f39c12' for x in rate_sorted.values])
ax2.set_yticks(range(len(rate_sorted)))
ax2.set_yticklabels(rate_sorted.index)
ax2.set_xlabel('False-ish Rate (%)')
ax2.set_title('Falsehood Rate by Context Group', fontsize=14, fontweight='bold')
for i, v in enumerate(rate_sorted.values):
    ax2.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10)

plt.tight_layout()
plt.show()
"""

if __name__ == "__main__":
    print("Copy these cells into your notebook after running classify_contexts.py:\n")
    print("=" * 60)
    print("CELL 1 - Load context groups:")
    print(CELL_1)
    print("=" * 60)
    print("CELL 2 - Falsehood rate by context group:")
    print(CELL_2)
    print("=" * 60)
    print("CELL 3 - Visuals:")
    print(CELL_3)
