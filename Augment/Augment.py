import pandas as pd

# Function to handle mutation logic and exclude mutations with the same amino acid transition
def combine_mutation_sites(stab_mutation_str, destab_mutation_str):
    stab_mutations = stab_mutation_str.split(',')
    destab_mutations = destab_mutation_str.split(',')
    combined_mutations = []
    
    stab_dict = {mutation[1:-1]: mutation for mutation in stab_mutations}
    destab_dict = {mutation[1:-1]: mutation for mutation in destab_mutations}
    
    # Iterate through stabilizing mutations and handle position overlaps
    for pos, stab_mut in stab_dict.items():
        if pos in destab_dict:
            initial_aa = stab_mut[-1]
            final_aa = destab_dict[pos][-1]
            if initial_aa != final_aa:
                combined_mutations.append(f"{initial_aa}{pos}{final_aa}")
        else:
            # Keep the flipped stabilizing mutation
            new_amino_acid = stab_mut[-1]
            original_amino_acid = stab_mut[0]
            combined_mutations.append(f"{new_amino_acid}{pos}{original_amino_acid}")

    # Add mutations from destabilizing mutations that do not overlap
    for pos, destab_mut in destab_dict.items():
        if pos not in stab_dict:
            combined_mutations.append(destab_mut)

    return ','.join(sorted(combined_mutations))


df = pd.read_csv('Q_multiple_aug.csv')

df_filtered = df[['B', 'E', 'F', 'H']]  # B: PDB, E: Mutations, F: Mutant Name, H: Stability

combined_mutations = []

# Group by PDB
for pdb, group in df_filtered.groupby('B'):
    stabilizing_mutations = group[group['H'] == 0]
    destabilizing_mutations = group[group['H'] == 1]
    
    # Iterate through stabilizing and destabilizing mutations to create pairs
    for idx_stab, stab_row in stabilizing_mutations.iterrows():
        for idx_destab, destab_row in destabilizing_mutations.iterrows():
            combined_mutation_sites = combine_mutation_sites(stab_row['E'], destab_row['E'])
            
            # Combine the mutations
            combined_mutation = {
                'PDB': pdb,
                'Initial_Mutation': stab_row['F'],
                'Evolving_Mutation': destab_row['F'],
                'Combined_Mutation': f"{stab_row['F']} mutates to {destab_row['F']}",
                'Mutation_Sites': combined_mutation_sites,
                'Stability_Change': 'Destabilizing',
                'L': 1  # Column L value = 1 since this is from stabilizing to destabilizing
            }
            combined_mutations.append(combined_mutation)

combined_mutations_df = pd.DataFrame(combined_mutations)

combined_mutations_df.to_csv('combined_mutations_output.csv', index=False)

print("Combined mutations have been created with correct mutation handling and saved to 'combined_mutations_output.csv'.")


