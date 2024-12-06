import pandas as pd 
from sklearn.utils import resample

# about age: age can be between 18-67
data = pd.read_csv('data/synth_data_for_training.csv')
#columns = data.columns
columns = data.filter(regex='adres_recent|persoon_geslacht|persoon_leeftijd').columns 
address_dict = {}
age_series = {}
for column in columns:
    #if column.contains("adres"):
    #    address_dict[column] = 
    #print(data[column].unique())
    if column.startswith('persoon_leeftijd'):
        age_series = data[column].value_counts()
    counts = data[column].value_counts()
    print(counts)

bins = [18, 27, 37, 47, 57, 67]
labels = ['18-27', '28-37', '38-47', '48-57', '58-67']

# Bin the data into groups
groups = pd.cut(age_series.index, bins=bins, labels=labels, right=True)
age_grouped = age_series.groupby(groups).sum()
print(age_grouped)
print(columns)

def oversample_age(df, sampling_factor, feature='persoon_leeftijd_bij_onderzoek', min_age=38, max_age=57):
    """Resample data of people falling within given age bracket."""
    majority_df = df[df[feature].between(min_age, max_age, inclusive='both')] # df containing people aged between min_age and max_age
    minority_df = df[df.isin(majority_df) == False].dropna()
    print(len(majority_df) + len(minority_df))

    # Upsample the majority class
    majority_upsampled = resample(majority_df, replace=True, n_samples=int(len(majority_df)*sampling_factor), random_state=42)
    print("length upsampled: ", len(majority_upsampled))

    # Combine the upsampled majority class with the minority class
    rebalanced_data = pd.concat([majority_upsampled, minority_df])
    print(len(rebalanced_data))
    return rebalanced_data

def oversample_gender(df, sampling_factor, feature='persoon_geslacht_vrouw', gender=0): 
    """Resample data of people with given gender. Default gender is male, male=0, female=1."""
    majority_df = df[df[feature] == gender] 
    minority_df = df[df[feature] != gender] 
    print(len(majority_df) + len(minority_df))

    # Upsample the majority class
    majority_upsampled = resample(majority_df, replace=True, n_samples=int(len(majority_df)*sampling_factor), random_state=42)
    print("length upsampled: ", len(majority_upsampled))

    # Combine the upsampled majority class with the minority class
    rebalanced_data = pd.concat([majority_upsampled, minority_df])
    print(len(rebalanced_data))  
    return rebalanced_data  

oversample_age(data, 1.25)
oversample_gender(data, 1.25)

address_columns = data.filter(regex='adres_recentste_wijk').columns 
print(address_columns)

def reweigh_address(df): # use like so: pipeline.fit(X, y, classification__sample_weight=sample_weights)
    addresses = df.filter(regex='adres_recentste_wijk').columns 
    address_weights = {}
    #dict_example['c'] = 3  # new key, add
    for address in addresses:
        proportion = len(df[df[address]== 1]) / len(df)
        address_weights[address] = proportion
    print(address_weights)

    values = address_weights.values()
    min_proportion = min(values)
    max_proportion = max(values)

    normalized_weights = {key: ((v - min_proportion) / (max_proportion - min_proportion) )  for (key, v) in address_weights.items() }
    print(normalized_weights)

    sample_weights = pd.Series(0, index=df.index, dtype=float)  
    for address in addresses:
        sample_weights[df[address] == 1] = normalized_weights[address]
    
    #print(sample_weights)
    return sample_weights

reweigh_address(data)
#print(len(data[data[address_columns[0]]== 1]))

def change_labels(df, percentage): # flips value in 'checked' column to create inconsistencies  
    selection = resample(df, replace=True, n_samples=int(len(df)*percentage), random_state=42)
    selection['checked'] = selection['checked'].replace([0,1], [1,0])
    extra_labeled_data = pd.concat([df, selection])
    return extra_labeled_data

change_labels(data, 0.1)
