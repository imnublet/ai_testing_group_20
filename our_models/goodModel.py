from random import random

import pandas as pd
from numpy.random import randint
from sklearn.utils import resample

# about age: age can be between 18-67
data = pd.read_csv('../data/synth_data_for_training.csv')

feature_labels = pd.read_csv('../data/feature_labels_v2.csv')
# print(len(feature_labels))
discriminatory_features = feature_labels[feature_labels['Category'] == 'D']
# print(len(discriminatory_features))

subjective_features = feature_labels[feature_labels['Category'] == 'S']
# print(len(subjective_features))

non_relevant_features = feature_labels[feature_labels['Category'] == 'NR']
# print(len(non_relevant_features))

unclear_features = feature_labels[feature_labels['Category'] == 'U']
# print(len(unclear_features))


def change_labels_data_augmentation_binary(df, feature):  # flips value in feature column for data augmentation
    selection = resample(df, replace=True, n_samples=int(len(df)), random_state=42)
    selection[feature] = selection[feature].replace([0, 1], [1, 0])
    extra_labeled_data = pd.concat([df, selection])
    return extra_labeled_data


def drop_personality_columns(df):
    columns_personality = [f for f in df.columns if 'persoonlijke_eigenschappen' in f]
    df_dropped = df.drop(columns=columns_personality)
    return df_dropped

def data_augmentation_age(df):
    feature = 'persoon_leeftijd_bij_onderzoek'
    age_ranges = [(18, 27), (28, 37), (38, 47), (48, 57), (58,67)]
    selection = resample(df, replace=True, n_samples=int(len(df)), random_state=42)

    def replace_with_random_ages(age):
        new_ages = []
        for target_range in age_ranges:
            if not (target_range[0] <= age <= target_range[1]):
                new_ages.append(randint(target_range[0], target_range[1]))
        return new_ages

    new_rows = []
    for index, row in selection.iterrows():
        original_age = row[feature]
        new_ages = replace_with_random_ages(original_age)
        for new_age in new_ages:
            new_row = row.copy()
            new_row[feature] = new_age
            new_rows.append(new_row)
    extra_labeled_data = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return extra_labeled_data

def data_augmentation_neighborhoods(df):
    wijk_features = [f for f in df.columns if 'adres_recentste_wijk_' in f]
    selection = resample(df, replace=True, n_samples=int(len(df)), random_state=42)
    new_rows = []
    for index, row in selection.iterrows():
        original_wijk = [w for w in wijk_features if row[w] == 1]
        if(len(original_wijk)) > 0:
            original_wijk = original_wijk[0]
            other_wijken = [w for w in wijk_features if w != original_wijk]
            for wijk in other_wijken:
                new_row = row.copy()
                new_row.loc[original_wijk] = 0
                new_row.loc[wijk] = 1
                new_rows.append(new_row)
    extra_labeled_data = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return extra_labeled_data

def data_augmentation_taaleis(df):
    features_taal = ['contacten_onderwerp_boolean_taaleis___voldoet', 'contacten_onderwerp_boolean_beoordelen_taaleis', 'contacten_onderwerp_beoordelen_taaleis']
    data = df
    for feature in features_taal:
        data = change_labels_data_augmentation_binary(data, feature)
    return data

def drop_taaleis_columns(df):
    features_taal = ['contacten_onderwerp_boolean_taaleis___voldoet', 'contacten_onderwerp_boolean_beoordelen_taaleis', 'contacten_onderwerp_beoordelen_taaleis', 'afspraak_verzenden_beschikking_i_v_m__niet_voldoen_aan_wet_taaleis']
    df_dropped = df.drop(columns=features_taal)
    return df_dropped

# print(len(data))
# print("Data augmentation neighborhoods")
# new_data = data_augmentation_neighborhoods(data)
# print(len(new_data))
#
print(len(data))
print("Data augmentation Taaleis")
data = data_augmentation_taaleis(data)
print(len(data))

# print('Augmenting data for the ages')
# print(len(data))
#
# data = data_augmentation_age(data)
# print(len(data))
# extra_data_gender = change_labels_data_augmentation_binary(data, 'persoon_geslacht_vrouw')
# print(len(data))
# print(len(extra_data_gender))
