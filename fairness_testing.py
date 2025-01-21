import pandas as pd 
import onnxruntime as ort
from sklearn.metrics import accuracy_score
import numpy as np
from statistics import mean

# relevant features
# relatie_kind_heeft_kinderen
# relatie_partner_huidige_partner___partner__gehuwd
# persoon_geslacht_vrouw

# compare: single mothers, single fathers, married mothers, single women without children
data = pd.read_csv('data/investigation_train_large_checked.csv')

MODEL_PATH = 'For Group 20/model_1/model_1.onnx'
# Load the ONNX model
session = ort.InferenceSession(MODEL_PATH)


# Load model 2
MODEL_PATH_2 = 'For Group 20/model_2/model_2.onnx'
# Load the ONNX model
session_2 = ort.InferenceSession(MODEL_PATH_2)


def filter_data(data, feature_parent, feature_gender, feature_married):
    single_mothers = data[(data[feature_parent] == 1) & (data[feature_gender] == 1) & (data[feature_married] == 0)].drop(columns=['Ja', 'Nee', 'checked'])
    single_fathers = data[(data[feature_parent] == 1) & (data[feature_gender] == 0) & (data[feature_married] == 0)].drop(columns=['Ja', 'Nee', 'checked'])
    married_mothers = data[(data[feature_parent] == 1) & (data[feature_gender] == 1) & (data[feature_married] == 1)].drop(columns=['Ja', 'Nee', 'checked'])
    single_childless_women = data[(data[feature_parent] == 0) & (data[feature_gender] == 1) & (data[feature_married] == 0)].drop(columns=['Ja', 'Nee', 'checked'])
    return single_mothers, single_fathers, married_mothers, single_childless_women

def calculate_percentages(data, groups):
    total = len(data)
    for group_name, group_data in groups.items():
        print(f"Percentage of {group_name}: {len(group_data)/total}")

def calculate_risk_scores(model, groups):
    risk_scores = {}
    for group_name, group_data in groups.items():
        risk_scores[group_name] = model.run(None, {'X': group_data.values.astype(np.float32)})
    return risk_scores

def calculate_mean_risk_scores(risk_scores):
    mean_scores = {}
    for group_name, scores in risk_scores.items():
        mean_scores[group_name] = mean([score[1] for score in scores[1]])
        print(f"Mean risk score for {group_name}: {mean_scores[group_name]}")
    return mean_scores

def calculate_percentage_checked(risk_scores, groups):
    for group_name, scores in risk_scores.items():
        percentage_checked = sum([score[1] > 0.5 for score in scores[1]]) / len(groups[group_name])
        print(f"Percentage checked for {group_name}: {percentage_checked}")

def change_single_mothers(df_single_mothers, feature_parent, feature_gender, feature_married):
    mothers_to_fathers = df_single_mothers
    single_to_married = df_single_mothers
    kill_children = df_single_mothers
    #mothers_to_fathers[feature_gender] =  mothers_to_fathers[feature_gender].replace([1], [0]) # 1 to 0
    mothers_to_fathers.loc[mothers_to_fathers[feature_gender] == 1, feature_gender] = 0
    #single_to_married[feature_married] = single_to_married[feature_married].replace([0], [1]) # 0 to 1
    single_to_married.loc[mothers_to_fathers[feature_married] == 0, feature_married] = 1
    #kill_children[feature_parent] = kill_children[feature_parent].replace([1], [0]) # 1 to 0 
    kill_children.loc[mothers_to_fathers[feature_parent] == 1, feature_parent] = 0
    return mothers_to_fathers, single_to_married, kill_children

def calculate_change_mean_score(score_dict, single_mother_score):
    score_diff_dict = {}
    for group, score in score_dict.items():
        #print("score: ", score)
        difference = (single_mother_score - score) / ((score + single_mother_score) / 2) * 100
        score_diff_dict[group] = difference
        print(f"Difference in mean risk score for {group}: {difference}%")
    return score_diff_dict

def compare_risk_score(data, model_1, model_2):
    feature_parent = 'relatie_kind_heeft_kinderen' # 0 is no children
    feature_gender = 'persoon_geslacht_vrouw' # 0 is man
    feature_married = 'relatie_partner_huidige_partner___partner__gehuwd_' # 0 is single 

    # Filter data
    groups = {
        "single mothers": filter_data(data, feature_parent, feature_gender, feature_married)[0],
        "single fathers": filter_data(data, feature_parent, feature_gender, feature_married)[1],
        "married mothers": filter_data(data, feature_parent, feature_gender, feature_married)[2],
        "single childless women": filter_data(data, feature_parent, feature_gender, feature_married)[3],
    }

    # Calculate and print percentages
    calculate_percentages(data, groups)

    # Calculate risk scores for both models
    print("Model 1 - Risk Scores:")
    model1_risk_scores = calculate_risk_scores(model_1, groups)
    print("Model 2 - Risk Scores:")
    model2_risk_scores = calculate_risk_scores(model_2, groups)

    # Calculate and print mean risk scores
    print("\nModel 1 - Mean Risk Scores:")
    model1_mean_scores = calculate_mean_risk_scores(model1_risk_scores)
    print("\nModel 2 - Mean Risk Scores:")
    model2_mean_scores = calculate_mean_risk_scores(model2_risk_scores)

    # Calculate and print percentage checked
    print("\nModel 1 - Percentage Checked:")
    calculate_percentage_checked(model1_risk_scores, groups)
    print("\nModel 2 - Percentage Checked:")
    calculate_percentage_checked(model2_risk_scores, groups)

    mothers_to_fathers, single_to_married, remove_children = change_single_mothers(groups["single mothers"], feature_parent, feature_gender, feature_married)
    modified_groups = { # single childless mother instances changed to no longer be single, childless or female
        "mothers to fathers": mothers_to_fathers,
        "single to married": single_to_married,
        "mother to childless": remove_children
    }
    
    print("\nModel 1 - Mean Risk Scores of Modified Groups:")
    model1_risk_modified = calculate_mean_risk_scores(calculate_risk_scores(model_1, modified_groups))
    
    print("\nModel 2 - Mean Risk Scores of Modified Groups:")
    model2_risk_modified = calculate_mean_risk_scores(calculate_risk_scores(model_2, modified_groups))

    print("\nModel 1 - Difference in Mean Risk Scores of Modified Groups:")
    calculate_change_mean_score(model1_risk_modified, model1_mean_scores["single mothers"])

    print("\nModel 2 - Difference in Mean Risk Scores of Modified Groups:")
    calculate_change_mean_score(model2_risk_modified, model2_mean_scores["single mothers"])
    


    

    


def oversample_gender(df, sampling_factor, feature='persoon_geslacht_vrouw', gender=0): 
    """Resample data of people with given gender. Default gender is male, male=0, female=1."""
    majority_df = df[df[feature] == gender] 
    minority_df = df[df[feature] != gender] 

compare_risk_score(data, session, session_2)

# flip labels to the four groups and compare
#def flip_label(data):
