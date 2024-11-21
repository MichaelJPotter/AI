import pandas as pd
from sklearn.metrics import f1_score, classification_report

#load dataset
file_path = 'fertility_Diagnosis.txt'
#create a dataFrame for the dataset
columns = [
    'Season', 'Age', 'Childish_disease', 'Serious_accident', 'Surgical_intervention', 'Fevers', 'Alcohol_consumption', 'Smoking_habit', 'Num_of_hours_sitting', 'Diagnosis'
]
#load dataset
data = pd.read_csv(file_path, header=None, names=columns)
#prints the first 5 rows from txt file
print("First five rows of dataset:")
print(data.head())

class FertilityInferenceEngine:
    def __init__(self, rules):
        self.rules = rules

    def apply_rules(self, person_data):
        scores = {'N': 0, 'O': 0}
        

        # Apply rules iteratively to deduce fertility status
        for rule in self.rules:
            if rule['condition'](person_data):  # If rule condition is met
                scores[rule['conclusion']] += rule['weight']

        return 'O' if scores['O'] > scores['N'] else 'N'  # Default to 'normal' if no rules are met

  #using rules rather than if statements for better code flexibility, using a rule based system is easier to understand then a block of IF's
rules = [
    #altered fertility due to 
    
    {'condition': lambda facts: facts['Season'] ==0.33, 'conclusion': 'O'}, # heat stress 
    {'condition': lambda facts: facts['Alcohol_consumption'] >=0.4, 'conclusion': 'O'},
    #lowering alcohol consumption rule from >=6 to >=4 increased the F1 score and increased the precision and recall for altered fertility diagnosis
    #{'condition': lambda facts: facts['Age'] >=0.75, 'conclusion': 'O'},
    {'condition': lambda facts: facts['Smoking_habit'] >=0, 'conclusion': 'O'},
    {'condition': lambda facts: facts['Serious_accident'] ==0, 'conclusion': 'O'},
    {'condition': lambda facts: facts['Num_of_hours_sitting'] >= 0.5, 'conclusion': 'O'},
    {'condition': lambda facts: facts['Serious_accident'] ==0, 'conclusion': 'O'},
    {'condition': lambda facts: facts['Smoking_habit'] >=0 and facts['Alcohol_consumption'] >=0.4, 'conclusion': 'O'},
    #Normal fertility due to
    {'condition': lambda facts: facts['Serious_accident'] ==0 and facts['Surgical_intervention'] ==0, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Season'] ==1, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Season'] ==-1, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Season'] ==-0.33, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Smoking_habit'] == -1 and facts['Alcohol_consumption'] <=0.3, 'conclusion': 'N' },
    {'condition': lambda facts: facts['Alcohol_consumption'] <=0.4, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Smoking_habit'] ==-1, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Num_of_hours_sitting'] <= 0.5, 'conclusion': 'N'}, #over 5 hours spent sitting a day can lead to altered fertility
    {'condition': lambda facts: facts['Surgical_intervention'] ==0, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Serious_accident'] ==1, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Age'] <=0.75, 'conclusion': 'N'}
   #{'condition': lambda facts: facts['']}
]

engine = FertilityInferenceEngine(rules)
#Apply inference engine to each row in the dataset
data['Predicted_Diagnosis'] = data.apply(lambda row: engine.apply_rules(row.to_dict()), axis=1)
#converts true labels and predictions to binary(1 Altered, 0 Normal)
y_true = data['Diagnosis'].apply(lambda x: 1 if x == 'O' else 0)
y_pred = data['Predicted_Diagnosis'].apply(lambda x: 1 if x == 'O' else 0)
#Calculate F1 score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.2f}")
#classification report for detailed metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Altered"]))

#high precision(1.00) but low recall for normal predictions meaning its getting them right but overly favouring altered
#add weighting to rules