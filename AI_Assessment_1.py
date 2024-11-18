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
        facts = person_data.copy()

        # Apply rules iteratively to deduce fertility status
        for rule in self.rules:
            if rule['condition'](facts):  # If rule condition is met
                return rule['conclusion']

        return 'N'  # Default to 'normal' (N) if no rules are met

  #using rules rather than if statements for better code flexibility, using a rule based system is easier to understand then a block of IF's
rules = [
    #altered fertility due to 
    {
        'condition': lambda facts: facts['Alcohol_consumption'] <=0.6, 'conclusion': 'N'
    },
     {
        'condition': lambda facts: facts['Alcohol_consumption'] >=6, 'conclusion': 'O'
    },
     {
        'condition': lambda facts: facts['Smoking_habit'] >=0, 'conclusion': 'O'
    },
     {
        'condition': lambda facts: facts['Serious_accident'] ==0, 'conclusion': 'O'
    },
     {
        'condition': lambda facts: facts['Surgical_intervention'] ==0, 'conclusion': 'N'
    },
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

