import pandas as pd
from sklearn.metrics import f1_score, classification_report

# Load dataset
file_path = 'fertility_Diagnosis.txt'
# Create a DataFrame for the dataset
columns = [
    'Season', 'Age', 'Childish_disease', 'Serious_accident', 'Surgical_intervention', 'Fevers',
    'Alcohol_consumption', 'Smoking_habit', 'Num_of_hours_sitting', 'Diagnosis'
]
# Load dataset
data = pd.read_csv(file_path, header=None, names=columns)

# Print the first 5 rows of the dataset
print("First five rows of dataset:")
print(data.head())

# Forward chaining engine
#ChatGPT: create a forward chaining AI system using python for vscode which will aim to predict whether a person will have normal or altered fertility from data provided by a dataset
#ChatGPT: it does not need machine learning however i must be able to analyse the F1 score of the algorithm

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


# Backward chaining engine
class BackwardChainingEngine:
    def __init__(self, rules):
        self.rules = rules

    def infer(self, goal, person_data):
        # Check if any rule concludes the goal
        for rule in self.rules:
            if rule['conclusion'] == goal and rule['condition'](person_data):
                return True  # The goal is satisfied by this rule
        return False  # No rules satisfied the goal

    def apply_backward_chaining(self, person_data):
        # Start with the goal of classifying as 'O' (altered fertility)
        if self.infer('O', person_data):
            return 'O'
        # If 'O' is not satisfied, classify as 'N' (normal fertility)
        return 'N'


# Define rules
rules = [
    # Altered fertility due to
    {'condition': lambda facts: facts['Season'] == 0.33, 'weight': 1, 'conclusion': 'O'},  # heat stress
    {'condition': lambda facts: facts['Alcohol_consumption'] >= 0.4, 'weight': 2, 'conclusion': 'O'},
    {'condition': lambda facts: facts['Smoking_habit'] >= 0, 'weight': 10, 'conclusion': 'O'},
    {'condition': lambda facts: facts['Serious_accident'] == 0, 'weight': 1, 'conclusion': 'O'},
    {'condition': lambda facts: facts['Num_of_hours_sitting'] >= 0.8, 'weight': 1, 'conclusion': 'O'},
    {'condition': lambda facts: facts['Smoking_habit'] >= 0 and facts['Alcohol_consumption'] >= 0.6, 'weight': 10, 'conclusion': 'O'},

    # Normal fertility due to
    {'condition': lambda facts: facts['Serious_accident'] == 0 and facts['Surgical_intervention'] == 0, 'weight': 3, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Season'] == 1, 'weight': 2, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Season'] == -1, 'weight': 2, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Smoking_habit'] == -1 and facts['Alcohol_consumption'] <= 0.3, 'weight': 3, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Alcohol_consumption'] <= 0.4, 'weight': 3, 'conclusion': 'N'},
    {'condition': lambda facts: facts['Smoking_habit'] == -1, 'weight': 3, 'conclusion': 'N'}
   # {#'condition': lambda facts: facts['Num_of_hours_sitting'] <= 0.7, 'weight': 2, 'conclusion': 'N'},
    #{'condition': lambda facts: facts['Surgical_intervention'] == 0, 'weight': 3, 'conclusion': 'N'},
   # {'condition': lambda facts: facts['Age'] <= 0.7, 'weight': 3, 'conclusion': 'N'}
]

# Create forward and backward inference engines
forward_engine = FertilityInferenceEngine(rules)
backward_engine = BackwardChainingEngine(rules)

# Apply forward chaining
data['Predicted_FC'] = data.apply(lambda row: forward_engine.apply_rules(row.to_dict()), axis=1)

# Apply backward chaining
data['Predicted_BC'] = data.apply(lambda row: backward_engine.apply_backward_chaining(row.to_dict()), axis=1)

# Convert true labels and predictions to binary (1 Altered, 0 Normal)
y_true = data['Diagnosis'].apply(lambda x: 1 if x == 'O' else 0)
y_pred_fc = data['Predicted_FC'].apply(lambda x: 1 if x == 'O' else 0)
y_pred_bc = data['Predicted_BC'].apply(lambda x: 1 if x == 'O' else 0)

# Evaluate forward chaining
print("\n--- Forward Chaining Classification Report ---")
print(classification_report(y_true, y_pred_fc, target_names=["Normal", "Altered"]))
print(f"Forward Chaining F1 Score: {f1_score(y_true, y_pred_fc):.2f}")

# Evaluate backward chaining
print("\n--- Backward Chaining Classification Report ---")
print(classification_report(y_true, y_pred_bc, target_names=["Normal", "Altered"]))
print(f"Backward Chaining F1 Score: {f1_score(y_true, y_pred_bc):.2f}")
