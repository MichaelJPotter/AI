import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class FertilityInferenceEngine:
    def_init_(self, rules):
    self.rules = rules #list of inference rules

    def apply_rules(self, person_data):
         