import sklearn
import json
import os
import numpy as np
import seaborn as sns
import seaborn as sns

import sklearn.metrics

survey = "Dwayne"
iteration = "7"

# Load the json files
with open('SIREN_test_sets/statistics/{0}{1}_confusion_matrix.json'.format(survey, iteration)) as f:
    confusion_matrix = json.load(f)

print(confusion_matrix)

# Define the order of your classes
SIREN_classes = ["Ia", "Ic", "Ib", "II"]
survey_classes =["Ia", "Ic", "Ib", "II"]

# Create the confusion matrix as a 2D array
confusion_matrix_array = []

for survey_label in survey_classes:
    row = []
    for SIREN_label in SIREN_classes:
        row.append(confusion_matrix[survey_label][SIREN_label])
    confusion_matrix_array.append(row)

# Convert to numpy array (optional, but useful for some sklearn functions)
confusion_matrix_array = np.array(confusion_matrix_array)

import matplotlib.pyplot as plt

# Plot the confusion matrix with swapped x and y data points
plt.figure(figsize=(8, 6), dpi=500)
sns.heatmap(confusion_matrix_array, annot=True, fmt=".2f", cmap='Blues', xticklabels=survey_classes, yticklabels=SIREN_classes)
plt.xlabel('{} 3 Classifications'.format(survey))
plt.ylabel('SIREN Classification')
plt.title('{} Confusion Matrix'.format(survey))
plt.savefig('SIREN_test_sets/statistics/{0}_{1}_confusion_matrix.png'.format(survey, iteration))
