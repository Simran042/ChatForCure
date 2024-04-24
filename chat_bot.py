import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
from collections import defaultdict
'''
Here, we load the necessary libraries and modules, 
read the training and testing data, preprocess the data, 
and split it into training and testing sets. 
Then we train a Decision Tree classifier and an SVM model 
using the training data. Cross-validation is performed for 
the Decision Tree classifier to evaluate its performance. 
'''

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Read training and testing data
training_data = pd.read_csv('Data/Training.csv')
testing_data = pd.read_csv('Data/Testing.csv')

# Prepare data columns
feature_columns = training_data.columns[:-1]
features = training_data[feature_columns]
labels = training_data['prognosis']

# Group data by prognosis
reduced_data = training_data.groupby(training_data['prognosis']).max()

# Mapping strings to numbers for labels
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(labels)
labels_encoded = label_encoder.transform(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.20, random_state=42)
test_features = testing_data[feature_columns]
test_labels = testing_data['prognosis']
test_labels_encoded = label_encoder.transform(test_labels)

# Train Decision Tree classifier and perform cross-validation
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train.values, y_train.ravel()) 
cross_validation_scores = cross_val_score(decision_tree_classifier, X_test, y_test, cv=3)
print("Cross-validation scores mean:", cross_validation_scores.mean())

# Train SVM model
svm_model = SVC()
svm_model.fit(X_train.values, y_train.ravel())
print("SVM score:", svm_model.score(X_test.values, y_test))


'''
Now, we load symptom and doctor data from CSV files 
into dictionaries for later use in the chatbot. 
Various CSV files contain information about symptom severity, 
description, precaution, specialty, and doctor details.
'''

# Get feature importances from the Decision Tree classifier
importances = decision_tree_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
features = feature_columns

# Initialize dictionaries for symptom severity, description, precaution, and specialty
severity_dictionary = dict()
description_dictionary = dict()
precaution_dictionary = dict()
speciality_dictionary = dict()
doctors_dictionary = defaultdict(list)
symptoms_dictionary = {}

# Load symptom data from CSV files
def load_symptom_and_doctors_data():
    global severity_dictionary, description_dictionary, precaution_dictionary, speciality_dictionary, doctors_dictionary
    try:
        with open('MasterData/symptom_severity.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Populate severity dictionary
            for row in csv_reader:
                if len(row) >= 2:  # Ensure row has at least 2 columns
                    severity_dictionary[row[0]] = int(row[1])
                else:
                    print("Error: Invalid format in symptom_severity.csv")
    except FileNotFoundError:
        print("Error: symptom_severity.csv not found")

    # Load symptom description data
    try:
        with open('MasterData/symptom_Description.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 2:  # Ensure row has at least 2 columns
                    description_dictionary[row[0]] = row[1]
                else:
                    print("Error: Invalid format in symptom_Description.csv")
    except FileNotFoundError:
        print("Error: symptom_Description.csv not found")

    # Load precaution data for symptoms
    try:
        with open('MasterData/symptom_precaution.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 5:  # Ensure row has at least 5 columns
                    precaution_dictionary[row[0]] = [row[1], row[2], row[3], row[4]]
                else:
                    print("Error: Invalid format in symptom_precaution.csv")
    except FileNotFoundError:
        print("Error: symptom_precaution.csv not found")

    # Load specialty data for symptoms
    try:
        with open('MasterData/symptom_Speciality.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 2:  # Ensure row has at least 2 columns
                    speciality_dictionary[row[0]] = row[1]
                else:
                    print("Error: Invalid format in symptom_Speciality.csv")
    except FileNotFoundError:
        print("Error: symptom_Speciality.csv not found")

    # Load doctors' data based on specialty
    try:
        with open('MasterData/speciality_Doctor.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 4:  # Ensure row has at least 2 columns
                    specialty = row[0]
                    doctor = row[1]
                    try:
                        rating = float(row[2])
                        num = int(row[3])
                        date= row[4]
                        time= row[5]
                        doctors_dictionary[specialty].append((doctor, rating, num, date, time))
                    except ValueError:
                        print(f"Error: Invalid rating format for doctor {doctor} in specialty {specialty}")
    except FileNotFoundError:
        print("Error: speciality_Doctor.csv not found")

'''
In this section, we initialize the text-to-speech 
engine and define a function to calculate the severity 
of a condition based on symptoms and duration.
'''

# Initialize pyttsx3 engine for text-to-speech
def initialize_engine():
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    return engine

# Function to calculate condition severity based on symptoms and duration
def calculate_condition_severity(symptoms, days):
    sum_severity = 0
    for symptom in symptoms:
        sum_severity += severity_dictionary[symptom]
    if (sum_severity * days) / (len(symptoms) + 1) > 13:
        print("\nYou should take consultation from a doctor.\n")
    else:
        print("\nIt might not be that bad, but you should take precautions.\n")

'''
This function checks for patterns in the input text 
to match with symptoms in the symptoms list.
'''

# Function to check pattern in input
def check_input_pattern(symptoms_list, input_text):
    input_text = input_text.lower()
    predicted_list = []
    input_text = input_text.replace(' ', '_')
    pattern = f"{input_text}"
    regexp = re.compile(pattern)
    predicted_list = [item for item in symptoms_list if regexp.search(item)]
    if len(predicted_list) == 1:  # Only one match found
        return True, predicted_list
    elif len(predicted_list) > 1:  # Multiple matches found
        return True, predicted_list
    else:  # No matches found
        return False, []

'''
This function predicts diseases from symptoms using a 
secondary model, in our case, a Decision Tree classifier.
'''

# Function to predict disease from symptoms
def secondary_predict(symptoms_list):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=20)
    random_forest_classifier = DecisionTreeClassifier()
    random_forest_classifier.fit(X_train.values, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_list:
        input_vector[[symptoms_dict[item]]] = 1

    return random_forest_classifier.predict([input_vector])

'''
This function prints details of the disease predicted.
'''

# Function to print disease details
def print_disease_details(node):
    node = node[0]
    val = node.nonzero()
    disease = label_encoder.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


'''
This function converts the decision tree to code and 
recursively traverses the tree to predict diseases 
based on symptoms provided by the user.
'''

# Function to convert decision tree to code
def decision_tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    check_disease = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        disease_input = input("\nEnter the symptom you are experiencing: ")
        confidence, confidence_disease = check_input_pattern(check_disease, disease_input)
        if confidence:
            confidence_input = 0
            if len(confidence_disease) > 1: 
                print("\nHere are the searches related to your symptom in our database:")
                for index, item in enumerate(confidence_disease):
                    print(index, ") ", item)
                confidence_input = int(input(f"Select the one you meant (0 - {len(confidence_disease) - 1}): "))
            else: print("\nHere is the input symptom we recorded: ", confidence_disease[0])
            disease_input = confidence_disease[confidence_input]
            break
        else:
            print("\n\n Please enter a valid symptom. Those present in our database are: ")
            training_data = pd.read_csv('Data/Training.csv')

            # Print all the symptoms
            for symptom in training_data.columns[:-1]:  # Excluding the last column which is the target variable
                print(symptom)

    while True:
        try:
            num_days = int(input("\nFrom how many days are you experiencing this symptom? "))
            break
        except:
            print("Enter a valid input.")

    '''
    This function converts the decision tree to code 
    and recursively traverses the tree to predict diseases 
    based on symptoms provided by the user.
    '''
    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease_details(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("\n\nAre you experiencing any of the following symptoms?")
            symptoms_exp = [disease_input]
            for syms in list(symptoms_given):
                inp = input(f"{syms}? (yes/no): ")
                while inp.lower() not in ['yes', 'no']:
                    inp = input("Please provide a valid answer (yes/no): ")
                if inp.lower() == 'yes':
                    symptoms_exp.append(syms)
            second_prediction = secondary_predict(symptoms_exp)
            calculate_condition_severity(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                print("\nYou may have", present_disease[0], "\n")
                print("Present disease:", present_disease)
                if present_disease[0] in description_dictionary:
                    print("Description:", description_dictionary[present_disease[0]])
                else:
                    print("Description not found for present disease:", present_disease[0])

            else:
                print("\nYou may have", present_disease[0], "or", second_prediction[0], "\n")
                print(description_dictionary[present_disease[0]])
                print("\n")
                print(description_dictionary[second_prediction[0]])

            if present_disease[0] == second_prediction[0]:
                print("\n\nYou may want a doctor working as ", speciality_dictionary[present_disease[0]])
            else:
                print("\n\nYou may want a doctor working as ", speciality_dictionary[present_disease[0]], "or",
                      speciality_dictionary[second_prediction[0]])

            if present_disease[0] == second_prediction[0]:
                print("\n\nHere are the recommended", speciality_dictionary[present_disease[0]], "doctors:")
                sorted_doctors = sorted(doctors_dictionary[speciality_dictionary[present_disease[0]]], key=lambda x: x[1],
                                         reverse=True)
                c = 0
                for doctor, rating, num, date, time in sorted_doctors:
                    c += 1
                    rating= round(rating, 2)
                    print(f"{c} \t {doctor} \t {rating} \t{date} \t {time}")
                
                choice = input("\n\nWould you like to rate any of the above doctors? (y/n): ")
                while choice.lower() not in ['y', 'n']:
                    print("Please enter 'y' for yes or 'n' for no.")
                    choice = input("Would you like to rate any of the above doctors? (y/n): ")
                
                if choice.lower() == 'y':
                    doctor_index = int(input("Enter the number corresponding to the doctor you want to rate: ")) - 1
                    new_rating = float(input("Enter your rating for the doctor (0-5): "))
                    num = sorted_doctors[doctor_index][2]
                    rating= sorted_doctors[doctor_index][1]
                    new_rating = (num * rating + new_rating) / (num + 1)
                    new_rating = round(new_rating, 2)
                    while(new_rating>5 or new_rating<0):
                        new_rating = float(input(print("Invalid rating. Please enter a number between 0 and 5.")))
                        new_rating = ((num * rating) + new_rating) / (num + 1)
                        new_rating= round(new_rating, 2)
                    if 0 <= new_rating <= 5:
                        sorted_doctors[doctor_index] = (sorted_doctors[doctor_index][0], new_rating, sorted_doctors[doctor_index][2]+1, sorted_doctors[doctor_index][3], sorted_doctors[doctor_index][4])
                        doctors_dictionary[speciality_dictionary[present_disease[0]]] = sorted_doctors
                        # Update CSV file with new ratings
                        with open('MasterData/speciality_Doctor.csv', mode='w', newline='') as file:
                            csv_writer = csv.writer(file)
                            for specialty, doctors in doctors_dictionary.items():
                                for doctor, rating, num, date, time in doctors:
                                    csv_writer.writerow([specialty, doctor, rating, num, date, time])
                        print("\nDoctor rating updated successfully!\n")
                    else:
                        print("Invalid rating. Please enter a number between 0 and 5.")
            else:
                print("\n\nHere are the recommended", speciality_dictionary[present_disease[0]], "doctors:")
                sorted_present_doctors = sorted(doctors_dictionary[speciality_dictionary[present_disease[0]]],
                                                key=lambda x: x[1], reverse=True)
                c = 0
                for doctor, rating, num, date, time in sorted_present_doctors:
                    c += 1
                    rating= round(rating, 2)
                    print(f"{c} \t {doctor} \t {rating} \t{date} \t {time}")
                
                choice = input("\n\nWould you like to rate any of the above doctors? (y/n): ")
                while choice.lower() not in ['y', 'n']:
                    print("Please enter 'y' for yes or 'n' for no.")
                    choice = input("Would you like to rate any of the above doctors? (y/n): ")
                if choice.lower() == 'y':
                    doctor_index = int(input("Enter the number corresponding to the doctor you want to rate: ")) - 1
                    new_rating = float(input("Enter your rating for the doctor (0-5): "))
                    num = sorted_present_doctors[doctor_index][2]
                    rating= sorted_present_doctors[doctor_index][1]
                    new_rating = (num * rating + new_rating) / (num + 1)
                    new_rating= round(new_rating, 2)
                    while(new_rating>5 or new_rating<0):
                        new_rating = float(input(print("Invalid rating. Please enter a number between 0 and 5.")))
                        new_rating = ((num * rating) + new_rating) / (num + 1)
                        new_rating= round(new_rating, 2)
                    if 0 <= new_rating <= 5:
                        sorted_present_doctors[doctor_index] = (sorted_present_doctors[doctor_index][0], new_rating, sorted_present_doctors[doctor_index][2]+1, sorted_present_doctors[doctor_index][3], sorted_present_doctors[doctor_index][4])
                        doctors_dictionary[speciality_dictionary[present_disease[0]]] = sorted_present_doctors
                        # Update CSV file with new ratings
                        with open('MasterData/speciality_Doctor.csv', mode='w', newline='') as file:
                            csv_writer = csv.writer(file)
                            for specialty, doctors in doctors_dictionary.items():
                                for doctor, rating, num, date, time in doctors:
                                    csv_writer.writerow([specialty, doctor, rating, num, date, time])
                        print("\nDoctor rating updated successfully!")
                    else:
                        print("Invalid rating. Please enter a number between 0 and 5.")

                print("\nHere are the recommended", speciality_dictionary[second_prediction[0]], "doctors:")
                sorted_second_doctors = sorted(doctors_dictionary[speciality_dictionary[second_prediction[0]]],
                                                key=lambda x: x[1], reverse=True)
                c = 0
                for doctor, rating, num, date, time in sorted_second_doctors:
                    c += 1
                    rating= round(rating, 2)
                    print(f"{c} \t {doctor} \t {rating} \t{date} \t {time}")
                choice = input("\n\nWould you like to rate any of the above doctors? (y/n): ")
                while choice.lower() not in ['y', 'n']:
                    print("Please enter 'y' for yes or 'n' for no.")
                    choice = input("Would you like to rate any of the above doctors? (y/n): ")
                if choice.lower() == 'y':
                    doctor_index = int(input("Enter the number corresponding to the doctor you want to rate: ")) - 1
                    new_rating = float(input("Enter your rating for the doctor (0-5): "))
                    num = sorted_second_doctors[doctor_index][2]
                    rating= sorted_second_doctors[doctor_index][1]
                    new_rating = ((num * rating) + new_rating) / (num + 1)
                    new_rating= round(new_rating, 2)
                    while(new_rating>5 or new_rating<0):
                        new_rating = float(input(print("Invalid rating. Please enter a number between 0 and 5.")))
                        new_rating = ((num * rating) + new_rating) / (num + 1)
                        new_rating= round(new_rating, 2)
                    if 0 <= new_rating <= 5:
                        sorted_second_doctors[doctor_index] = (sorted_second_doctors[doctor_index][0], new_rating, sorted_second_doctors[doctor_index][2]+1, sorted_second_doctors[doctor_index][3], sorted_second_doctors[doctor_index][4])
                        doctors_dictionary[speciality_dictionary[present_disease[0]]] = sorted_second_doctors
                        # Update CSV file with new ratings
                        with open('MasterData/speciality_Doctor.csv', mode='w', newline='') as file:
                            csv_writer = csv.writer(file)
                            for specialty, doctors in doctors_dictionary.items():
                                for doctor, rating, num, date, time in doctors:
                                    csv_writer.writerow([specialty, doctor, rating, num, date, time])
                        print("\nDoctor rating updated successfully!")
                    else:
                        print("Invalid rating. Please enter a number between 0 and 5.")

                    
            precaution_list = precaution_dictionary[present_disease[0]]
            print("\n\nTake the following measures:")
            for i, j in enumerate(precaution_list):
                print(i + 1, ")", j)

    recurse(0, 1)

'''
Finally, the initialize() function is called to 
start the execution of the program. It initializes the chatbot, 
loads symptom and doctor data, prompts the user for their name, 
and then begins the conversation with the user to predict diseases 
based on symptoms.
'''
# Main function to initialize and execute the program
def initialize():
    load_symptom_and_doctors_data()
    print("-----------------------------------HealthCare ChatBot-----------------------------------\n")
    name = input("\nWhat is your name? ")
    print("Hello,", name)
    engine = initialize_engine()
    decision_tree_to_code(decision_tree_classifier, feature_columns)
    print("\n\n----------------------------------------------------------------------------------------")

if __name__ == "__main__":
    initialize()
