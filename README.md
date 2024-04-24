# ChatForCure
“ChatForCure” is a chatbot which takes user input for the symptoms he is experiencing, asks follow- up questions to be sure of the prediction and shows the predicted disease(s), their description(s), specialist required, list of specialists( ranked according to their rating) and their ratings. It also allows users to rate a doctor. My future plans include separate logins for doctors and patients. For the patient side, I plan on preparing a better user interactive frontend, including a time field for doctors’ availability, location field to recommend the doctors closest to the patient, booking and cancellation facility and chat with doctor.  For the doctor’s side, I plan on giving them their appointments schedule and the chat history of patients who have booked their appointment, giving them the same advantage we engineers get by preprocessing. Further we can also build more on this by using text-to-speech converter and speech-to-text converter to make our solution more inclusive for the specially- abled.


Technologies Used:
  1. Programming Language: Python.
  2. Frameworks: Will use Flask for building the web application.
  3. Libraries: Scikit-learn for symptom analysis and recommendation generation
  4. Database: Currently using CSV files to store symptoms, description, precautions          and healthcare providers’ data.


Algorithm Approach: 
The algorithm employs a Decision Tree Classifier to predict diseases based on input symptoms. Decision trees are intuitive and easy to interpret, making them suitable for medical diagnosis tasks. The algorithm recursively splits the data based on feature values to create a tree-like structure, where each leaf node represents a disease. During training, the algorithm learns to make decisions about which symptoms are most informative for predicting diseases. It also uses a Support Vector Machine model for classification  based on input symptoms. It also calculates feature importances from the trained Decision Tree Classifier which indicate the relevance of each symptom in predicting diseases and helps identify the most significant symptoms for diagnosis, guiding the user interaction process. Before training, the algorithm preprocesses the data by encoding categorical variables (symptoms and diseases) into numerical format using Label Encoding. It performs cross-validation to evaluate the performance of the Decision Tree Classifier on unseen data by splitting the dataset into multiple train-test splits and averaging the evaluation metrics.


HOW TO RUN? 

1. install all requirements from requirements.txt with pip install -r requirements.txt
2. python chat_bot.py