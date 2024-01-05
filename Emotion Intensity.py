import os
import json
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

#Preprocessing
def load_data(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_confusion_matrix_to_file(cm, classifier_name):
    if not os.path.exists('confusion_matrix'):
        os.makedirs('confusion_matrix')

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f"Confusion Matrix for {classifier_name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig(f"confusion_matrix/confusion_matrix_{current_time}_{classifier_name}.png")
    plt.close()

dataset = load_data("ESConv.json")
data = pd.DataFrame(dataset)

X = []
y = []

for _, conversation in data.iterrows():
    seeker_initial = int(conversation['survey_score']['seeker'].get('initial_emotion_intensity', 0))
    empathy = int(conversation['survey_score']['seeker'].get('empathy', 0))
    relevance = int(conversation['survey_score']['seeker'].get('relevance', 0))
    final_emotion_intensity = int(conversation['survey_score']['seeker'].get('final_emotion_intensity', 0))

    X.append([seeker_initial, empathy, relevance])
    y.append(final_emotion_intensity)

df = pd.DataFrame(X, columns=['seeker_initial_intensity', 'empathy', 'relevance'])
df['final_emotion_intensity'] = y

print(df.head())

# Processing
X_train, X_test, y_train, y_test = train_test_split(df[['seeker_initial_intensity', 'empathy', 'relevance']], df['final_emotion_intensity'], test_size=0.2, random_state=42)

classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': MultinomialNB(),
    'Gradient Boosting': GradientBoostingClassifier()
}

emotion_labels = df['final_emotion_intensity'].unique()

for name, model in classifiers.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {accuracy*100:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix_to_file(cm, name)
