import json
import pickle

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors, Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

'''This Python program performs emotion classification in conversations. 
It involves data loading and extraction, emotion mapping, dataset balancing, 
splitting into training and testing sets, tokenization, and feature engineering using CountVectorizer. 
The model is trained using a Random Forest Classifier, 
and its performance is evaluated with accuracy metrics, 
a classification report, and a confusion matrix visualized as a heatmap. 
The program utilizes libraries like pandas, numpy, nltk, gensim, scikit-learn, seaborn, 
and matplotlib for data manipulation, natural language processing, machine learning, and visualization.'''


''' Results:
Accuracy: 28.19%
Classification Report:
              precision    recall  f1-score   support

     anxiety       0.46      0.15      0.23       354
  depression       0.30      0.12      0.17       334
     sadness       0.26      0.87      0.40       308
       anger       1.00      0.01      0.02       111
        fear       0.00      0.00      0.00        95
       shame       0.00      0.00      0.00        42
     disgust       0.00      0.00      0.00        40

    accuracy                           0.28      1284
   macro avg       0.29      0.16      0.12      1284
weighted avg       0.35      0.28      0.20      1284

Confusion matrix:
[[ 53  37 264   0   0   0   0]
 [ 24  41 269   0   0   0   0]
 [ 14  27 267   0   0   0   0]
 [  8  14  88   1   0   0   0]
 [ 11   5  79   0   0   0   0]
 [  2   8  32   0   0   0   0]
 [  3   3  34   0   0   0   0]]

'''
# File path to the JSON dataset
file_path = "emo2vec/ESConv.json"

def load_data(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_data(dataset):
    # Function to extract conversation ID, seeker dialog, and emotion type

    extracted_data = []
    for idx, conversation in enumerate(dataset):
        conversation_id = idx + 1  # Conversation ID starts from 1
        emotion_type = conversation["emotion_type"]
        seeker_dialog = [
            message["content"] for message in conversation["dialog"] if message["speaker"] == "seeker"
        ]
        seeker_dialog_text = " ".join(seeker_dialog).strip()
        extracted_data.append({"Conversation ID": conversation_id, "Seeker Dialog": seeker_dialog_text, "Emotion Type": emotion_type})
    return extracted_data

dataset = load_data(file_path)

# Extract data
extracted_data = extract_data(dataset)


def map_emotion_to_numbers(data):
    # Define emotion type to number mapping
    emotion_mapping = {
        'anxiety': 0,
        'depression': 1,
        'sadness': 2,
        'anger': 3,
        'fear': 4,
        'shame': 5,
        'disgust': 6,
        'nervousness': 7,
        'pain': 8,
        'guilt': 9,
        'jealousy': 10
    }

    mapped_data = []
    for entry in data:
        emotion_type = entry["Emotion Type"]
        mapped_emotion = emotion_mapping.get(emotion_type)
        mapped_entry = entry.copy()
        mapped_entry["Emotion Type"] = mapped_emotion
        mapped_data.append(mapped_entry)

    return mapped_data

mapped_data = map_emotion_to_numbers(extracted_data)
mapped_data = pd.DataFrame(mapped_data)
balanced_data = mapped_data[~mapped_data['Emotion Type'].isin([7, 10, 9, 8])]
# print(balanced_data['Emotion Type'].value_counts())

def train_test(dialogues):
    # divide dialogues on train and test 80:20
    splitted_dialogues = {'train': [], 'test': []}

    for _, dialogue in dialogues.iterrows():
        conversation_id = dialogue['Conversation ID']
        seeker_dialog = dialogue['Seeker Dialog']
        emotion_type = dialogue['Emotion Type']

        # Obliczenie indeksu, który dzieli dane na 80:20
        split_index = int(0.8 * len(seeker_dialog))

        # Podział danych na dane treningowe (80%) i testowe (20%)
        train_dialog = seeker_dialog[:split_index]
        test_dialog = seeker_dialog[split_index:]
        train_emotion = emotion_type
        test_emotion = emotion_type

        splitted_dialogues['train'].append(
            {'Conversation ID': conversation_id, 'Seeker Dialog': train_dialog, 'Emotion Type': train_emotion})
        splitted_dialogues['test'].append(
            {'Conversation ID': conversation_id, 'Seeker Dialog': test_dialog, 'Emotion Type': test_emotion})
    return splitted_dialogues

splitted_dialogues = train_test(balanced_data)
print(splitted_dialogues)

# Check if conversation lengths are the same in the training and test sets
train_lengths = [len(dialog['Seeker Dialog']) for dialog in splitted_dialogues['train']]
test_lengths = [len(dialog['Seeker Dialog']) for dialog in splitted_dialogues['test']]

print(f"Average conversation length in the training set: {np.mean(train_lengths)}")
print(f"Average conversation length in the test set: {np.mean(test_lengths)}")

if train_lengths == test_lengths:
    print("Conversation lengths are the same.")
else:
    print("Conversation lengths are different.")

def token(dialog):
    tokenized_dialogues = {'train': [], 'test': []}

    for dialogue in dialog['train']:
        conversation_id = dialogue['Conversation ID']
        seeker_dialog = dialogue['Seeker Dialog']
        emotion_type = dialogue['Emotion Type']

        train_tokens = word_tokenize(seeker_dialog)

        tokenized_dialogues['train'].append(
            {'Conversation ID': conversation_id, 'Seeker Dialog': train_tokens, 'Emotion Type': emotion_type})

    for dialogue in dialog['test']:
        conversation_id = dialogue['Conversation ID']
        seeker_dialog = dialogue['Seeker Dialog']
        emotion_type = dialogue['Emotion Type']

        test_tokens = word_tokenize(seeker_dialog)

        tokenized_dialogues['test'].append(
            {'Conversation ID': conversation_id, 'Seeker Dialog': test_tokens, 'Emotion Type': emotion_type})

    return tokenized_dialogues

tokenized_dialogues = token(splitted_dialogues)
print(len(list(dialog['Seeker Dialog'] for dialog in tokenized_dialogues['train'])))

print(tokenized_dialogues)

X_train = [dialog['Seeker Dialog'] for dialog in tokenized_dialogues['train']]
y_train = [dialog['Emotion Type'] for dialog in tokenized_dialogues['train']]
X_test = [dialog['Seeker Dialog'] for dialog in tokenized_dialogues['test']]
y_test = [dialog['Emotion Type'] for dialog in tokenized_dialogues['test']]

print("Size of training set:", len(X_train))
print("Size of test set:", len(X_test))

print("Example of training data:")
print(X_train)

print("\nExample of training labels:")
print(y_train)

print("\nExample of test data:")
print(X_test)

print("\nExample of test labels:")
print(y_test)

vectorizer = CountVectorizer(analyzer=lambda x: x)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
print(X_test_vectorized)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_vectorized, y_train)

predicted_emotions = clf.predict(X_test_vectorized)
print(predicted_emotions)

accuracy = accuracy_score(y_test, predicted_emotions)
class_report = classification_report(y_test, predicted_emotions, target_names=['anxiety', 'depression', 'sadness', 'anger', 'fear', 'shame', 'disgust'])

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(class_report)

conf_matrix = confusion_matrix(y_test, predicted_emotions)
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=['anxiety', 'depression', 'sadness', 'anger', 'fear', 'shame', 'disgust'],
            yticklabels=['anxiety', 'depression', 'sadness', 'anger', 'fear', 'shame', 'disgust'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
