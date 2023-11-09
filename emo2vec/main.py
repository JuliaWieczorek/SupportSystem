import json
import pickle

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors, Word2Vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

# File path to the JSON dataset
file_path = "ESConv.json"

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
print('mapped data: done')

mapped_emotions = [entry["Emotion Type"] for entry in mapped_data]
data_frame = pd.DataFrame({'Emotion Type': mapped_emotions})
emotion_counts = data_frame['Emotion Type'].value_counts()
print(emotion_counts)

def train_test(dialogues):
    # divide dialogues on train and test 80:20
    splitted_dialogues = {'train': [], 'test': []}

    for dialogue in dialogues:
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


    #print("Dane treningowe:")
    #print(splitted_dialogues['train'])
    #print("\nDane testowe:")
    #print(splitted_dialogues['test'])

splitted_dialogues = train_test(mapped_data)
print('train_test: done')

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
print('token: done')

def word2vec(dialog):
    word_vectors = KeyedVectors.load_word2vec_format('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz', binary=False, limit=50000)

    embedded_dialogs = {'train': [], 'test': []}

    for dialogue in dialog['train']:
        conversation_id = dialogue['Conversation ID']
        seeker_dialog = dialogue['Seeker Dialog']
        emotion_type = dialogue['Emotion Type']

        embeddings = {}
        for word in seeker_dialog:
            if word in word_vectors.wv:
                embeddings[word] = word_vectors.wv[word]

        embedded_dialogs['train'].append(
            {'Conversation ID': conversation_id, 'Seeker Dialog': embeddings, 'Emotion Type': emotion_type})

    for dialogue in dialog['test']:
        conversation_id = dialogue['Conversation ID']
        seeker_dialog = dialogue['Seeker Dialog']
        emotion_type = dialogue['Emotion Type']

        embeddings = {}
        for word in seeker_dialog:
            if word in word_vectors.wv:
                embeddings[word] = word_vectors.wv[word]

        embedded_dialogs['test'].append(
            {'Conversation ID': conversation_id, 'Seeker Dialog': embeddings, 'Emotion Type': emotion_type})

    return embedded_dialogs

w0rd2vec = word2vec(tokenized_dialogues)
print('word2vec: done')

balanced_data = mapped_data[mapped_data['Emotion Type'].isin([7, 10, 9, 8])]

print(balanced_data['Emotion Type'].value_counts())


X_train = [dialog['Seeker Dialog'] for dialog in w0rd2vec['train']]
y_train = [dialog['Emotion Type'] for dialog in splitted_dialogues['train']]
X_test = [dialog['Seeker Dialog'] for dialog in w0rd2vec['test']]
y_test = [dialog['Emotion Type'] for dialog in splitted_dialogues['test']]

print(X_train)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform([' '.join(dialog) for dialog in X_train])
X_test_vectorized = vectorizer.transform([' '.join(dialog) for dialog in X_test])

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_vectorized, y_train)

predicted_emotions = clf.predict(X_test_vectorized)
print(predicted_emotions)

# from sklearn.metrics import accuracy_score, classification_report
#
# accuracy = accuracy_score(y_test, predicted_emotions)
# class_report = classification_report(y_test, predicted_emotions, target_names=predicted_emotions)
#
# print(f'Accuracy: {accuracy * 100:.2f}%')
# print('Classification Report:')
# print(class_report)

# accuracy = accuracy_score(y_test, predicted_emotions)
# print(f'Accuracy: {accuracy * 100:.2f}%')
#
# original_emotions = [entry["Emotion Type"] for entry in mapped_data]
# possible_labels = ['anxiety', 'depression', 'sadness', 'anger', 'fear', 'shame', 'disgust', 'nervousness', 'pain', 'guilt', 'jealousy']
# class_report = classification_report(original_emotions, predicted_emotions, target_names=possible_labels)
#
# print(class_report)
#
# conf_matrix = confusion_matrix(y_test, predicted_emotions)
# print(conf_matrix)
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
#             xticklabels=['anxiety', 'depression', 'sadness', 'anger', 'fear', 'shame', 'disgust', 'nervousness', 'pain', 'guilt', 'jealousy'],
#             yticklabels=['anxiety', 'depression', 'sadness', 'anger', 'fear', 'shame', 'disgust', 'nervousness', 'pain', 'guilt', 'jealousy'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
