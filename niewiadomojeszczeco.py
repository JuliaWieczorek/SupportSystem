import json
import pickle

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize

import spacy
from collections import Counter


# File path to the JSON dataset
file_path = "../emo2vec/ESConv.json"

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
# print(len(list(dialog['Seeker Dialog'] for dialog in tokenized_dialogues['train'])))

# print(tokenized_dialogues['train'][0])

nlp = spacy.load("en_core_web_sm")

dialogs = []
first_person_counts = []
past_references_count = Counter()

for record in tokenized_dialogues['train']:
    dialog = record['Seeker Dialog']
    doc = nlp(dialog)

    first_person_count = sum(1 for word in dialog if word.lower() == 'i')

    past_references_count = sum(1 for token in doc if token.pos_ == 'VERB' and 'Tense=Past' in token.tag_)

    dialogs.append(dialog)
    first_person_counts.append(first_person_count)
    past_references_counts.append(past_references_count)

df = pd.DataFrame(
    {'Dialog': dialogs, 'First Person Count': first_person_counts, 'Past References Count': past_references_counts})

print(df)

#n_i = 0
#for dialog in tokenized_dialogues['train']:
#    for word in dialog['Seeker Dialog']:
#        if word == 'I' or word == 'i':
#            n_i =+ 1
#        else:
#            print(word)
#print(n_i)


