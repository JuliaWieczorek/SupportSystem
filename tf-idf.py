import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Define constants
EMOTION_MAPPING = {
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
EXCLUDED_EMOTIONS = [7, 10, 9, 8]
NRC_LEXICON_FOLDER = "NRC_Emotion_Lexicon/NRC_Emotion_Lexicon/OneFilePerEmotion"

def load_data(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_data(dataset):
    extracted_data = []
    for idx, conversation in enumerate(dataset):
        conversation_id = idx + 1
        emotion_type = conversation["emotion_type"]
        seeker_dialog = [
            message["content"] for message in conversation["dialog"] if message["speaker"] == "seeker"
        ]
        seeker_dialog_text = " ".join(seeker_dialog).strip()
        extracted_data.append({"Conversation ID": conversation_id, "Seeker Dialog": seeker_dialog_text, "Emotion Type": emotion_type})
    return extracted_data

def map_emotion_to_numbers(data):
    mapped_data = []
    for entry in data:
        emotion_type = entry["Emotion Type"]
        mapped_emotion = EMOTION_MAPPING.get(emotion_type)
        mapped_entry = entry.copy()
        mapped_entry["Emotion Type"] = mapped_emotion
        mapped_data.append(mapped_entry)
    return mapped_data

def train_test_split_dialogues(dialogues):
    splitted_dialogues = {'train': [], 'test': []}
    for _, dialogue in dialogues.iterrows():
        conversation_id = dialogue['Conversation ID']
        seeker_dialog = dialogue['Seeker Dialog']
        emotion_type = dialogue['Emotion Type']
        split_index = int(0.8 * len(seeker_dialog))
        train_dialog = seeker_dialog[:split_index]
        test_dialog = seeker_dialog[split_index:]
        train_emotion = emotion_type
        test_emotion = emotion_type
        splitted_dialogues['train'].append(
            {'Conversation ID': conversation_id, 'Seeker Dialog': train_dialog, 'Emotion Type': train_emotion})
        splitted_dialogues['test'].append(
            {'Conversation ID': conversation_id, 'Seeker Dialog': test_dialog, 'Emotion Type': test_emotion})
    return splitted_dialogues

def tokenize_dialogues(dialogues):
    tokenized_dialogues = {'train': [], 'test': []}
    for dialogue in dialogues['train']:
        conversation_id = dialogue['Conversation ID']
        seeker_dialog = dialogue['Seeker Dialog']
        emotion_type = dialogue['Emotion Type']
        train_tokens = word_tokenize(seeker_dialog)
        tokenized_dialogues['train'].append(
            {'Conversation ID': conversation_id, 'Seeker Dialog': train_tokens, 'Emotion Type': emotion_type})
    for dialogue in dialogues['test']:
        conversation_id = dialogue['Conversation ID']
        seeker_dialog = dialogue['Seeker Dialog']
        emotion_type = dialogue['Emotion Type']
        test_tokens = word_tokenize(seeker_dialog)
        tokenized_dialogues['test'].append(
            {'Conversation ID': conversation_id, 'Seeker Dialog': test_tokens, 'Emotion Type': emotion_type})
    return tokenized_dialogues

def self_reference_count(dialog):
    return dialog.count('i')

def focus_past_count(dialog):
    past_verbs = ['was', 'were', 'had', 'did', 'went', 'said', 'made', 'came', 'took', 'saw', 'knew', 'got', 'thought', 'found', 'told', 'asked',
                  'worked', 'called', 'tried', 'used', 'wrote', 'read', 'played', 'bought', 'ate', 'drank', 'listened',
                  'watched', 'left', 'met', 'began', 'finished', 'learned', 'helped', 'gave', 'visited', 'lived',
                  'studied', 'loved', 'hated', 'missed', 'spoke', 'believed', 'forgot', 'remembered', 'opened',
                  'closed', 'sang', 'danced']
    past_expressions = ['yesterday', 'last week', 'ago', 'when I was a child', 'in the past',
                        'a long time ago', 'back then', 'in the old days', 'in my youth']
    past_words = past_verbs + past_expressions
    return sum(dialog.count(word) for word in past_words)

def load_nrc_lexicon(folder_path):
    lexicon = {}
    for filename in os.listdir(folder_path):
        emotion = os.path.splitext(filename)[0]
        with open(os.path.join(folder_path, filename), 'r') as file:
            words = [line.strip() for line in file.readlines()]
            lexicon[emotion] = words
    return lexicon

def nrc_emotions_count(dialog, nrc_lexicon):
    emotion_counts = {emotion: 0 for emotion in nrc_lexicon.keys()}
    for word in dialog:
        for emotion, emotion_words in nrc_lexicon.items():
            if word in emotion_words:
                emotion_counts[emotion] += 1
    return emotion_counts

def main():
    # Load data
    dataset = load_data("ESConv.json")
    print('dataset')

    # Extract and map emotion to numbers
    extracted_data = extract_data(dataset)
    mapped_data = map_emotion_to_numbers(extracted_data)
    print('mapped data')

    # Create DataFrame and filter out excluded emotions
    mapped_data_df = pd.DataFrame(mapped_data)
    balanced_data = mapped_data_df[~mapped_data_df['Emotion Type'].isin(EXCLUDED_EMOTIONS)]
    print('balanced data')

    # Train-test split
    splitted_dialogues = train_test_split_dialogues(balanced_data)
    print('train-test')

    # Tokenize dialogues
    tokenized_dialogues = tokenize_dialogues(splitted_dialogues)
    print('tokenize')

    # Load NRC Lexicon
    nrc_lexicon = load_nrc_lexicon(NRC_LEXICON_FOLDER)
    print('load NRC lexicon')

    # Features for train
    X_train_self_reference = [self_reference_count(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['train']]
    print('X_train_self_reference')
    X_train_focus_past = [focus_past_count(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['train']]
    print('X_train_focus_past')
    X_train_nrc_emotions = [nrc_emotions_count(dialog['Seeker Dialog'], nrc_lexicon) for dialog in tokenized_dialogues['train']]
    print('X_train_nrc_emotions')

    # Features for test
    X_test_self_reference = [self_reference_count(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['test']]
    print('X_test_self_reference')
    X_test_focus_past = [focus_past_count(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['test']]
    print('X_test_focus_past')
    X_test_nrc_emotions = [nrc_emotions_count(dialog['Seeker Dialog'], nrc_lexicon) for dialog in tokenized_dialogues['test']]
    print('X_test_nrc_emotions')

    # Features arrays
    X_train_features = np.array([X_train_self_reference, X_train_focus_past, X_train_nrc_emotions]).T
    X_test_features = np.array([X_test_self_reference, X_test_focus_past, X_test_nrc_emotions]).T
    print('features')

    # Combine features with NRC Lexicon features
    for emotion in nrc_lexicon.keys():
        X_train_nrc_emotion = [dialog[emotion] for dialog in X_train_nrc_emotions]
        X_test_nrc_emotion = [dialog[emotion] for dialog in X_test_nrc_emotions]
        X_train_features = np.column_stack((X_train_features, X_train_nrc_emotion))
        X_test_features = np.column_stack((X_test_features, X_test_nrc_emotion))

    # Model TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform([" ".join(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['train']])
    X_test_tfidf = vectorizer.transform([" ".join(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['test']])
    print('mode tf-idf')

    # Combine features with TF-IDF
    X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_features))
    X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_features))
    print('combined')

    # Classification model - RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train_combined, [dialog['Emotion Type'] for dialog in tokenized_dialogues['train']])
    print('classification model')

    y_pred = model.predict(X_test_combined)
    accuracy = accuracy_score([dialog['Emotion Type'] for dialog in tokenized_dialogues['test']], y_pred)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
