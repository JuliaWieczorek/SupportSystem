import os
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

import spacy
from spacy.matcher import Matcher

import seaborn as sns
import matplotlib.pyplot as plt

"""
    Description:

    The function emotion_classifier_workflow orchestrates a complete workflow for emotion classification based on dialogues. This function automates the process from data loading to model evaluation, offering a structured approach for emotion analysis. It supports multiple classifiers and provides insights into the emotional content of dialogues.

    Parameters:

    None

    Returns:

    comparison_df: A DataFrame containing the comparison of classifiers with their respective accuracies.
    Saved confusion matrices and results for each classifier to disk.

    Workflow Steps:

    Data Loading: Load the dataset from the file "ESConv.json".
    Data Preprocessing:
        Extract relevant data from the dataset.
        Map emotions to numerical values.
        Create a DataFrame and filter out excluded emotions.
    Data Splitting: Split the dataset into training and testing sets.
    Tokenization: Tokenize the dialogues in the training and testing sets.
    Feature Extraction: Extract features related to self-reference, focus on the past, and emotions using the NRC Lexicon.
    TF-IDF Vectorization: Convert the tokenized dialogues into TF-IDF vectors.
    Model Training and Evaluation:
        Train a Random Forest Classifier using the combined features.
        Evaluate the model's accuracy on the testing set.
    Classifier Comparison:
        Train and evaluate multiple classifiers including Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Naive Bayes, and Gradient Boosting.
        Generate a DataFrame with a comparison of classifiers and their accuracies.
    Results Storage:
        Save the confusion matrices and results of each classifier to disk for further analysis

    Usage Example:
    comparison_results = emotion_classifier_workflow()
    print(comparison_results)

    """

nlp = spacy.load("en_core_web_sm")

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
    past_words = set()
    past_expressions = set()

    for i in dialog:
        for token in nlp(i):
            if token.tag_ in ["VBD", "VBN"]:  # VBD - Past tense, VBN - Past participle
                past_words.add(token.text.lower())
            if token.text.lower() in ["yesterday", "ago", "last", "past"]:
                past_expressions.add(token.text.lower())
    return len(past_words) + len(past_expressions)

def load_nrc_lexicon(folder_path, emotions_to_load):
    lexicon = {}
    for emotion in emotions_to_load:
        filename = emotion + "-NRC-Emotion-Lexicon.txt"
        with open(os.path.join(folder_path, filename), 'r') as file:
            words_and_values = [line.strip().split('\t') for line in file]
            words = {word: int(value) for word, value in words_and_values}
            lexicon[emotion] = words
    return lexicon

def nrc_emotions_count(dialog, nrc_lexicon):
    emotion_counts = []
    for emotion, emotion_words in nrc_lexicon.items():
        emotion_count = sum(word in emotion_words for word in dialog)
        emotion_counts.append(emotion_count)
    return emotion_counts

def sentence_lengths(dialogs):
    avg_lengths = []
    min_lengths = []
    max_lengths = []

    for dialog in dialogs:
        # Podziel dialog na zdania za pomocą znaku kropki
        sentences = dialog.split('.')

        # Usuń puste ciągi i spacje z listy
        sentences = [s.strip() for s in sentences if s.strip()]

        # Oblicz długość każdego zdania
        sentence_lengths = [len(sentence.split()) for sentence in sentences]

        if not sentence_lengths:
            avg_lengths.append(0)
            min_lengths.append(0)
            max_lengths.append(0)
            continue

        avg_lengths.append(sum(sentence_lengths) / len(sentence_lengths))
        min_lengths.append(min(sentence_lengths))
        max_lengths.append(max(sentence_lengths))

    return avg_lengths, min_lengths, max_lengths

def conversation_length(dialogs):
    return [len(word_tokenize(dialog)) for dialog in dialogs]

def divide_and_analyze_conversation(dialog_tokens, num_parts=3):
    """
    Divide the conversation into multiple parts and analyze how the number of features changes.

    :param dialog_tokens: List of tokenized dialog.
    :param num_parts: Number of parts to divide the conversation.
    :return: List of number of features for each part.
    """
    part_length = len(dialog_tokens) // num_parts
    features_per_part = []

    for i in range(num_parts):
        start_idx = i * part_length
        end_idx = (i + 1) * part_length if i != num_parts - 1 else len(dialog_tokens)
        part_dialog_tokens = dialog_tokens[start_idx:end_idx]

def save_confusion_matrix_to_file(cm, classifier_name, emotion_labels):
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    file_name = f"confusion_matrix_{classifier_name}_{current_time}.png"
    if not os.path.exists(f'confusion_matrix/{current_time}'):
        os.makedirs(f'confusion_matrix/{current_time}')

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=emotion_labels.values(),
                yticklabels=emotion_labels.values())
    plt.title(f"Confusion Matrix for {classifier_name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(file_name)
    plt.close()
    print('Confusion matrix save in ', f'confusion_matrix/{classifier_name}_confusion_matrix.png')

def save_results_to_disk(df_results, classifier_name):
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    file_name = f"results_{classifier_name}_{current_time}.csv"
    if not os.path.exists(f"results/{current_time}"):
        os.makedirs(f"results/{current_time}")
    file_name = f"results/{current_time}/" + file_name
    df_results.to_csv(file_name, index=False)

def emotion_classifier_workflow():

    # Load data
    dataset = load_data("ESConv.json")

    # Extract and map emotion to numbers
    extracted_data = extract_data(dataset)
    mapped_data = map_emotion_to_numbers(extracted_data)

    # Create DataFrame and filter out excluded emotions
    mapped_data_df = pd.DataFrame(mapped_data)
    balanced_data = mapped_data_df[~mapped_data_df['Emotion Type'].isin(EXCLUDED_EMOTIONS)]

    # Train-test split
    splitted_dialogues = train_test_split_dialogues(balanced_data)

    # Features before tokenization
    avg_sentence_lengths_train, min_sentence_lengths_train, max_sentence_lengths_train = sentence_lengths(
        [dialog['Seeker Dialog'] for dialog in splitted_dialogues['train']])
    conversation_lengths_train = conversation_length([dialog['Seeker Dialog'] for dialog in splitted_dialogues['train']])

    avg_sentence_lengths_test, min_sentence_lengths_test, max_sentence_lengths_test = sentence_lengths(
        [dialog['Seeker Dialog'] for dialog in splitted_dialogues['test']])
    conversation_lengths_test = conversation_length([dialog['Seeker Dialog'] for dialog in splitted_dialogues['test']])


    # Tokenize dialogues
    tokenized_dialogues = tokenize_dialogues(splitted_dialogues)

    # Load NRC Lexicon
    # Load emotion lexicons which can be found in our dataframe
    nrc_lexicon = load_nrc_lexicon(NRC_LEXICON_FOLDER, ["disgust", "fear", "sadness", "negative"])
    # TODO: in NRC lexicon lack of anxiety, depression, anger and shame emotion (which can be found in dataframe)

    # Features for train
    X_train_self_reference = [self_reference_count(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['train']]
    X_train_focus_past = [focus_past_count(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['train']]
    X_train_nrc_emotions = [nrc_emotions_count([word.lower() for word in dialog['Seeker Dialog']], nrc_lexicon) for
                            dialog in tokenized_dialogues['train']]
    # Separate counts for emotions
    X_train_disgust = [emotion_counts[0] for emotion_counts in X_train_nrc_emotions]
    X_train_fear = [emotion_counts[1] for emotion_counts in X_train_nrc_emotions]
    X_train_sadness = [emotion_counts[2] for emotion_counts in X_train_nrc_emotions]
    X_train_negative = [emotion_counts[3] for emotion_counts in X_train_nrc_emotions]

    # Features for test
    X_test_self_reference = [self_reference_count(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['test']]
    X_test_focus_past = [focus_past_count(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['test']]
    X_test_nrc_emotions = [nrc_emotions_count([word.lower() for word in dialog['Seeker Dialog']], nrc_lexicon) for
                           dialog in tokenized_dialogues['test']]
    # Separate counts for emotions
    X_test_disgust = [emotion_counts[0] for emotion_counts in X_test_nrc_emotions]
    X_test_fear = [emotion_counts[1] for emotion_counts in X_test_nrc_emotions]
    X_test_sadness = [emotion_counts[0] for emotion_counts in X_test_nrc_emotions]
    X_test_negative = [emotion_counts[1] for emotion_counts in X_test_nrc_emotions]

    # Features arrays
    X_train_features = np.array(
        [X_train_self_reference, X_train_focus_past, X_train_fear, X_train_disgust, X_train_sadness,
         X_train_negative, avg_sentence_lengths_train, min_sentence_lengths_train, max_sentence_lengths_train,
         conversation_lengths_train]).T
    X_test_features = np.array(
        [X_test_self_reference, X_test_focus_past, X_test_fear, X_test_disgust, X_test_sadness, X_test_negative,
         avg_sentence_lengths_test, min_sentence_lengths_test, max_sentence_lengths_test, conversation_lengths_test]).T

    tokenized_dialogues = tokenize_dialogues(splitted_dialogues)

    # Model TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(
        [" ".join(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['train']])
    X_test_tfidf = vectorizer.transform([" ".join(dialog['Seeker Dialog']) for dialog in tokenized_dialogues['test']])

    # Combine features with TF-IDF
    X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_features))
    X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test_features))

    # Classification model - RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train_combined, [dialog['Emotion Type'] for dialog in tokenized_dialogues['train']])

    y_pred = model.predict(X_test_combined)
    accuracy = accuracy_score([dialog['Emotion Type'] for dialog in tokenized_dialogues['test']], y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    #
    # Create DataFrame with features and true/predicted labels
    df_results = pd.DataFrame({
        'Self Reference': X_test_self_reference,
        'Focus Past': X_test_focus_past,
        'Fear': X_train_fear,
        'Disgust': X_train_disgust,
        'Sadness': X_test_sadness,
        'Negative': X_test_negative,
        'True Label': [dialog['Emotion Type'] for dialog in tokenized_dialogues['test']],
        'Predicted Label': y_pred
    })

    # Map emotion labels to their corresponding names
    emotion_labels = {0: 'anxiety', 1: 'depression', 2: 'sadness', 3: 'anger', 4: 'fear', 5: 'shame', 6: 'disgust',
                      7: 'nervousness', 8: 'pain', 9: 'guilt', 10: 'jealousy'}
    # TODO: check if still correct labelling
    df_results['True Label'] = df_results['True Label'].map(emotion_labels)
    df_results['Predicted Label'] = df_results['Predicted Label'].map(emotion_labels)

    print(df_results)

    # features_per_part = divide_and_analyze_conversation(tokenized_dialogues, num_parts=3)
    # print("Number of features (TF-IDF) per part:", features_per_part)

    classifiers = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': MultinomialNB(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    comparison_df = pd.DataFrame(columns=['Classifier', 'Accuracy'])

    for classifier_name, classifier in classifiers.items():
        classifier.fit(X_train_combined, [dialog['Emotion Type'] for dialog in tokenized_dialogues['train']])

        y_pred_classifier = classifier.predict(X_test_combined)
        accuracy_classifier = accuracy_score([dialog['Emotion Type'] for dialog in tokenized_dialogues['test']],
                                             y_pred_classifier)

        temp_df = pd.DataFrame({'Classifier': [classifier_name], 'Accuracy': [accuracy_classifier]})
        comparison_df = pd.concat([comparison_df, temp_df], ignore_index=True)

        print(f"Results for {classifier_name}:")
        print(f'Accuracy ({classifier_name}): {accuracy_classifier * 100:.2f}%\n')

        cm = confusion_matrix([dialog['Emotion Type'] for dialog in tokenized_dialogues['test']], y_pred_classifier)
        save_confusion_matrix_to_file(cm, classifier_name, emotion_labels)

        df_results = pd.DataFrame({
            'Self Reference': X_test_self_reference,
            'Focus Past': X_test_focus_past,
            'Fear': X_test_fear,
            'Disgust': X_test_disgust,
            'Sadness': X_test_sadness,
            'Negative': X_test_negative,
            'True Label': [dialog['Emotion Type'] for dialog in tokenized_dialogues['test']],
            'Predicted Label': y_pred_classifier
        })
        save_results_to_disk(df_results, classifier_name)
    print("Comparison of Classifiers:")
    print(comparison_df)

def main():
    emotion_classifier_workflow()

if __name__ == "__main__":
    main()
