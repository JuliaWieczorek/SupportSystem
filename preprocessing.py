''' preprocessing.py '''

import sys
import csv
import json
import pandas as pd
import numpy as np

def read_dict():
    data_dict = {}

    with open("anger-NRC-Emotion-Lexicon.txt", "r") as file:
        reader = csv.reader(file, delimiter='\t')
    
        for row in reader:
            word = row[0]
            label = int(row[1])
            data_dict[word] = label

    return data_dict

def read_dataset():
    file_path = "ESConv.json"

    with open(file_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)
    return dataset

def dataset_to_df():
    # base on the ESConv.json create the pd.DataFrame with Conversation ID and emotion dominate in this ID (emotion taking form the conversation with psychologist)

    emotion_type_list = []
    conv_id = 0
    for corpus in dataset:
        conv_id += 1
        emotion_type = corpus['emotion_type']
        emotion_type_list.append([conv_id, emotion_type])
    
    emotion_type_df = pd.DataFrame(emotion_type_list, columns=['Conversation ID', 'Emotion Type'])
    return emotion_type_df

def emotionToSentiment(emotion):
    
    """ This function will change the type of emotion base on the NRC Emotion Intensity Lexicon to the sentiment (positive/negative)"""
    
    # Surprise should be in positive or negative sentiment? check it!!
    positive = ['joy', 'surprise', 'trust']
    negative = ['anger', 'anticipation', 'disgust', 'fear', 'sadness']
    
    if emotion in positive:
        return 'positive'
    else:
        return 'negative'


def dominate_emotion():
    emotion_type_df['Sentiment'] = emotion_type_df['Emotion Type'].apply(emotionToSentiment)
    emotion_type_df['Dominate Emotion'] = emotion_type_df['Emotion Type'] + ' emotion'

def seeker_df():
    ''' Create a pd.df with Conversation ID and Seeker Dialog '''
    corpus_data = []
    for corpus in dataset:
        dialog = corpus['dialog']
        corpus_data.append(dialog)

    df = pd.DataFrame({'Dialog': pd.Series(corpus_data)})

    seeker_df = pd.DataFrame(columns=['ConversationID', 'Seeker Dialog'])
    for index, row in df.iterrows():
        dialog_list = row['Dialog']
        for dialog in dialog_list:
            if dialog['speaker'] == 'seeker':
                seeker_df = seeker_df.append({'ConversationID': index, 'Seeker Dialog': dialog['content']}, ignore_index=True)            
    return seeker_df

def normalization(dataset):
    """Input: pd.Dataframe"""
    # Removing unnecessary characters such as newlines and extra spaces
    dataset['Seeker Dialog'] = dataset['Seeker Dialog'].str.replace('\n', ' ')  # Replacing newline characters with spaces

    # Converting the text to lowercase for consistency
    dataset['Seeker Dialog'] = dataset['Seeker Dialog'].str.lower()

    # Tokenizing the text into individual words or sentences
    dataset['Tokens'] = dataset['Seeker Dialog'].apply(word_tokenize)

    # Removing any stop words if required
    stop_words = set(stopwords.words('english'))
    dataset['Tokens'] = dataset['Tokens'].apply(lambda tokens: [token for token in tokens if token not in stop_words])

    # Lemmatization or stemming to reduce words to their base forms
    lemmatizer = WordNetLemmatizer()
    dataset['Tokens'] = dataset['Tokens'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

    return dataset

#seeker_df = normalization(seeker_df)

def read_sentiment_file(sentiment): 
    
    '''read a dataset with sentiments
       return: dictionary word and label'''
    
    data_dict = {}
    file = "NRC_Emotion_Lexicon/NRC_Emotion_Lexicon/OneFilePerEmotion/" + sentiment + "-NRC-Emotion-Lexicon.txt"
    
    print('The ', file, 'was read.')
    
    with open(file, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        
        for row in reader:
            word = row[0]
            label = int(row[1])
            data_dict[word] = label
    
    return data_dict

def wordSentimentAssociations(word_to_find, dataset_of_sentiments):
    
    '''input: sentiment = ['positive', 'negative', ]
       read dataset with proper sentiment, and search the word in dataset
       return label'''
    

    if word_to_find in dataset_of_sentiments:
        label_for_word = dataset_of_sentiments[word_to_find]
        return label_for_word
    #else:
    #    print(f"'{word_to_find}' not found in the data.") # now it's an error, have to be changed (probably) to Na

def totalRawWordSentiment(text, data_dict_emotion):
    
    '''input: text <- tokens and dict_emotion <- NRC file (depends of the emotion different file)
       read a proper text (sentence) to map
       for each word search for a sentiment (0 or 1)
       return: total of sentiments in sentence
    '''
    sentiments = []
    
    # map sentiment for each word
    for i in text:
        sent = wordSentimentAssociations(i, data_dict_emotion)
        sentiments.append(sent) 
    sentiments = [i for i in sentiments if i is not None] # delete None form the list of sentiments
    if not sentiments:
        return None # if there is no sentiment return None
    total = sum(sentiments)
    return total

def applyTotalFor8Emotion(dataset):
    
    '''Read sentiments for each of the 8 emotions
    at the end return file .exe with tokens and total of emotions
    INPUT: dataframe with tokens (pd.Dataframe)
    '''
    
    emotion8 = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    for emotion in emotion8:
        data_dict = read_sentiment_file(emotion)
        #dataset[emotion + ' sentiment'] = dataset['Tokens'].apply(totalRawWordSentiment(emotion))
        dataset[emotion + ' emotion'] = dataset['Tokens'].apply(lambda row: totalRawWordSentiment(row, data_dict))
        
    output_file = '8emotions.xlsx'
    dataset.to_excel(output_file, index=False)

    print("Total of 8 emotions saved to", output_file)

def applyTotalForSentiment(dataset):
    
    '''Read sentiments for each of the positive and negative sentiment
    at the end return file .exe with tokens and total of emotions
    INPUT: dataframe with tokens (pd.Dataframe)
    '''
    
    sentiments = ['positive', 'negative']
    for sentiment in sentiments:
        data_dict = read_sentiment_file(sentiment)
        dataset[sentiment + ' sentiment'] = dataset['Tokens'].apply(lambda row: totalRawWordSentiment(row, data_dict))
        
    output_file = '8emotions.xlsx'
    dataset.to_excel(output_file, index=False)

    print("Total of 8 emotions saved to", output_file)

def dominateEmotion(df):
    df1 = df.filter(regex='emotion')
    mxs = df1.eq(df1.max(axis=1), axis=0)
    # join the column names of the max values of each row into a single string
    df['Dominate Emotion'] = mxs.dot(mxs.columns + ', ').str.rstrip(', ')


    
