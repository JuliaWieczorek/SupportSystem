import json
import pandas as pd

def load_data(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        data = json.load(file)
    return data

def base_stats(df):
    emotions_stats = df['emotion_type'].value_counts()
    problem_stats = df['problem_type'].value_counts()
    supporter_strategies_stats = df['dialog'].apply(lambda x: [item['annotation']['strategy'] for item in x if
                                                               'strategy' in item[
                                                                   'annotation']]).explode().value_counts()

    print("Emotion Statistics:")
    print(emotions_stats)
    print("\nProblem Type Statistics:")
    print(problem_stats)
    print("\nSupporter Strategy Statistics:")
    print(supporter_strategies_stats)
def lenght_stats(df):
    df['dialog_length'] = df['dialog'].apply(len)
    average_dialog_length = df['dialog_length'].mean()
    max_dialog_length = df['dialog_length'].max()
    min_dialog_length = df['dialog_length'].min()


    questions_count = 0
    answers_count = 0

    for dialog in df['dialog']:
        for i in range(len(dialog) - 1):
            if dialog[i]['speaker'] == 'seeker' and dialog[i + 1]['speaker'] == 'supporter':
                questions_count += 1
            elif dialog[i]['speaker'] == 'supporter' and dialog[i + 1]['speaker'] == 'seeker':
                answers_count += 1

    print("Average Dialogue Length:", average_dialog_length)
    print("Maximum Dialogue Length: ", max_dialog_length)
    print("Minimum Dialogue Length: ", min_dialog_length)
    print("Number of Questions from Seeker:", questions_count)
    print("Number of Responses from Supporter:", answers_count)

def additional_analysis(df):
    # Frequency of Emotions by Experience Type
    emotion_experience_stats = df.groupby(['experience_type', 'emotion_type']).size()

    # Average Survey Scores
    average_survey_scores = df['survey_score'].apply(lambda x: pd.Series(x['seeker'])).apply(pd.to_numeric,
                                                                                             errors='coerce').mean()
    print("\nFrequency of Emotions by Experience Type:")
    print(emotion_experience_stats)

    print("\nAverage Survey Scores (Seeker):")
    print(average_survey_scores)

def main():
    dataset = load_data("ESConv.json")
    df = pd.DataFrame(dataset)
    base_stats(df)
    lenght_stats(df)
    additional_analysis(df)



if __name__ == "__main__":
    main()
