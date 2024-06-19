# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import nltk
# # nltk.download('stopwords')
# # nltk.download('punkt')
#
# # Example text
# text = "I'm so happy"
#
# # Tokenize the text and remove stopwords
# tokens = word_tokenize(text.lower())
# stop_words = set(stopwords.words('english'))
# filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
#
# # Load NRC Emotion Lexicon
# emotion_lexicon = {}
# with open("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", "r") as lexicon_file:
#     for line in lexicon_file.readlines():
#         word, emotion, value = line.strip().split("\t")
#         if word in filtered_tokens:
#             if word not in emotion_lexicon:
#                 emotion_lexicon[word] = {}
#             emotion_lexicon[word][emotion] = int(value)
#
# # Aggregate emotion scores
# emotion_scores = {emotion: sum(emotion_lexicon[word].get(emotion, 0) for word in emotion_lexicon) for emotion in set(emotion_lexicon)}
#
# # Print emotion scores
# print("Emotion Scores:")
# for emotion, score in emotion_scores.items():
#     print(f"{emotion}: {score}")
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

# Example text
# text = "I'm so happy and excited to see you, but also a little nervous about the presentation."
def lexicon_sentence(text):

    # Tokenize the text and remove stopwords
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Load NRC Emotion Lexicon
    emotion_lexicon = {}
    with open("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", "r") as lexicon_file:
        for line in lexicon_file.readlines():
            word, emotion, value = line.strip().split("\t")
            if word in filtered_tokens:
                if word not in emotion_lexicon:
                    emotion_lexicon[word] = {}
                emotion_lexicon[word][emotion] = int(value)

    # Aggregate emotion scores
    emotion_scores = {emotion: sum(emotion_lexicon.get(word, {}).get(emotion, 0) for word in filtered_tokens) for emotion in ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']}
    # Print emotion scores
    print("Emotion Scores:")
    for emotion, score in emotion_scores.items():
        print(f"{emotion}: {score}")
    df=pd.DataFrame(list(emotion_scores.items()),columns=["Emotions","Scores"]).sort_values(by="Scores",ascending=False)
    # Print words not found in lexicon
    print(df)
    not_found_words = [word for word in filtered_tokens if word not in emotion_lexicon]
    print("Words not found in lexicon:", not_found_words)
    return {'df':df,'notfound':not_found_words,'scores':emotion_scores}
# lexicon_sentence(text)

def lexicon_df(df):
    res = {}
    for i, row in df.iterrows():
        text = row['Text']
        myid = row['Id']
        res[myid] = lexicon_sentence(text)['scores']
    lexicon = pd.DataFrame(res).T
    # column_mapping = {i: f'Top emotion {i + 1}' for i in range(len(lexicon.columns))}
    # lexicon = lexicon.rename(columns=column_mapping)

    lexicon = lexicon.reset_index().rename(columns={'index': 'Id'})
    lexicon = lexicon.merge(df, how='left')
    lexicon = lexicon.drop(
        ['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time'],
        axis=1)
    f_col = lexicon.pop('Text')
    lexicon.insert(0, 'Text', f_col)
    print(lexicon)
    return lexicon
# df=pd.read_csv("demo.csv")
# df1=lexicon_df(df)
# print(df1)