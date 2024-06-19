
# pip install https://huggingface.co/docs/transformers/v4.15.0/en/installation check out this page and follow instruction 
# for downloading 
import pandas as pd
from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
# emotions_list = classifier(['I am not having a great day.',"i'm sorry that the order got delayed"])


def getting_top_emotions(text,top:int):

    # Iterate through each list of emotions
    for index, emotion_list in enumerate(text):

        # Sort the list based on the score in descending order
        sorted_emotions = sorted(emotion_list, key=lambda x: x['score'], reverse=True)
        
        # Select the top two emotions
        top_emotions = sorted_emotions[:top]
        print(top_emotions)
        # Print the statement and top emotions
        # print(f"Statement {index + 1}:")
        for emotion in top_emotions:
            print(f"{emotion['label']}: {emotion['score']}")
        print()  # Add a blank line for readability
        return top_emotions

# getting_top_emotions(emotions_list,3)
def roberta_sentence(text):
    d1=getting_top_emotions(classifier([text]),3)
    df=pd.DataFrame.from_records(d1)
    df=df.reset_index(drop=True)
    return df
def roberta_df(df):
    res = {}
    for i, row in df.iterrows():
        text = str(row['Text'])
        myid = row['Id']
        res[myid] = getting_top_emotions(classifier([text]),3)
        # getting_top_emotions()

    roberta = pd.DataFrame(res).T
    column_mapping={i:f'Top emotion {i+1}' for i in range (len(roberta.columns))}
    roberta=roberta.rename(columns=column_mapping)
    def swap_keys_values(d):
        return {d['label']:d['score']}
    roberta=roberta.map(swap_keys_values)
    roberta = roberta.reset_index().rename(columns={'index': 'Id'})
    roberta = roberta.merge(df, how='left')
    roberta=roberta.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time'],axis=1)
    f_col=roberta.pop('Text')
    roberta.insert(0,'Text',f_col)
    print(roberta)
    return roberta
    # output = vaders[['Text', 'neg', 'neu', 'pos', 'compound']]

text="I am happy"
d1=roberta_sentence(text)
print(d1)
# df=pd.read_csv("demo.csv")
# df1=roberta_df(df)
# print(df1)