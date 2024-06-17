
# pip install https://huggingface.co/docs/transformers/v4.15.0/en/installation check out this page and follow instruction 
# for downloading 

from transformers import pipeline
# classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
classifier = pipeline(task="text-classification",model = "j-hartmann/emotion-english-distilroberta-base", top_k=None)
emotions_list = classifier(['I am not having a great day.',
                            "i'm sorry that the order got delayed",
                            "why are we not fair on promotions?",
                            "why are our appraisal hike is so low while HSBC has done well financially?",
                            "why the insurance limit is so less for GCB6?"])
# emotions_list

classifier_sentiment = pipeline(task="sentiment-analysis")
sentiment_list = classifier_sentiment(["why the increment was so lower while the bank has done so well?","why the insurance limit is so less for GCB6?"])
sentiment_list

def getting_top_emotions(text,top:int):
    
    # Iterate through each list of emotions
    for index, emotion_list in enumerate(text):

        # Sort the list based on the score in descending order
        sorted_emotions = sorted(emotion_list, key=lambda x: x['score'], reverse=True)
        
        # Select the top two emotions
        top_emotions = sorted_emotions[:top]
        
        # Print the statement and top emotions
        print(f"Statement {index + 1}:")
        for emotion in top_emotions:
            print(f"{emotion['label']}: {emotion['score']}")
        print()  # Add a blank line for readability

getting_top_emotions(emotions_list,3)
   
