
# pip install https://huggingface.co/docs/transformers/v4.15.0/en/installation check out this page and follow instruction 
# for downloading #

from transformers import pipeline
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
emotions_list = classifier(['I am not having a great day.',"i'm sorry that the order got delayed"])


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
   
