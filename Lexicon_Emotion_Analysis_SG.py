from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Example text
text = "I'm so happy"

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
emotion_scores = {emotion: sum(emotion_lexicon[word].get(emotion, 0) for word in emotion_lexicon) for emotion in set(emotion_lexicon)}

# Print emotion scores
print("Emotion Scores:")
for emotion, score in emotion_scores.items():
    print(f"{emotion}: {score}")


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Example text
text = "I'm so happy and excited to see you, but also a little nervous about the presentation."

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

# Print words not found in lexicon
not_found_words = [word for word in filtered_tokens if word not in emotion_lexicon]
print("Words not found in lexicon:", not_found_words)

