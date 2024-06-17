import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.style.use('ggplot')

#
import nltk
import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()
# Read in data
df = pd.read_csv('Reviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)

example = df['Text'][50]
print(example)

tokens = nltk.word_tokenize(example)
tokens[:10]

tagged = nltk.pos_tag(tokens)
tagged[:10]

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

sia.polarity_scores('I am so happy!')

sia.polarity_scores('This is the worst thing ever.')

sia.polarity_scores(example)

# Run the polarity score on the entire dataset
res = {}
for i, row in df.iterrows():
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
output = vaders[['Text','neg', 'neu', 'pos', 'compound']]
# Now we have sentiment score and metadata
print(output.head())