import pandas as pd
import streamlit as st
import Vader_sentiment_scoring as vader
import RoBERta_Emotion_HF_model as Roberta
import Lexicon_Emotion_Analysis as Lexicon
st.title(":red[MoodMapper]")
st.write("**Emotions unfolded!**")
st.write("------------")
st.sidebar.image('HSBC-logo.png',use_column_width=True)
add_selectbox = st.sidebar.selectbox(
    "Choose model",
    ("RoBERTa", "Lexicon Emotion Analysis")
)
text=st.text_input("Enter your statemennt")
if st.button("Detect"):
    if add_selectbox == "RoBERTa":
        if text:
            c1,c2=st.columns(2)
            with c1:
                st.write("Emotion Scores:")
                st.dataframe(Roberta.roberta_sentence(text),hide_index=True)
            with c2:
                st.write("Sentiment Analysis:")
                st.dataframe(vader.vader_sentence(text),hide_index=True)
    if add_selectbox == "Lexicon Emotion Analysis":
        if text:
            c1,c2=st.columns(2)
            with c1:
                st.write("Emotion Scores:")
                d1=Lexicon.lexicon_sentence(text)['df']
                d1=d1[d1["Scores"]!=0]
                st.dataframe(d1,hide_index=True)
            with c2:
                st.write("Sentiment Analysis:")
                st.dataframe(vader.vader_sentence(text),hide_index=True)

data=st.sidebar.file_uploader("Upload csv file",type=["csv","xlsx"])
try:
    df=pd.read_csv(data)
except:
    st.sidebar.write("No data uploaded")
if st.sidebar.button("Run"):
    with st.spinner("In progress..."):
        if add_selectbox == "RoBERTa":
            st.write("Emotion Scores:")
            st.dataframe(Roberta.roberta_df(df),hide_index=True)
            st.write("Sentiment Analysis:")
            st.dataframe(vader.vader_df(df),hide_index=True)
        if add_selectbox == "Lexicon Emotion Analysis":
            st.write("Emotion Scores:")
            st.dataframe(Lexicon.lexicon_df(df),hide_index=True)
            st.write("Sentiment Analysis:")
            st.dataframe(vader.vader_df(df),hide_index=True)


