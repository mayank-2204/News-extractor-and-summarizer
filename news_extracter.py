import streamlit as st
from newspaper import fulltext,Article
import requests
import nltk
nltk.download('punkt')
# headers = {'user-agent':'AppleWebKit/535.1'}
link = st.text_area("Enter the link")
st.write(link)
article_url=Article(link)
# article_url=link

article_url.download()
# html = requests.get(link).text
# article = fulltext(html)
article_url.parse()
article = article_url.text
article_url.nlp()
article_summary = article_url.summary
st.write("Original: ")
st.write(article)
st.write("Summary")
st.write(article_summary)

from summarizer import Summarizer, TransformerSummarizer
model = Summarizer()
result = model(article, min_length=30,max_length=300)
summary = "".join(result)
st.write("BERT summary")
st.write(summary)

GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
gpt2_result = GPT2_model(article, min_length = 30, max_length = 300 )
gpt_summary = "".join(gpt2_result)
st.write("GPT2 summary")
st.write(gpt_summary)

xlnet_model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
XLNET_result = xlnet_model(article, min_length = 30, max_length = 300 )
XLNET_summary = "".join(XLNET_result)
st.write("XLNET summary")
st.write(XLNET_summary)
