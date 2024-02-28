# app_functions.py
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
from rouge import Rouge
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import plotly.express as px
from collections import Counter
import string
from transformers import PreTrainedTokenizer

model_ckpt = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer_t5 = T5Tokenizer.from_pretrained(model_ckpt)
model_mini = T5ForConditionalGeneration.from_pretrained(model_ckpt, device_map='auto', torch_dtype=torch.float32)

tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
model_gpt2 = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')

def file_preprocessing(file):
    loader= PyPDFLoader(file)
    pages=loader.load_and_split()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=40)
    texts= text_splitter.split_documents(pages)
    final_texts=""
    for text in texts:
        final_texts=final_texts+ text.page_content
    return final_texts

def summarize_document(file_path):
    pipe_sum=pipeline(
        'summarization',
        model=model_mini,
        tokenizer=tokenizer_t5,
        max_length=1000,
        min_length=50
    )
    input_text=file_preprocessing(file_path)
    result= pipe_sum(input_text)
    result= result[0]['summary_text']
    return result

def display_pdf(file):
    with open(file, "rb") as f:
        base64_pdf=base64.b64encode(f.read()).decode('utf-8')
    
    pdf_disp=F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_disp, unsafe_allow_html=True)

def evaluate_model(generated_summary, reference_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores

def calc_perplexity(text):
    encoded_input = tokenizer_gpt2.encode(text, add_special_tokens=False, return_tensors='pt')
    input_ids = encoded_input[0]

    with torch.no_grad():
        outputs = model_gpt2(input_ids)
        logits = outputs.logits

    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))
    return perplexity.item()

def calc_burstiness(text):
    tokens = nltk.word_tokenize(text.lower())
    word_freq = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_freq.values())
    burstiness_score = repeated_count / len(word_freq)
    return burstiness_score

def plot_top_repeated_words(text):
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]

    word_counts = Counter(tokens)

    top_words = word_counts.most_common(10)

    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]

    fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Counts'}, title='Top 10 Most Repeated Words')
    st.plotly_chart(fig, use_container_width=True)

def analyze_text(text):
    perplexity = calc_perplexity(text)
    burstiness_score = calc_burstiness(text)
    return perplexity, burstiness_score
