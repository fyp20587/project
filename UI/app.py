#pip install  numpy
import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer,AutoModel
import torch.nn.functional as F
device = torch.device("cpu")
import numpy as np
import matplotlib.pyplot as plt


#define architecture
class BERT_architecture(nn.Module):

  def __init__(self, bert):
    super(BERT_architecture, self).__init__()
    self.bert = bert 
    self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
    self.drop = nn.Dropout(p=0.25)
    #self.relu = nn.ReLU()
    self.out = nn.Linear(512, 3)
    for param in self.bert.parameters():
            param.requires_grad = False
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask, return_dict=False)
    
    outputs = self.fc1(pooled_output)
    #outputs = self.relu(outputs)
    outputs = self.drop(outputs)
    return self.out(outputs)


#train_tokenized
class TextTokenizer:
    def __init__(self, max_len, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def tokenize_texts(self, texts):
        tokenized_texts = [self.tokenizer.encode_plus(
            text=text,
            max_length=self.max_len,
            add_special_tokens=True,
            truncation=True,
            pad_to_max_length=True
        ) for text in texts]

        input_ids = torch.tensor([tokenized['input_ids'] for tokenized in tokenized_texts],dtype=torch.long)
        attention_mask = torch.tensor([tokenized['attention_mask'] for tokenized in tokenized_texts],dtype=torch.long)
            
        return input_ids, attention_mask
#this does prediciton, input id, attention mask (both are obtained from bert tokenizer)
    
#model b input id b attn mask, model is the trained model saved in the directory, input id is what is taken from (paramters from previous class)
        
def bert_predict(model, b_input_ids, b_attn_mask):

    model.eval()

    all_logits = []
    # Assuming 'device' is already defined
    b_input_ids = test_seq.to(b_input_ids)
    b_attn_mask = test_mask.to(b_attn_mask)

   #test_seq, test_mask = tuple(t.to(device) for t in batch)[:2]

    # compute logits
    #no gradients, means just want to predict, not update the model weight 
    with torch.no_grad():
        logits = model(b_input_ids, b_attn_mask)
    all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=1)

    #apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

#function to draw pie chart
def draw_pie_chart(neutral_probability, negative_probability, positive_probability):
    labels = ['Neutral', 'Negative', 'Positive']
    sizes = [neutral_probability, negative_probability, positive_probability]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=10, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig

#loading all from directory 
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('./model')
    bert = AutoModel.from_pretrained('./model')
    model = BERT_architecture(bert)
   # model.bert.embeddings.position_ids = torch.arange(512).unsqueeze(0)
    model.load_state_dict(torch.load("./model/bertModel.pt", map_location=torch.device('cpu')), strict=False)
    
    return model, tokenizer  

max_len=512 #extend the length of word or sentence bc when trained the model said 512 is the hidden input for the model so has 
#to keep the same length 
#page layout below is the streamlit code
#setting page configuration:
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📊",
    layout="centered"
)

# set colour for each sentiment
sentiment_color = {
    "Positive": "#2ecc71",  # Green
    "Neutral": "#3498db",   # Blue
    "Negative": "#e74c3c"   # Red
}

# TITLE section
st.title("Real-time Sentiment Analysis on News Data")
#st.write("hi") --> test
st.markdown(
    f"""
    <div style="background-color:{sentiment_color['Neutral']};padding:10px;border-radius:5px">
        <h3 style="color:white;">Enter the news for sentiment analysis:</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Session State INITIALIZATION
if 'sentiment_result' not in st.session_state:
    st.session_state.sentiment_result = None

# User input section
user_input = st.text_input("")
# Analyze button
if st.button("Analyze"):
    if (user_input) and (len(user_input)<= 512):

        model, tokenizer = load_bert_model()
        max_len = 512  # You might need to set max_len value
        text_tokenizer = TextTokenizer(max_len, tokenizer)
        test_seq, test_mask = text_tokenizer.tokenize_texts([user_input])
        probs = bert_predict(model, test_seq, test_mask)
        pred = np.argmax(probs, axis=1)
        sentiment = ["Neutral", "Negative", "Positive"][pred.tolist()[0]]
        
        # Set background color based on sentiment
        sentiment_section_color = sentiment_color[sentiment]
        
        # Save the sentiment result to session state
        st.session_state.sentiment_result = {
            'sentiment': sentiment,
            'section_color': sentiment_section_color
        }
    else: 
        st.error("The text length cannot be over 512 words.")

# Display sentiment result
if st.session_state.sentiment_result:
    result = st.session_state.sentiment_result
    st.markdown(
        f"""
        <div style="background-color:{result['section_color']};padding:10px;border-radius:5px">
            <h3 style="color:white;">Sentiment: {result['sentiment']}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    # Show Probability button only if analysis has been done
    #EXECUTION: calling all the functions occurs
    if st.button("Show Probability"):
        if user_input:
            model, tokenizer = load_bert_model()
            text_tokenizer = TextTokenizer(max_len, tokenizer)
            test_seq, test_mask = text_tokenizer.tokenize_texts([user_input])
            probs = bert_predict(model, test_seq, test_mask)
            
            neutral_probability = probs[0][0] * 100
            negative_probability = probs[0][1] * 100
            positive_probability = probs[0][2] * 100

            st.subheader("Sentiment Probability Estimates:")

            # Use st.columns to organize content into two columns
            col1, col2 = st.columns(2)

            # Column 1: Display probability estimates
            col1.write(f"This text is {positive_probability:.2f}% likely to be Positive.")
            col1.write(f"This text is {negative_probability:.2f}% likely to be Negative.")
            col1.write(f"This text is {neutral_probability:.2f}% likely to be Neutral.")
            

            # Column 2: Draw pie chart
            col2.pyplot(draw_pie_chart(neutral_probability, negative_probability, positive_probability))
        else:
            st.warning("Please enter a sentence.")