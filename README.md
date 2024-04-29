 Final Year Project: SentimentToday 
 To access deployed website: https://sentimentoday.streamlit.app/ 


- Download the zip file and rename it 'zakyabert'
- Use conda environment:
    -> python -m venv myen
    -> source myenv/bin/activate  # On Unix/macOS
    -> myenv\Scripts\activate  # On Windows

- Install Dependences
    -> pip install -r requirements.txt

Main project (BERT finetuning): zakyabert/sentiment_modelling.ipynb 

Data preprocessing: zakyabert/datasets/data_pre-processing

Supervised learning models: zakyabert/zakyascikit/supervisedmodels.ipynb


Website source code: zakyabert/UI/app.py
To run the website: streamlit run app.py


Main Requirements:
streamlit>=0.84.0
torch==1.9.0
transformers==4.10.0
numpy==1.21.1
matplotlib==3.4.3
huggingface-hub>=0.15.1,<1.0


If you want to run everything yourself please also follow these steps: 

Run ‘data pre-processing’ for datasets ‘financialdata.csv’, ‘SEN_en_AMT_nooutlier.csv’ and ‘SEN_en_R_nooutlier.csv’ to produce ‘processed_data.csv’

Run ‘sentiment_modelling’ which is the BERT Finetuned code to produce models (bertModel.pt, config.json, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, vocab.txt) needed.

Run app.py with command ‘streamlit run app.py’ to access website with the deployed models’.



Note: will take longer to load when using CPU (instead of GPU), also the processed dataset and models are already produced so all steps not necessary 


 Github repository for access to all files: https://github.com/fyp20587/project
 
