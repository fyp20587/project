 Final Year Project


Main project (BERT finetuning): zakyabert/sentiment_modelling.ipynb 

Data preprocessing: zakyabert/datasets/data_pre-processing

Supervised learning models: zakyabert/zakyascikit/supervisedmodels.ipynb


Website source code: zakyabert/UI/app.py To run the website: streamlit run app.py



Requirements:
streamlit>=0.84.0
torch==1.9.0
transformers==4.10.0
numpy==1.21.1
matplotlib==3.4.3
huggingface-hub>=0.15.1,<1.0


Steps:
To access the website, go to zakyabert/UI/app.py
And on terminal run with ‘streamlit run app.py’

To access deployed website: https://sentimentoday.streamlit.app/
Please note the deployed website may take long to run due to the cloud services…

If you want to run everything yourself please follow these steps: 
Run ‘data pre-processing’ for datasets ‘financialdata.csv’, ‘SEN_en_AMT_nooutlier.csv’ and ‘SEN_en_R_nooutlier.csv’ to produce ‘processed_data.csv’

Run ‘sentiment_modelling’ which is the BERT Finetuned code to produce models (bertModel.pt, config.json, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, vocab.txt) needed.

Run app.py with command ‘streamlit run app.py’ to access website with the deployed models’.



Note: will take longer to load when using CPU (instead of GPU), also the processed dataset and models are already produced so all steps not necessary 


 Github repository for access to all files: https://github.com/fyp20587/project
 
