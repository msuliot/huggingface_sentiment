import os
from transformers import pipeline, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

from dotenv import load_dotenv
load_dotenv()
hf_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def set_local_vars():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment" 
    pipeline_task = "sentiment-analysis"
    return model_name, pipeline_task


def hf_local(model_name, pipeline_task, text):
    # This will download the model and tokenizer to your local machine and run on your local machine. 
    # saved and cached ~/.cache/huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_tensors="pt") # return_tensors="pt" or "tf"
    model = AutoModelForSequenceClassification.from_pretrained(model_name) # AutoModelForSequenceClassification or TFAutoModelForSequenceClassification
    pipe = pipeline(pipeline_task, model=model, tokenizer=tokenizer)
    output = pipe(text)
    return output


def main():
    model_name, pipeline_task = set_local_vars()

    text = "I am very happy that you're watching this video."
    
    return_value = hf_local(model_name, pipeline_task, text)
    print(return_value)

if __name__ == "__main__":
    main() 