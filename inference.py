import torch
from transformers import BertTokenizer, BertForSequenceClassification


def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    sentiment_labels = ['negative', 'positive']
    predicted_sentiment = sentiment_labels[predicted_label]
    return predicted_sentiment
