
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split



df = load_data("C:\\Users\\vahan.yeghoyan\\Desktop\\projects\\roBERTa\\data\\IMDB Dataset.csv")
df = preprocess_data(df)


max_length = 512
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_input_ids, train_attention_masks = tokenize_text(train_df['review'].values, tokenizer, max_length)
val_input_ids, val_attention_masks = tokenize_text(val_df['review'].values, tokenizer, max_length)


train_labels = torch.tensor(train_df['sentiment'].values)
val_labels = torch.tensor(val_df['sentiment'].values)


batch_size = 16
train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_loader = DataLoader(val_data, batch_size=batch_size)



model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 5)


num_epochs = 10
train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs)




text_to_analyze = "I love spending time with my family; they bring so much joy into my life."
predicted_sentiment = predict_sentiment(text_to_analyze, model, tokenizer)
print(f"Predicted Sentiment: {predicted_sentiment}")
