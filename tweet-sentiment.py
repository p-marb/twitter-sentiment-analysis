import tensorflow as tf
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
class SentimentDataset(Dataset):
    def __init__(self, tokenizer, filepath):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(filepath)
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=512)
        item['labels'] = self.labels[idx]
        return item

def train_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = SentimentDataset(tokenizer, "sent_train.csv")
    val_dataset = SentimentDataset(tokenizer, "sent_valid.csv")

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(labels, pred)
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def run_model(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-1500")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
    outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)  # Apply softmax to convert logits to probabilities
    confidence, predictions = torch.max(probabilities, dim=1)  # Get the most likely class and its confidence

    sentiments = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    sentiment = sentiments[predictions.item()]

    # Convert probabilities to percentages
    probabilities = probabilities.detach().numpy()[0]
    prob_percentages = {sentiments[i]: f"{probabilities[i] * 100:.2f}%" for i in range(len(sentiments))}

    return sentiment, prob_percentages


def main():
    tf.config.list_physical_devices('GPU')
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')

    choice = input("Enter 'train' to train the model or 'run' to predict sentiment: ")
    if choice == 'train':
        train_model()
    elif choice == 'run':
        text = input("Enter the text to analyze: ")
        sentiment, prob_percentages = run_model(text)
        print(f"Sentiment: {sentiment}")
        for sentiment, percentage in prob_percentages.items():
            print(f"{sentiment}: {percentage}")

if __name__ == "__main__":
    main()
