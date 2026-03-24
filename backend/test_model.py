from transformers import pipeline

pipe = pipeline("sentiment-analysis")
print(pipe("This project is amazing"))