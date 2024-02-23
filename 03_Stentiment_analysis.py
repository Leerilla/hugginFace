from transformers import pipeline

classifier = pipeline("sentiment-analysis")

results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it.","It’s snowing today so I’m sleepy"])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
