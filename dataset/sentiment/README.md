---
annotations_creators:
- other
language:
- en
language_creators:
- other
license:
- mit
multilinguality:
- monolingual
pretty_name: twitter financial news
size_categories:
- 10K<n<100K
source_datasets:
- original
tags:
- twitter
- finance
- markets
- stocks
- wallstreet
- quant
- hedgefunds
- markets
task_categories:
- text-classification
task_ids:
- multi-class-classification
---

### Dataset Description

The Twitter Financial News dataset is an English-language dataset containing an annotated corpus of finance-related tweets. This dataset is used to classify finance-related tweets for their sentiment.

1. The dataset holds 11,932 documents annotated with 3 labels:

```python
sentiments = {
    "LABEL_0": "Bearish", 
    "LABEL_1": "Bullish", 
    "LABEL_2": "Neutral"
}  
```

The data was collected using the Twitter API. The current dataset supports the multi-class classification task.

### Task: Sentiment Analysis

# Data Splits
There are 2 splits: train and validation. Below are the statistics:

| Dataset Split | Number of Instances in Split                |
| ------------- | ------------------------------------------- |
| Train         | 9,938                                       |
| Validation    | 2,486                                       |


# Licensing Information
The Twitter Financial Dataset (sentiment) version 1.0.0 is released under the MIT License.