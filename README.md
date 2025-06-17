# Project Title : Detecting Fake Job Postings using Transformer Models

This project is a part of the MS AAI-510 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

### Project Status: [In Progress]

## Installation

Launch Jupyter notebook and open the `FinalProjectSection5 - Team4.ipynb` file(s) from this repository. 

## Required libraries to be installed including:

    import re
    import torch
    import numpy as np
    from tqdm import tqdm
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from scipy.sparse import hstack
    from transformers import AutoTokenizer, AutoModel
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    from wordcloud import WordCloud
    from torch.utils.data import Dataset, DataLoader
    from transformers import DistilBertForSequenceClassification
    from torch.optim import AdamW
    from transformers import Trainer, TrainingArguments
    from datasets import load_metric
    from torch.nn import CrossEntropyLoss
    from torch import tensor

  
## Project Intro/Objective

There are an increasing number of fake jobs getting posted and job seekers are getting scammed into entering their application details into fake websites.

### Partner(s)/Contributor(s)

•	Donavan Trigg

•	Matthew Ongcapin

•	Mustafa Yunus


### Methods Used

•	Machine Learning

•	Neural Networks

•	Deep Learning


### Technologies

•	Python

•	Jupyter Notebook

•	PyTorch


### Project Description

TBD
