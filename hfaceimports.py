
#IMPORTS

import pandas as pd
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import zipfile as zp, pickle as pkl,os
from functools import reduce
from itertools import product
from datasets import load_dataset, DatasetDict, Dataset
import textwrap as twp, re
import numpy as np
import torch
from datetime import datetime
from functools import partial
# import zenml as zml
subdict = lambda keys: lambda d: {k:d[k] for k in keys}
subdset = lambda dset_, n: Dataset.from_dict({k:dset_[k][:n] for k in dset_.column_names})
# pd.Series(range(20)).plot()
from collections import defaultdict, Counter
import json
from matplotlib import pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, DistilBertForSequenceClassification
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
from transformers import DistilBertTokenizer, DistilBertTokenizerFast, AutoTokenizer
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased",truncation_side = 'left')      # written in Python
# print(tokenizer)
# tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased",truncation_side = 'left')  # written in Rust
# print(tokenizer)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased",truncation_side = 'left') # convenient! Defaults to Fast
print(tokenizer)
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

# FORWARD PASS

def forward_pass_(batch,
                            device='cpu',
                            model=model,
                            loss=None):
    # Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        if loss: # batch must have labels with key 'label'
            loss = cross_entropy(output.logits, batch["label"].to(device),
        reduction="none")
    # Place outputs on CPU for compatibility with other dataset columns
    out = {"predicted_label": pred_label.cpu().numpy()}
    if loss is not None: out.update({"loss": loss.cpu().numpy()})
    return out
make_forward_pass = lambda model, device = 'cpu',loss = True: partial(forward_pass_,device=device,model=model,loss=loss)
# forward_pass = make_forward_pass(allocation_model)
# forward_pass()

# CONFUSION MATRIX

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def plot_confusion_matrix(y_preds, y_true, labels,add2title = None):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    title = "Normalized confusion matrix"
    if add2title: title = ' '.join((title,add2title))
    plt.title(title)
    
# DATASET CONFIGURATION

config_dataset = dict(
   function = lambda example: tokenizer(example['line'], padding= True,truncation=True),
   batched=True,
   batch_size=16)
forward_pass = None # MUST PROVIDE before runtime
config_forward_pass = dict(
    function = forward_pass,#_with_label,
    batched=True,
    batch_size=16)
# totmbsargs[datatype] = totmbsargs[datatype].map(**config_forward_pass)

# MISCELLANEOUS

inputkeys = 'input_ids attention_mask'.split()
inputkeys
inputdict = subdict(inputkeys)
inputkeys