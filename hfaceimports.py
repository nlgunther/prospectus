
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
    
# USEFUL UTILITIES

listify = lambda f: lambda *args,**kwargs: list(f(*args,**kwargs))
mapl = listify(map)
filterl = listify(filter)
reducel = listify(reduce)
mapl(str,range(10))
###############
get_methods = lambda obj: [e for e in dir(obj) if not e[0]=='_']
find_lines = lambda rex,lines,i=None: [l for l in lines if re.search(rex,l[i] if i else l)]
def xgetd(d,args):
    k,args = args[0],args[1:]
    d = d[k]
    if args: return xgetd(d,args)
    else: return d
def find_block_boundary(regexp, ls):
    return filterl(lambda l: re.search(regexp,l[1]) and not len(re.findall('\W',l[1]))>10,enumerate(ls))
beginsection = '^Hypothetical Structure and Calculations with Respect to the Reference Tranches$'
endsection = '^THE AGREEMENTS$;^The Reference Pool$'
def find_starts_ends(ls):
    starts = [tpl for begin in beginsection.split(';') for tpl in find_block_boundary(begin,ls)]
    ends = [tpl for end in endsection.split(';') for tpl in find_block_boundary(end,ls)]
    return starts,ends
def find_textblocks(starts,ends):
    ar = []
    for pair in product(starts, ends):
        # print(pair)
        if (dist:= pair[1][0] - pair[0][0]) > 0:
            ar.append((dist,pair))
        ar.sort()
    return ar
def pairs2dict(pairs):
    ar = []
    for p_ in pairs:
        dist, pair = p_
        ar.append(dict(zip('startstop tags'.split(),zip(*pair))))
    return ar
viewi = lambda span, i,labels,width=5: labels[span[i]-width:span[i]+width]#[:10]
# test =lambda i: any([ pair[0][0] <= i < pair[0][1] for pair in ar ])
get_range_test_from_spans = lambda spans: lambda i: any([ pair[0] <= i < pair[1] for pair in spans ])


# FILE CONFIGURATION

gdrive = r'G:\My Drive'
gfld = r'NLP\prospectus'
fld = os.path.join(gdrive,gfld)
print(fld)

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

# TEXT CLEANING

class TextPipe(object):

    def __init__(self,text,transforms=list()):
        self.text = text
        self.transforms = transforms

    def transform(self):
        for transform in self.transforms:
            self.text = transform(self.text)
    def get_lines(self,strip=True):
        lines =  self.text.split('\n')
        return lines if not strip else [l.strip() for l in lines]

class File2LL(object):

    def __init__(self,path,
                mpl=150,
                mll = 75,
                linemap = None):
        self.path = path
        self.minParaLen = mpl
        self.maxLineLen = mll
        self.linemap = linemap

    def process(self,longlines):
        return longlines
    
    def readfile(self):
        with open(self.path,errors='replace') as f:
            lines = f.readlines()
        if self.linemap: 
            print('modifying lines')
            lines = self.linemap(lines)
        # labeled_para = mapl(lambda l: l.replace('\n','<NL>'),longlines)
        return lines
        

# MISCELLANEOUS

inputkeys = 'input_ids attention_mask'.split()
inputkeys
inputdict = subdict(inputkeys)
inputkeys

# DATE

today_ = datetime.today().date()
today = str(today_).replace('-','_')
today