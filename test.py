import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns

if torch.cuda.is_available():       
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data = pd.read_csv("./openstack/openstack_full.csv", index_col=None)
model = torch.load("model_saved/hdfs.pth")

convert_categorical = {"Label": {"Anomaly": 1, "Normal": 0}}
data = data.replace(convert_categorical)

templates = data.Template.values
params = data.Parameter.values
labels = data.Label.values

input_ids = []
attention_masks = []

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

for temp in templates:
    encoded_dict = tokenizer.encode_plus(
                        temp,                      
                        add_special_tokens = True, 
                        max_length = 64, 
                        truncation = True,
                        pad_to_max_length = True,
                        return_attention_mask = True, 
                        return_tensors = 'pt',
                   )    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)
 
batch_size = 32  

prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

model.eval()

predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  batch = tuple(t.to(device) for t in batch)
  
  b_input_ids, b_input_mask, b_labels = batch
  
  with torch.no_grad():
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # move to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  predictions.append(logits)
  true_labels.append(label_ids)

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

flat_true_labels = np.concatenate(true_labels, axis=0)

from sklearn.metrics import classification_report
cm = classification_report(flat_true_labels, flat_predictions)
print(cm)
