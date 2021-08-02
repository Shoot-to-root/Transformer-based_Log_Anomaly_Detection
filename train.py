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

#data = pd.read_csv("./HDFS/final_HDFS.csv", index_col=None)
data = pd.read_csv("./openstack/openstack_full.csv", index_col=None)
# combine HDFS and openstack
#data = pd.concat([data1, data2], axis=0)

convert_categorical = {"Label": {"Anomaly": 1, "Normal": 0}}
data = data.replace(convert_categorical)
#print(data.head())
#print(data.describe())
#print(data.isnull().sum())

templates = data.Template.values
params = data.Parameter.values
labels = data.Label.values

# tokenization
temp_ids = []
param_ids = []
temp_attention_masks = []
param_attention_masks = []

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
    temp_ids.append(encoded_dict['input_ids'])
    temp_attention_masks.append(encoded_dict['attention_mask'])
    
for param in params:
    encoded_dict = tokenizer.encode_plus(
                        param,                      
                        add_special_tokens = True, 
                        max_length = 64, 
                        truncation = True,
                        pad_to_max_length = True,
                        return_attention_mask = True, 
                        return_tensors = 'pt',
                   )
    param_ids.append(encoded_dict['input_ids'])
    param_attention_masks.append(encoded_dict['attention_mask'])

# convert to pytorch tensors
temp_ids = torch.cat(temp_ids, dim=0)
param_ids = torch.cat(param_ids, dim=0)
temp_attention_masks = torch.cat(temp_attention_masks, dim=0)
param_attention_masks = torch.cat(param_attention_masks, dim=0)
labels = torch.tensor(labels)

# concatenate templates and parameters
combined_ids = torch.cat((temp_ids, param_ids), 1)
combined_attention_masks = torch.cat((temp_attention_masks, param_attention_masks), 1)

dataset = TensorDataset(combined_ids, combined_attention_masks, labels)

# train test split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('training samples {}'.format(train_size))
print('validation samples {}'.format(val_size))

# create dataloader
batch_size = 16

train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size 
        )

validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            batch_size = batch_size 
        )

# initialize transformer with BERT
# BertForSequenceClassification is a transformer with a classification/regression head on top,
# attention scores calculated in previous layers will be used to perform classification
# config {
#    "attention_probs_dropout_prob": 0.1,
#    "hidden_act": "gelu",
#    "hidden_dropout_prob": 0.1,
#    "initializer_range": 0.02,
#    "max_position_embeddings": 512,
#    "num_attention_heads": 12,
#    "num_hidden_layers": 12
#    }
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = 2,    
    output_attentions = False, 
    output_hidden_states = False,
    return_dict = False
    )

# run on GPU
model.cuda()

optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

epochs = 10

# total training steps = number of batches * number of epochs
total_steps = len(train_dataloader) * epochs

# create learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

seed_val = 42

stats = []

total_t0 = time.time()

for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        # every 40 batches, print progress
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # unpack training data
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # clear calculated gradients before backward pass
        optimizer.zero_grad()        

        # forward pass
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        # calculate loss
        total_train_loss += loss.item()

        # backward pass
        loss.backward()

        # clip the norm of gradients to 1.0 to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # Update learning rate
        scheduler.step()

    # calculate average loss
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    training_time = format_time(time.time() - t0)

    print("")
    print("Average training loss: {0:.2f}".format(avg_train_loss))
    print("Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        
        # unpack data
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():   
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # calculate accuracy 
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # calculate average accuracy
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("Accuracy: {0:.2f}".format(avg_val_accuracy))

    # calculate average loss
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)
    
    print("Validation Loss: {0:.2f}".format(avg_val_loss))
    print("Validation took: {:}".format(validation_time))

    # record statistics
    stats.append(
        {
            'Epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training time {:}".format(format_time(time.time()-total_t0)))

pd.set_option('precision', 2)
df_stats = pd.DataFrame(data=stats)
df_stats = df_stats.set_index('Epoch')

# plot the learning curve
sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# save model
torch.save(model, "model_saved/hdfs.pth")
#torch.save(model, "model_saved/openstack.pth")
#torch.save(model, "model_saved/combined.pth")
