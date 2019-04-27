# ### Conll 2003 evaluation
# Data downloaded from [here](https://github.com/kyzhouhzau/BERT-NER/tree/master/NERdata).

import torch
import pandas as pd
import warnings
import os
import sys
import codecs
import re
import pickle

from modules import NerLearner
from modules import BertNerData as NerData
from modules.models.bert_models import BertBiLSTMAttnCRF

from sparsity_pruning import SparsityPruner

def param_count(model):
    """ Sum of l0 norm across all model params """
    total = 0
    for name,param in model.named_parameters():
        total += torch.norm(param, p=0).item()
    return total

def parse_f1(result):
    rows = result.split('\n')
    mat = list(map(lambda r: r.split('     '), rows))
    f1 = float(mat[-2][-2])
    return f1


def read_data(input_file):
    """Reads a BIO data."""
    with codecs.open(input_file, "r", encoding="utf-8") as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue
            
            if len(contends) == 0 and not len(words):
                words.append("")
            
            if len(contends) == 0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label.replace("-", "_"))
        return lines

def select_params(model, keyword, enc=True):
    matches = []
    for name,param in model.named_parameters():
        if 'bias' in name or 'gamma' in name or 'beta' in name:
            continue
        if enc and 'decoder' in name:
            continue
        if not enc and 'encoder' in name:
            continue
        if re.match(keyword,name):
            matches.append(name)
    return matches



# Load training data and BERT Multilingual fitted checkpoint

sys.path.append("../")
warnings.filterwarnings("ignore")
data_path = "/home/ec2-user/datadrive/conll-2003/"
train_path = data_path + "train.txt"
dev_path = data_path + "dev.txt"
test_path = data_path + "test.txt"


train_f = read_data(train_path)
dev_f = read_data(dev_path)
test_f = read_data(test_path)

train_df = pd.DataFrame(train_f, columns=["0", "1"])
train_df.to_csv(data_path + "train.csv", index=False)

valid_df = pd.DataFrame(dev_f, columns=["0", "1"])
valid_df.to_csv(data_path + "valid.csv", index=False)

test_df = pd.DataFrame(test_f, columns=["0", "1"])
test_df.to_csv(data_path + "test.csv", index=False)

data_path = "/home/ec2-user/datadrive/conll-2003/"
train_path = data_path + "train.csv"
valid_path = data_path + "valid.csv"
test_path = data_path + "test.csv"

model_dir = "/home/ec2-user/datadrive/models/multi_cased_L-12_H-768_A-12/"
init_checkpoint_pt = os.path.join("/home/ec2-user/datadrive/bert/multi_cased_L-12_H-768_A-12/", "pytorch_model.bin")
bert_config_file = os.path.join("/home/ec2-user/datadrive/bert/multi_cased_L-12_H-768_A-12/", "bert_config.json")
vocab_file = os.path.join("/home/ec2-user/datadrive/bert/multi_cased_L-12_H-768_A-12/", "vocab.txt")

# Instantiate model & set cuda device

torch.cuda.set_device(0)
print(torch.cuda.is_available(), torch.cuda.current_device())


data_train = NerData.create(train_path, valid_path, vocab_file)
sup_labels = ['B_ORG', 'B_MISC', 'B_PER', 'I_PER', 'B_LOC', 'I_LOC', 'I_ORG', 'I_MISC']

# Create model
model = BertBiLSTMAttnCRF.create(len(data_train.label2idx), bert_config_file, init_checkpoint_pt, enc_hidden_dim=256)
torch.save(model.state_dict(), open('/home/ec2-user/datadrive/models/conll-2003/lottery-ticket-init-2.cpt', 'wb'))

# Define param groups for sensitivity analysis
sens_embedding = select_params(model, 'encoder.*embedding.*word_embeddings')
sens_intermediate = select_params(model, 'encoder.*intermediate')
sens_attn_query = select_params(model, 'encoder.*attention.*query')
sens_attn_key = select_params(model, 'encoder.*attention.*key')
sens_attn_value = select_params(model, 'encoder.*attention.*value')
sens_attn_output = select_params(model, 'encoder.*attention.*output.*')
sens_out_dense = list(set(select_params(model, 'encoder.*output.*dense')) - set(select_params(model, 'encoder.*attention.*output.*dense')))
sens_dec_linear = select_params(model, 'decoder.linear*', enc=False)
sens_dec_attn = select_params(model, 'decoder.attn*', enc=False)

sens_attn = sens_attn_key + sens_attn_output + sens_attn_query + sens_attn_value
sens_enc = sens_embedding + sens_intermediate + sens_out_dense + sens_attn
sens_dec = sens_dec_attn + sens_dec_linear


"""
Experiment outline:

1. Train original BERT Base (Multilingual, Uncased) for 25 epochs
2. Pruning schedule: [0.625**(i+1) for i in range(7)]
3. Prune, fine-tune for 15 epochs

"""

# ### 2a. Pruning/Sensitivity Analysis
# TODO: Set this to actual training params after checking that saving works
num_epochs = 25
num_ft_epochs = 10
prune_iter = 10

learner = NerLearner(model, data_train,
                     best_model_path="/home/ec2-user/datadrive/models/conll-2003/other-2.cpt",
                     lr=0.001, clip=1.0, sup_labels=data_train.id2label[5:],
                     t_total=num_epochs * len(data_train.train_dl))


from modules.data.bert_data import get_bert_data_loader_for_predict
from modules.train.train import validate_step
dl = get_bert_data_loader_for_predict(data_path + "valid.csv", learner)
data_valid = NerData.create(train_path, data_path + "valid.csv", vocab_file)

torch.cuda.empty_cache()
fit_history = {}

lr_epoch = 2
print('Training base model: BERT_BASE (MULTI)')
learner.fit(lr_epoch, target_metric='f1', pruner=None)
torch.save(model.state_dict(), open('/home/ec2-user/datadrive/models/conll-2003/lottery-ticket-lr-2.cpt', 'wb'))
learner.fit(num_epochs-lr_epoch, target_metric='f1', pruner=None)
fit_history['init'] = learner.history
fit_history['init_pct'] = 1.
torch.save(model.state_dict(), open('/home/ec2-user/datadrive/models/conll-2003/lottery-ticket-trained-2.cpt', 'wb'))


# Pruners - different pruning schedule for each layer
pruner_embed = SparsityPruner(model, sens_embedding)
pruner_dense = SparsityPruner(model, sens_intermediate + sens_out_dense)
pruner_attn = SparsityPruner(model, sens_attn)

base_size = param_count(model)

for prune_idx in range(prune_iter):
    print('Starting pruning iteration {}'.format(prune_idx+1))
    # Eagerly evict cache
    torch.cuda.empty_cache()

    pruner_embed.prune(0.625)
    pruner_dense.prune(0.625)
    pruner_attn.prune(0.9)
    
    new_size = param_count(model)

    learner_ft = NerLearner(model, data_train,
                        best_model_path="/home/ec2-user/datadrive/models/conll-2003/ft-iter-{}-2.cpt".format(prune_idx+1),
                        lr=0.001, clip=1.0, sup_labels=data_train.id2label[5:],
                        t_total=num_ft_epochs * len(data_train.train_dl))


    learner_ft.fit(num_ft_epochs, target_metric='f1', pruner=[pruner_embed, pruner_dense, pruner_attn])

    # Save pruners & checkpoint model
    embed_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_embed_{}.p'.format(prune_idx+1)
    dense_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_dense_{}.p'.format(prune_idx+1)
    attn_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_attn_{}.p'.format(prune_idx+1)
    pickle.dump(pruner_embed, open(embed_path, 'wb'))
    pickle.dump(pruner_dense, open(dense_path, 'wb'))
    pickle.dump(pruner_attn, open(attn_path, 'wb'))

    pct_retained = new_size/base_size
    fit_history['late_reset_lt_{}_pct'.format(prune_idx+1)] = pct_retained

    print('Retained {}% of parameters'.format(new_size/base_size))

# Retrain from "late reset"
for prune_idx in range(prune_iter):
    print('Retraining from late reset initialization at iter {}'.format(prune_idx+1))
    # Eagerly evict cache
    torch.cuda.empty_cache()

    # Reload checkpoint & pruners for prune_idx level
    trained_path = '/home/ec2-user/datadrive/models/conll-2003/lottery-ticket-lr-2.cpt'
    model.load_state_dict(torch.load(trained_path))

    embed_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_embed_{}.p'.format(prune_idx+1)
    dense_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_dense_{}.p'.format(prune_idx+1)
    attn_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_attn_{}.p'.format(prune_idx+1)

    pruner_embed = pickle.load(open(embed_path, 'rb'))
    pruner_dense = pickle.load(open(dense_path, 'rb'))
    pruner_attn = pickle.load(open(attn_path, 'rb'))

    pruner_embed.apply_mask()
    pruner_dense.apply_mask()
    pruner_attn.apply_mask()

    learner_retrain = NerLearner(model, data_train,
                        best_model_path="/home/ec2-user/datadrive/models/conll-2003/late-reset-iter-{}-2.cpt".format(prune_idx+1),
                        lr=0.001, clip=1.0, sup_labels=data_train.id2label[5:],
                        t_total=num_epochs * len(data_train.train_dl))
    learner_retrain.fit(num_epochs, target_metric='f1', pruner=[pruner_embed, pruner_dense, pruner_attn])
    fit_history['late_reset_lt_{}'.format(prune_idx+1)] = learner_retrain.history

# Retrain from "winning ticket initialization"
for prune_idx in range(prune_iter):
    print('Retraining from winning ticket initialization at iter {}'.format(prune_idx+1))
    # Eagerly evict cache
    torch.cuda.empty_cache()

    # Reload checkpoint & pruners for prune_idx level
    trained_path = '/home/ec2-user/datadrive/models/conll-2003/lottery-ticket-init-2.cpt'
    model.load_state_dict(torch.load(trained_path))

    embed_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_embed_{}.p'.format(prune_idx+1)
    dense_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_dense_{}.p'.format(prune_idx+1)
    attn_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_attn_{}.p'.format(prune_idx+1)

    pruner_embed = pickle.load(open(embed_path, 'rb'))
    pruner_dense = pickle.load(open(dense_path, 'rb'))
    pruner_attn = pickle.load(open(attn_path, 'rb'))

    pruner_embed.apply_mask()
    pruner_dense.apply_mask()
    pruner_attn.apply_mask()

    learner_retrain = NerLearner(model, data_train,
                        best_model_path="/home/ec2-user/datadrive/models/conll-2003/reinit-iter-{}-2.cpt".format(prune_idx+1),
                        lr=0.001, clip=1.0, sup_labels=data_train.id2label[5:],
                        t_total=num_epochs * len(data_train.train_dl))
    learner_retrain.fit(num_epochs, target_metric='f1', pruner=[pruner_embed, pruner_dense, pruner_attn])
    fit_history['reinit_lt_{}'.format(prune_idx+1)] = learner_retrain.history

# Retrain from "winning ticket initialization"
for prune_idx in range(prune_iter):
    print('Retraining from random initialization at iter {}'.format(prune_idx+1))

    # Reload pruners for prune_idx level and random reinit
    model = BertBiLSTMAttnCRF.create(len(data_train.label2idx), bert_config_file, init_checkpoint_pt, enc_hidden_dim=256)
    # Eagerly evict cache
    torch.cuda.empty_cache()

    embed_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_embed_{}.p'.format(prune_idx+1)
    dense_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_dense_{}.p'.format(prune_idx+1)
    attn_path = '/home/ec2-user/datadrive/models/conll-2003/pruner_attn_{}.p'.format(prune_idx+1)

    pruner_embed = pickle.load(open(embed_path, 'rb'))
    pruner_dense = pickle.load(open(dense_path, 'rb'))
    pruner_attn = pickle.load(open(attn_path, 'rb'))

    pruner_embed.apply_mask()
    pruner_dense.apply_mask()
    pruner_attn.apply_mask()

    learner_retrain = NerLearner(model, data_train,
                        best_model_path="/home/ec2-user/datadrive/models/conll-2003/random-iter-{}-2.cpt".format(prune_idx+1),
                        lr=0.001, clip=1.0, sup_labels=data_train.id2label[5:],
                        t_total=num_epochs * len(data_train.train_dl))
    learner_retrain.fit(num_epochs, target_metric='f1', pruner=[pruner_embed, pruner_dense, pruner_attn])
    fit_history['random_lt_{}'.format(prune_idx+1)] = learner_retrain.history

history_path = '/home/ec2-user/datadrive/models/conll-2003/fit_history.p'
pickle.dump(fit_history, open(history_path, 'wb'))
print('Done')
