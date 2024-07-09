import pandas as pd
import numpy as np
import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import sentencepiece as spm
import random
from collections import OrderedDict
from matplotlib import pyplot as plt
import time

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

VOCAB_SIZE = 130
BATCH_SIZE = 32
CONTEXT_WINDOW = 16
EPOCHS = 1000
DIM = 128
LOG_INTERVAL = 10
HEADS = 8
LAYERS = 4

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

if response.status_code == 200:
    tinyshakespeare = response.text
else:
    print(response.status_code)

tinyshakespeare_list = tinyshakespeare.split("\n")
tinyshakespeare_list = [i for i in tinyshakespeare_list if i != ""]

spm.SentencePieceTrainer.Train(
    sentence_iterator = iter(tinyshakespeare_list),
    model_prefix = "tinyshakespeare_model",
    vocab_size = VOCAB_SIZE,
    character_coverage = 1.0,
    model_type = "bpe",
    pad_id = 0,
    unk_id = 1,
    bos_id = 2,
    eos_id = 3,
)

sp = spm.SentencePieceProcessor(model_file = "tinyshakespeare_model.model")
dataset_tensor = torch.tensor(sp.Encode(tinyshakespeare))

def get_batch_train(dataset, batch_size, context_window):
    train_data = dataset[:int(.7 * len(dataset))]
    ix = torch.randint(0, train_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([train_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([train_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y


def get_batch_val(dataset, batch_size, context_window):
    val_data = dataset[int(.7 * len(dataset)): int(.85 * len(dataset))]
    ix = torch.randint(0, val_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([val_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([val_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y

def get_batch_test(dataset, batch_size, context_window):
    test_data = dataset[int(.85 * len(dataset)): len(dataset)]
    ix = torch.randint(0, test_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([test_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([test_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y

@torch.no_grad()
def calculate_loss(model):    
    model.eval()
    train_losses = []
    val_losses = []
    for i in range(EPOCHS):
        #train evaluation
        x_train, y_train = get_batch_train(dataset_tensor, BATCH_SIZE, CONTEXT_WINDOW)
        _, train_loss = model(x_train, y_train)
        train_losses.append(train_loss.item())
        
        #val evaluation
        x_val, y_val = get_batch_val(dataset_tensor, BATCH_SIZE, CONTEXT_WINDOW)
        _, val_loss = model(x_val, y_val)
        val_losses.append(val_loss.item())

    losses_dict = {"train": np.mean(train_losses), "val": np.mean(val_losses)}
    return losses_dict


@torch.no_grad()
def calculate_accuracy(model):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(EPOCHS):
        # Get a batch of validation data
        x_val, y_val = get_batch_val(dataset_tensor, BATCH_SIZE, CONTEXT_WINDOW)
        
        # Get model predictions
        logits = model(x_val)
        
        # Convert predictions to class labels
        predicted_labels = torch.argmax(logits, dim=-1)
        
        # Compare with true labels
        correct_predictions += (predicted_labels == y_val).sum().item()
        total_predictions += y_val.numel()
    
    accuracy = correct_predictions / total_predictions
    return accuracy

@torch.no_grad()
def calculate_perplexity(model):
    model.eval()
    val_losses = []
    
    for i in range(EPOCHS):
        # Get a batch of validation data
        x_val, y_val = get_batch_val(dataset_tensor, BATCH_SIZE, CONTEXT_WINDOW)
        
        # Get model predictions and loss
        _, val_loss = model(x_val, y_val)
        val_losses.append(val_loss.item())
    
    # Calculate the mean validation loss
    mean_val_loss = np.mean(val_losses)
    
    # Perplexity is the exponential of the cross-entropy loss
    perplexity = np.exp(mean_val_loss)
    return perplexity

def train(model, optimizer, checkpoint_path="/checkpoints"):
    losses = []
    accs = []
    perps = []
    for epoch in range(EPOCHS):
        optimizer.zero_grad()        
        x_train, y_train = get_batch_train(dataset_tensor, BATCH_SIZE, CONTEXT_WINDOW)
        logits, loss = model(x_train, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % LOG_INTERVAL == 0:
            current_loss = calculate_loss(model)
            current_accuracy = calculate_accuracy(model)
            current_perplexity = calculate_perplexity(model)

            losses.append(current_loss)
            accs.append(current_accuracy)
            perps.append(current_perplexity)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'accuracy': current_accuracy,
                'perplexity': current_perplexity
            }, f"{checkpoint_path}/checkpoint_epoch_{epoch}.pth")
            
            print(f"Epoch {epoch}: Loss - {current_loss['val']}, Accuracy - {current_accuracy}, Perplexity - {current_perplexity}")


    print("validation Loss: ", losses[-1]['val'])
    print("validation Accuracy: ", accs[-1])
    print("validation Perplexity: ", perps[-1])
    return pd.DataFrame(losses).plot()

class RMSNorm(torch.nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", torch.nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        return self.scale[:x.shape[1], :].unsqueeze(0) * ((torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5).unsqueeze(-1).unsqueeze(-1))
    
def get_rotary_matrix(context_window, embedding_dim):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim//2):
            theta = 10000. ** (-2.*(i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2*i,2*i] = np.cos(m_theta)
            R[position, 2*i,2*i+1] = - np.sin(m_theta)
            R[position, 2*i+1,2*i] = np.sin(m_theta)
            R[position, 2*i+1,2*i+1] = np.cos(m_theta)
    return R


class RoPEAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_q = nn.Linear(DIM, DIM, bias=False)
        self.w_k = nn.Linear(DIM, DIM, bias=False)
        self.w_v = nn.Linear(DIM, DIM, bias=False)

        self.R = get_rotary_matrix(CONTEXT_WINDOW, DIM)

    def get_rotary_matrix(context_window, embedding_dim):
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):
                theta = 10000. ** (-2.*(i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i,2*i] = np.cos(m_theta)
                R[position, 2*i,2*i+1] = - np.sin(m_theta)
                R[position, 2*i+1,2*i] = np.sin(m_theta)
                R[position, 2*i+1,2*i+1] = np.cos(m_theta)
        return R
    
    def forward(self, x, return_attn_weights=False):
        b,m,d = x.shape
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0,1), self.R[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1), self.R[:m])).transpose(0,1)

        activations = F.scaled_dot_product_attention(
            q_rotated,k_rotated,v,dropout_p =.1
        )

        if return_attn_weights:
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations
    
class RoPEAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_q = nn.Linear(DIM, DIM, bias=False)
        self.w_k = nn.Linear(DIM, DIM, bias=False)
        self.w_v = nn.Linear(DIM, DIM, bias=False)

        self.R = get_rotary_matrix(CONTEXT_WINDOW, DIM)

    def get_rotary_matrix(context_window, embedding_dim):
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):
                theta = 10000. ** (-2.*(i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i,2*i] = np.cos(m_theta)
                R[position, 2*i,2*i+1] = - np.sin(m_theta)
                R[position, 2*i+1,2*i] = np.sin(m_theta)
                R[position, 2*i+1,2*i+1] = np.cos(m_theta)
        return R
    
    def forward(self, x, return_attn_weights=False):
        b,m,d = x.shape
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0,1), self.R[:m])).transpose(0,1)
        k_rotated = (torch.bmm(k.transpose(0,1), self.R[:m])).transpose(0,1)

        activations = F.scaled_dot_product_attention(
            q_rotated,k_rotated,v,dropout_p =.1, is_causal=True
        )

        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m,m)), diagonal=0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations
    
class RoPEMultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([
            RoPEAttentionHead() for _ in range(HEADS)
        ])
        self.linear = nn.Linear(HEADS * DIM, DIM)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x): 
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out
    

class LlamaBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.rms = RMSNorm((CONTEXT_WINDOW, DIM))
        
        self.attention = RoPEMultiheadAttention()
        self.feedforward = nn.Sequential(
            nn.Linear(DIM, DIM),
            SwiGLU(DIM),
        )

    def forward(self, x):
        x = self.rms(x)             #RMS NORMALIZATION 
        x = x + self.attention(x)   #Self attention

        x = self.rms(x)             #RMS NORMALIZATION
        x = x + self.feedforward(x) #Feed Foward: SwiGlu
        return x
    
class Llama(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, DIM)
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock()) for i in range(LAYERS)])
        )

        self.ffn = nn.Sequential(
            nn.Linear(DIM, DIM),
            SwiGLU(DIM),
            nn.Linear(DIM, VOCAB_SIZE),
        )

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)

        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            return logits, loss
        

llama = Llama()
optimizer = torch.optim.Adam(llama.parameters())
train(llama, optimizer)