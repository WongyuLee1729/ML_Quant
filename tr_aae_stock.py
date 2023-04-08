
import pandas as pd
import numpy as np 
import torch
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_context(context="talk")
import argparse
import os
# import math
import itertools
# import torchvision.transforms as transforms
# from torchvision.utils import save_image

# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
GICS - (CoSe, CoDi, CoSt, Ener, Fina, HeCa, Indu, InTe, Mate, ReEs, Util, 총 11개) 
Take the 10 ten based on market cap for training 
Test the other stocks and check the clustering result 

- financial crisis: 2007-01-01 ~ 2010-01-01 finC
- Covid: 2019-11-01 ~ 최근         Covid
- Bull period: 2017-01-01 ~ 2020-01-01    Bull
- Box period: 2014-01-01 ~ 2017-01-01    Box

'''
def first_pop(df1, df2, key:str):

    name = df1.iloc[0]
    tmp = df2.loc[df2[key] ==name].index.values[0]
    df2 = df2.drop(tmp)
    return name, df2


df = pd.read_csv('data.csv', index_col = 0)
df.index = pd.to_datetime(df.index)

gics = pd.read_csv('data.csv')

market_cap = pd.read_csv('data.csv', usecols=['Ticker','Market Cap'])

gics_list = {}
for ix in gics['Sector'].unique():
    if " " in ix:
        pre, post = ix.split()
        abb = pre[:2] + post[:2]
    else:
        abb = ix[:4]
    gics_list[ix] = abb

tic = []
sec = []
# mul_idx = {}
for t, s in gics.values:
    tic.append(t)
    sec.append(gics_list[s])

mul_idx = pd.MultiIndex.from_arrays([sec,tic], names=['GICS','Ticker'])

data = pd.DataFrame(df.values, index=df.index , columns= mul_idx)

# finC -> Training , Covid -> Testing

finc = data.loc[((data.index.year >= 2007) & (data.index.year <= 2010)),:]
finc = finc.dropna(axis=1)
covid = data.loc[((data.index.year > 2019) & (data.index.year <= 2021)),:]
covid = covid.dropna(axis=1)
# data.columns.get_level_values(0)
# global finc, market_cap


def z_score(lst):
    normalized = []
    df = pd.DataFrame()
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        # normalized.append(normalized_num)
        df = pd.concat([df,normalized])
    return normalized

# x = x.apply(stats.zscore,axis=0).values

def preprocess(flag):
    global finc, market_cap
    tick_size = 200
    data = []
    # mx = MinMaxScaler() # for mxscaler
    for g in finc.columns.get_level_values(0).unique(): # market cap ranking calc
        sec = finc[g].columns.values
        ticks = market_cap.where(market_cap['Ticker'].isin(sec)).dropna()
        ticks = ticks.sort_values(by=['Market Cap'],ascending=False)
        ticks = ticks['Ticker'] #[0:10] 
        if len(ticks) <=tick_size and flag == 0: # whenever the flag= 0 produce sector based global min  
            tick_size = len(ticks)
        
        ticks, market_cap = first_pop(ticks, market_cap,'Ticker')
        ticks = finc.loc[:, (finc.columns.get_level_values('Ticker').isin([ticks]))]
        price = np.log(ticks.pct_change()+1)
        # price = ticks.rolling(window =3).mean()
        price = price.fillna(0) # price = price.dropna()
        price = price.apply(stats.zscore,axis=0)
        # mx.fit(price) # for mxscaler
        # price =mx.transform(price) # for mxscaler
        # plt.plot(price)
        data.append(price+20)
    
    data = np.array(data).reshape(len(data),len(data[0]))
    data = torch.Tensor(data).unsqueeze(-2)
    if flag == 0:
        return tick_size, data
    else:
        return data

data = []
tik_sze, tmp_dat =preprocess(0)
data.append(tmp_dat)
for _ in range(tik_sze-1):
    data.append(preprocess(1))

# plt.plot(data[0].squeeze(-2).detach().numpy())

#%%


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=11, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=11, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--d_size", type=int, default=1008, help="feature size of data")
parser.add_argument("--d_model", type=int, default=1008, help="feature size of data")
parser.add_argument("--num_head", type=int, default=8, help="number of heads for attention")
parser.add_argument("--num_layers", type=int, default=2, help="number of layers for transformer block")



opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size) # (1, 32, 32) -> (1,1007)

cuda = True if torch.cuda.is_available() else False

# =============================================================================
# Encoder
# =============================================================================
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# class TrBlock(nn.Module): # TrBlock(TrBlockLayer(d_model, attntion, attntion, dropout), layer_num)
#     "Core TrBlock is a stack of N layers"
#     def __init__(self, layer, N):
#         super(TrBlock, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)
        
#     def forward(self, x, mask):
#         "Pass the input (and mask) through each layer in turn."
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
# That is, the output of each sub-layer is LayerNorm(x+Sublayer(x)) 
# where Sublayer(x) is the function implemented by the sub-layer itself. 
# We apply dropout to the output of each sub-layer, before it is added to the sub-alyer input and normalized 

class ResidualConn(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualConn, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, 
# position-wise fully connected feed-forward network.

class TrBlockLayer(nn.Module):
    "TrBlock is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TrBlockLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualConn(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# =============================================================================
#  Attention
# =============================================================================

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# =============================================================================
# Position-wise Feed-Forward & Embeddings and positional encoding
# =============================================================================

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(d_model, vocab) # self.lut ->Embedding(1008, 512)    , x.shape  -> torch.Size([11, 1008])
        self.d_model = d_model

    def forward(self, x):
        # x = x.type(torch.long)
        return self.lut(x) * math.sqrt(self.d_model) # x -> torch.Size([11, 1007])  ,  self.lut(x).shape -> torch.Size([11, 1007, 512])
        # self.lut(x).shape -> [11, 1007, 1006]    nn.Sequential(Embeddings(d_model= 512, src_vocab= 128), c(position))

class PositionalEncoding(nn.Module): # done 
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # [5000, 512]
        position = torch.arange(0, max_len).unsqueeze(1) # [5000, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *-(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# =============================================================================
# AAE
# =============================================================================
# shifted residual connection 

# model의 구조를 설계하는 단계 
class TrBlock(nn.Module): # TrBlock(TrBlockLayer(d_model, attention, attention, dropout), layer_num)
    "Core TrBlock is a stack of N layers"
    def __init__(self, layer, N):
        super(TrBlock, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



def reparameterization(mu, logvar): # x = mu + sigma^2 * noise
    '''
    stochastic node를 stochastic한 부분과 deterministic한 부분으로 분해시켜 deterministic한 부분으로 
    backpropagation을 흐르게 하려는것이 핵심
    '''
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 256))))
    z = sampled_z * std + mu
    return z

class Compressor(nn.Module):
    def __init__(self,):
        super(Compressor,self).__init__()
        self.mu = nn.Linear(opt.d_size*opt.d_model, 256)
        self.logvar = nn.Linear(opt.d_size*opt.d_model, 256)
        
    def forward(self, tr_out): # tr_output = model output    
        x = torch.reshape(tr_out,(opt.latent_dim,opt.d_size*opt.d_model)) 
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z

class Decompressor(nn.Module):
    def __init__(self,):
        super(Decompressor,self).__init__()
        self.ff = nn.Linear(opt.latent_dim,256) # at z expand the size using ffnn and feed it to tr block
        
    def forward(self, data):
        ff = self.ff(data)
        img = ff.view(ff.shape[0], *(1,1006)) # * <- unpacking  print(*img_shape) -> 1 32 32
        return img

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256, opt.d_model),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.d_model, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, opt.d_model),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *(1,opt.d_model)) # * <- unpacking  print(*img_shape) -> 1 32 32
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(), # In order to use BCELoss the range has to be between 0~1 -> Use Sigmoid
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss() # L=w*[y*logx+(1-y)*log(1-x)]
pixelwise_loss = torch.nn.L1Loss()


class EncoderDecoder(nn.Module):
    """
    Base for this and many 
    other models.
    """
    def __init__(self, trblock, src_embed, compress, decompress):
        super(EncoderDecoder, self).__init__()
        self.trblock = trblock
        self.src_embed = src_embed
        self.compress = compress
        self.decompress = decompress
    def forward(self, src, src_mask):        # src.shape -> torch.Size([30, 10]) , src_mask -> torch.Size([30, 1, 10])
        return self.trblock(self.src_embed(src), src_mask) 


# Initialize generator and discriminator

def set_model(src_vocab, N=opt.num_layers, d_model=opt.d_model, d_ff=2048, h=opt.num_head, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        TrBlock(TrBlockLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)), 
        # Embeddings와 positonalEncoding을 nn.Sequential에 넣어 순차적 연산이 되도록 함
        Compressor(),
        Decompressor())
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg. 
    # initialize model parameters looping by layer 
    for p in model.parameters(): # [512, 512]->[512]->[512,512] ...
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# Initialize generator and discriminator
encoder = set_model(opt.d_model) # self.src_mask = (src != pad).unsqueeze(-2) # (src != pad)  
# con_vec = Context_vec()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()
    
# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
) # print(list(itertools.chain(a,b)))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



# ----------
#  Training
# ----------

'''
GAN에서는 alternating SGD를 이용하여 2 단계학습이 이루어짐 
1. 판별자가 생성자로부터 생성된 가짜 sample로부터 진짜를 구별하는 방법을 학습하고, 
2. 생성자는 Sample을 통해 판별자를 속이는 방법을 학습함
따라서 
1. 먼저 AE의 reconstruction error를 이용해 loss를 업데이트 (theta, phi) => -E[log(p)theta(x_i|z)] 
2. GAN의 Discriminator를 업데이트 (lambda) => -log(d_lambda(z_i))-log(1-d_lambda(q_phi(x_i)))
3. 마지막으로 GAN의 generator를 업데이트 (phi) => -log(d_lambda(q_phi(x_i)))
'''
def visualization(real_img, pred_img, batches_done):
    x = np.arange(real_img.shape[-1]) #.reshape(1,real_img.shape[1])
    f, a = plt.subplots(11,2, figsize=(20,20))
    for i in range(11):
        img = real_img.detach().numpy().reshape(11,opt.d_model)[i] 
        a[i][0].plot(x, img, color='blue')
    a[0,0].title.set_text('real images')
    for j in range(11):
        img = pred_img.detach().to("cpu").numpy().reshape(11,opt.d_model)[j]
        a[j][1].plot(x, img, color='red')
    a[0,1].title.set_text('decoded images')
    
    f.savefig("images/{}.png".format(batches_done))

    
pad = 0

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(data): # imgs.shape = [64,1,32,32]

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        real_imgs = Variable(imgs.type(torch.IntTensor))
        real_imgs = real_imgs.squeeze(-2)
        # -----------------
        #  Train Generator
        # -----------------
        
        optimizer_G.zero_grad()
        en_mask = (real_imgs.squeeze(-2) != pad).unsqueeze(-2)
        # src_mask = 6
        encoded_imgs = encoder(real_imgs, en_mask)
        z_vector = encoder.compress(encoded_imgs)
        # de_mask = (z_vector.squeeze(-2) != pad).unsqueeze(-2)
        decoded_imgs = decoder(z_vector)
        decoded_imgs = decoded_imgs.reshape((11,opt.d_model))
        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(z_vector), valid) + 0.999 * pixelwise_loss(decoded_imgs, real_imgs) # recon error
        # 1. Reconstruction error 

        g_loss.backward() 
        optimizer_G.step() 

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0],256)))) 
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid) # 2. Discriminator update for lambda weights
        fake_loss = adversarial_loss(discriminator(z_vector.detach()), fake) # 3. Generator update for phi weights
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, 16, d_loss.item(), g_loss.item())
        )

        batches_done = epoch * 16 + i
        if batches_done % opt.sample_interval == 0:
            visualization(real_imgs, decoded_imgs, batches_done)
        #     sample_image(n_row=10, batches_done=batches_done)





