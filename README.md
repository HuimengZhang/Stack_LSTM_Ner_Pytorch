# transition-joint-tagger

This is a code based on the model proposed by [Meishan Zhang](https://zhangmeishan.github.io/ChineseLexicalProcessing.pdf).

## Installation

For training, a GPU is strongly recommended for speed. CPU is supported but training could be extremely slow.

### PyTorch

The code is based on PyTorch 0.3.0. You can find installation instructions [here](http://pytorch.org/). 

## Data

We mainly focus on 5 datasets, including CTB5, CTB6, CTB7, PKU and NCC. 

## Training 

To train a tagger model, simpliy run ```train.py``` with the following parameters:
```
--rand_embedding      # use this if you want to randomly initialize the embeddings
--char_emb_file       # file dir for character embedding
--bichar_emb_file     # file dir for bi-character embedding
--word_file     	  # file dir for word embedding
--train_file		  # path to training file
--dev_file		  	  # path to development file
--test_file		  	  # path to test file
--gpu 				  # gpu id, set to -1 if use cpu mode
--batch_size  		  # batch size, default=16')
--checkpoint 		  # path to checkpoint and saved model
```

## Decoding

To tag a raw file, simpliy run ```predict.py``` with the following parameters:
```
--load_arg      	  # path to saved json file with all args
--load_check_point    # path to saved model
--test_file     	  # path to test file 
--test_file_out   	  # path to test file output
--batch_size  		  # batch size, default=16')
```

## Performance 

#### Meishan's paper

dataset | SEG | POS |
---|---|---|
CTB50 | 98.50 | 94.95 |
CTB60 | 96.36 | 92.51 | 
CTB70 | 96.25 | 91.87 | 
PKU | 96.35 | 94.14 |
NCC | 95.30 | 90.42 |

#### Ours results

dataset | SEG | POS |
---|---|---|
CTB50 | |
CTB60 | |
CTB70 | |
PKU | |
NCC | |
 