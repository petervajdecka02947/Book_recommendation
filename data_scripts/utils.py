import seaborn as sns
import matplotlib.pyplot as plt
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#!pip install pyspark
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession
from pyspark.sql.functions import format_number as fmt

def import_csv(col_names,df_dir,encoding='latin-1'):
# import dataframe 
    df=pd.read_csv(df_dir, sep=';', names=col_names, encoding=encoding,low_memory=False)
    return df.drop(df.index[0]) # remove first 

def describe_df(df):
    return display(df.describe()), display( df.head())

def get_empty_byCol(df,column):
        print("Empty rows for column " + column)
        return display(df.loc[df[column].isnull(),:])

def plot_column_distribution(df,column,title):    
        plt.figure(figsize=(10,8))
        sns.countplot(x=column,data=df,palette=sns.color_palette("hls", 8))
        plt.title(title,size=20)
        plt.show() 

# Calculate cosine similarity between two vectors 
def cossim(v1, v2): 
    return np.dot(v1, v2) / np.sqrt(np.dot(v1, v1)) / np.sqrt(np.dot(v2, v2)) 

def get_embeddingsBert(text):
        start = time.time()
        
        text_length=len(text)
        
        if text_length!=1:
            
            model = SentenceTransformer(BERT_small_model_name)  # model  bert-base-uncased 

            #Sentences are encoded by calling model.encode()
            sentence_vectors = model.encode(text)
                    
            
            end = time.time()
            print("Time for creating "+ str(len(sentence_vectors))+" embedding vectors " + str((end - start)/60))
            print('Model used :'+ BERT_small_model_name )
            return sentence_vectors
        
        else:
        
            
            dev = torch.device('cuda') # cuda could be used as well

            # Load pre-trained model tokenizer (vocabulary)
            tokenizer = BertTokenizer.from_pretrained(BERT_huge_model_name,do_lower_case=True,device=dev)

            # cretae secial format for BERT 
            text_tokenized=[tokenizer.tokenize(str("[CLS] " + sentence + " [SEP]")) for sentence in text]

            # create token IDS means  means same sentence gets same IDs, however sentence with same word but diverse others words gets divergent ID
            indexed_tokens = [tokenizer.convert_tokens_to_ids(sentence) for sentence in text_tokenized]

            # Mark each of the tokens as belonging to sentences with different ID.
            segments_ids =[ [i] * len(sentence) for i,sentence in enumerate(text_tokenized) ]

            # Convert inputs to PyTorch tensors
            tokens_tensors = [torch.tensor([indexed_token], device=dev) for indexed_token in indexed_tokens]
            segments_tensors = [torch.tensor([segments_id], device=dev) for segments_id   in segments_ids]

            # Load pre-trained model (weights)
            model = BertModel.from_pretrained(BERT_huge_model_name,
                                              output_hidden_states = True ).to(device=dev) # Whether the model returns all hidden-states.
            # Put the model in "evaluation" mode, meaning feed-forward operation.
            model.eval().to(device=dev)

            # Run the text through BERT, and collect all of the hidden states produced
            # from all 12 layers. 
            with torch.no_grad():

                hidden_states=[model(tokens_tensors[num], segments_tensors[num])[2] for num in range(len(segments_tensors))]
                # Since we have `output_hidden_states = True`, the 3. item will be the hidden states from all layers.


            # Finaly I get sentence vector embeddings for all sentences
            # `token_vecs` is a tensor with shape [XXX x 768]
            token_vecs_for_sentences = [hidden_state[-2][0].to(device=dev) for hidden_state in hidden_states]

            # Calculate the average of all token vectors.
            sentence_embeddings=[torch.mean(token_vecs_for_sentence, dim=0).to(device=dev) for token_vecs_for_sentence in  token_vecs_for_sentences]
            # convert tensors to vectors with float values
            sentence_vectors=[[float(sentence_embedding) for sentence_embedding in sentence_embeddings[i]] for i in range(len(sentence_embeddings))]

            end = time.time()
    
            print("Time for creating "+ str(len(sentence_embeddings))+" embedding vectors " + str((end - start)/60))
            print('Model used :'+ BERT_huge_model_name )

            return sentence_vectors
