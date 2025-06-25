import pandas as pd
import numpy as np
df = pd.read_csv('C:/Users/admin/Downloads/MajorProj/clean_songs_dataset2.csv')
df['soup'] = df['language']+" "+df['artist']+" "+df['top genre']


from sklearn.feature_extraction.text import CountVectorizer
# Vectorizes words to numbers and builds a sparse matrix
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(df['soup'])


from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
#Computes similarity between songs using cosine similarity metric
cosine_sim = cosine_similarity(count_matrix, count_matrix)
df = df.reset_index()
indices = pd.Series(df.index, index=[df['language'],df['top genre'],df['artist']])


def hybrid(top_genre,language,artist):
    
    #Extract index of song and genre
    idx = indices[language,top_genre,artist]
    
    #Extract the similarity scores and their corresponding index for every song from the cosine similarity matrix
    #print(idx)
    sim_scores = list(enumerate(cosine_sim[int(idx[0])]))
    #Sort the (index,score) tuples in decreasing order of similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    #Select top 25
    sim_scores = sim_scores[1:26]
    #Store the cosine_sim indices of the top 25 songs in a list
    drug_indices = [i[0] for i in sim_scores]
    
    #Extract metadata of the drug
    drugs = df.iloc[drug_indices][['title','artist','top genre','pop','Link']]
    drugs.columns = ['Title','Artist','Emotion','Popularity','Link']
    drugs = drugs.sort_values(by='Popularity', ascending=False)
    drugs = drugs[['Title','Link','Emotion']]
    #Return top 10 drugs as recommendations
    return drugs.head(10)