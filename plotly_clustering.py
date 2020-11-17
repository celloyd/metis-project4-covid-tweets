import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

# Reading in 12-topic LDA results as previously made with gensim
tweet_topic_df_12 = pd.read_csv('12_topic_LDA_w_tweets.csv', index_col = False)
tweet_topic_columns = ['index'] + list(range(12)) + ['tweet']
tweet_topic_df_12.columns = tweet_topic_columns
tweet_topic_df_12.set_index('index', drop = True, inplace = True)
# Getting the dominant topic for each tweet
tweet_topic_df_12['topic'] = tweet_topic_df_12[list(range(12))].idxmax(axis = 1)

pca = PCA(n_components = 2)
projected_12 = pca.fit_transform(tweet_topic_df_12[list(range(12))])
# PCA to dataframe for graphing
final_projection_12 = pd.DataFrame(data = projected_12, columns = ['principal component 1', 'principal component 2'])
final_projection_12 = final_projection_12.join(tweet_topic_df_12[['topic', 'tweet']])

# Plotly to allow for exploration
fig = px.scatter(final_projection_12, x = 'principal component 1', y = 'principal component 2', 
                 opacity = 0.5, color = 'topic', hover_data = ['tweet'], color_continuous_scale= 'rainbow')
fig.show()
