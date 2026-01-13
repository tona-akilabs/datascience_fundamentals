import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
data = {
    'movie_id': [1, 2, 3, 4, 5, 6],
    'title': ['Toy Story', 'The Dark Knight', 'Shrek', 'Batman Begins', 'Aladdin', 'Renegade Immortal'],
    'genres': ['Animation Children Comedy', 'Action Crime Drama', 'Animation Children Comedy', 'Action Crime Drama', 'Animation Children Adventure', 'Animation']
}

df = pd.DataFrame(data)
#print(df)

# Initialize the vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df['genres'])
#print(tfidf_matrix)

# Look at the feature names (the genres the model identified)
#print(tfidf.get_feature_names_out())

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#print(cosine_sim)

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 3 most similar movies (excluding itself)
    sim_scores = sim_scores[1:4]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 3 most similar movies
    return df['title'].iloc[movie_indices]
    #return df[['title', 'genres']].iloc[movie_indices]


# Test it out!
print(get_recommendations('Toy Story'))
# Output: Shrek, Aladdin, Renegade Immortal
print(get_recommendations('Renegade Immortal'))
# Output: Toy Story, Shrek, Aladdin