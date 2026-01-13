import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = pd.DataFrame({
    "user": ["A", "A", "B", "B", "C"],
    "item": ["Book", "Pen", "Book", "Laptop", "Pen"],
    "rating": [5, 3, 4, 5, 4]
})

# Pivot table
matrix = data.pivot_table(index="user", columns="item", values="rating").fillna(0)

# Similarity
similarity = cosine_similarity(matrix)

print(similarity)

similarity_df = pd.DataFrame(
    similarity,
    index=matrix.index,
    columns=matrix.index
)
#print(similarity_df)


def recommend(user, matrix, similarity_df, top_n=2):
    similar_users = similarity_df[user].sort_values(ascending=False)[1:]
    recommendations = {}

    for other_user, score in similar_users.items():
        for item in matrix.columns:
            if matrix.loc[user, item] == 0 and matrix.loc[other_user, item] > 0:
                recommendations[item] = recommendations.get(item, 0) + score

    #print(recommendations)
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

print(recommend("A", matrix, similarity_df))