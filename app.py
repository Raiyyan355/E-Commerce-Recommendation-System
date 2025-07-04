from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# --- Data Loading and Preprocessing ---
# Load the dataset
try:
    train_data = pd.read_csv('marketing_sample.tsv', sep='\t')
except FileNotFoundError:
    print("Dataset file not found. Please make sure the .tsv file is in the same directory.")
    exit()


# Select and rename columns for clarity
train_data = train_data[['Uniq Id','Product Id', 'Product Rating', 'Product Reviews Count', 'Product Category', 'Product Brand', 'Product Name', 'Product Image Url', 'Product Description', 'Product Tags']]
column_name_mapping = {
    'Uniq Id': 'ID',
    'Product Id': 'ProdID',
    'Product Rating': 'Rating',
    'Product Reviews Count': 'ReviewCount',
    'Product Category': 'Category',
    'Product Brand': 'Brand',
    'Product Name': 'Name',
    'Product Image Url': 'ImageURL',
    'Product Description': 'Description',
    'Product Tags': 'Tags',
}
train_data.rename(columns=column_name_mapping, inplace=True)

# Fill missing values
train_data['Rating'] = train_data['Rating'].fillna(0)
train_data['ReviewCount'] = train_data['ReviewCount'].fillna(0)
train_data['Category'] = train_data['Category'].fillna('')
train_data['Brand'] = train_data['Brand'].fillna('')
train_data['Description'] = train_data['Description'].fillna('')


# --- Recommendation Functions from the Notebook ---

def content_based_recommendations(train_data, item_name, top_n=10):
    # Try to find the best matching product
    matched_rows = train_data[train_data['Name'].str.contains(item_name, case=False, na=False)]
    
    if matched_rows.empty:
        return pd.DataFrame()
    
    # Use the first matching product
    item_index = matched_rows.index[0]

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'].astype(str))
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    
    return train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]


def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    # This is a simplified version for demonstration as user IDs are not real
    # In a real app, user IDs would be managed properly.
    
    # Extract numeric part of ID for pivoting
    train_data['ID_numeric'] = train_data['ID'].str.extract(r'(\d+)').astype(float).fillna(0).astype(int)
    train_data['ProdID_numeric'] = train_data['ProdID'].str.extract(r'(\d+)').astype(float).fillna(0).astype(int)


    user_item_matrix = train_data.pivot_table(index='ID_numeric', columns='ProdID_numeric', values='Rating', aggfunc='mean').fillna(0)
    
    if target_user_id not in user_item_matrix.index:
        return pd.DataFrame() # Return empty if user not in matrix

    user_similarity = cosine_similarity(user_item_matrix)
    
    try:
        target_user_index = user_item_matrix.index.get_loc(target_user_id)
    except KeyError:
        return pd.DataFrame()

    user_similarities = user_similarity[target_user_index]
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    recommended_items = []
    target_user_ratings = user_item_matrix.iloc[target_user_index]

    for user_index in similar_users_indices:
        similar_user_ratings = user_item_matrix.iloc[user_index]
        # Recommend items that the similar user has rated but the target user has not
        new_recommendations = similar_user_ratings[target_user_ratings == 0].index
        recommended_items.extend(new_recommendations)
        if len(recommended_items) >= top_n:
            break
            
    recommended_items = list(dict.fromkeys(recommended_items)) # Remove duplicates

    return train_data[train_data['ProdID_numeric'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']].head(top_n)

def hybrid_recommendations(train_data, target_user_id, item_name, top_n=10):
    content_based_rec = content_based_recommendations(train_data, item_name, top_n)
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data, target_user_id, top_n)
    
    if content_based_rec.empty and collaborative_filtering_rec.empty:
        return pd.DataFrame() # Return empty if both are empty

    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates(subset=['Name']).head(top_n)
    return hybrid_rec

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    item_name = ""
    num_rec = 10
    error = None

    # Get a list of unique product names for the dropdown
    product_names = sorted(train_data['Name'].unique().tolist())

    if request.method == 'POST':
        item_name = request.form.get('item_name')
        num_rec_str = request.form.get('num_rec')

        if not item_name:
            error = "Please select a product."
        elif not num_rec_str or not num_rec_str.isdigit() or int(num_rec_str) <= 0:
            error = "Please enter a valid number of recommendations."
        else:
            num_rec = int(num_rec_str)
            # For demonstration, we'll use a fixed user ID. In a real application,
            # this would come from the logged-in user's session.
            target_user_id = 4 
            
            recs = hybrid_recommendations(train_data, target_user_id, item_name, num_rec)
            if not recs.empty:
                recommendations = recs.to_dict(orient='records')
            else:
                error = f"Could not generate recommendations for '{item_name}'. It might not have enough data."


    return render_template('index.html', 
                           product_names=product_names, 
                           recommendations=recommendations, 
                           selected_item=item_name,
                           num_rec=num_rec,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)