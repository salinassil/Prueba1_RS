import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.title("Recommended for you!")





# fetch few past orders of a user, based on which personalized recommendations are to be made
def get_latest_user_orders(user_id, orders, num_orders=3):
    counter = num_orders
    order_indices = []
    
    for index, row in orders[['user_id']].iterrows():
        if row.user_id == user_id:
            counter = counter -1
            order_indices.append(index)
        if counter == 0:
            break
            
    return order_indices

# utility function that returns a DataFrame given the food_indices to be recommended
def get_recomms_df(food_indices, df1, columns, comment):
    row = 0
    df = pd.DataFrame(columns=columns)
    
    for i in food_indices:
        df.loc[row] = df1[['title', 'canteen_id', 'price']].loc[i]
        df.loc[row].comment = comment
        row = row+1
    return df

# return food_indices for accomplishing personalized recommendation using Count Vectorizer
def personalised_recomms(orders, df1, user_id, columns, comment="based on your past orders"):
    order_indices = get_latest_user_orders(user_id, orders)
    food_ids = []
    food_indices = []
    recomm_indices = []
    
    for i in order_indices:
        food_ids.append(orders.loc[i].food_id)
    for i in food_ids:
        food_indices.append(indices_from_food_id[i])
    for i in food_indices:
        recomm_indices.extend(get_recommendations(idx=i))
        
    return get_recomms_df(set(recomm_indices), df1, columns, comment)

# Simply fetch new items added by vendor or today's special at home canteen
def get_new_and_specials_recomms(new_and_specials, users, df1, canteen_id, columns, comment="new/today's special item  in your home canteen"):
    food_indices = []
    
    for index, row in new_and_specials[['canteen_id']].iterrows():
        if row.canteen_id == canteen_id:
            food_indices.append(indices_from_food_id[new_and_specials.loc[index].food_id])
            
    return get_recomms_df(set(food_indices), df1, columns, comment)

# utility function to get the home canteen given a user id
def get_user_home_canteen(users, user_id):
    for index, row in users[['user_id']].iterrows():
        if row.user_id == user_id:
            return users.loc[index].home_canteen
    return -1

# fetch items from previously calculated top_rated_items list
def get_top_rated_items(top_rated_items, df1, columns, comment="top rated items across canteens"):
    food_indices = []
    
    for index, row in top_rated_items.iterrows():
        food_indices.append(indices_from_food_id[top_rated_items.loc[index].food_id])
        
    return get_recomms_df(food_indices, df1, columns, comment)

# fetch items from previously calculated pop_items list
def get_popular_items(pop_items, df1, columns, comment="most popular items across canteens"):
    food_indices = []
    
    for index, row in pop_items.iterrows():
        food_indices.append(indices_from_food_id[pop_items.loc[index].food_id])
        
    return get_recomms_df(food_indices, df1, columns, comment)

df1 = pd.read_csv('./db/df1.csv')
orders = pd.read_csv('./db/orders.csv')
new_and_specials = pd.read_csv('./db/new_and_specials.csv')
users = pd.read_csv('./db/users.csv')

columns = ['title', 'canteen_id', 'price', 'comment']
#current_user = 2


# Import CountVectorizer and create the count matrix

count = CountVectorizer(stop_words='english')

# df1['soup']
count_matrix = count.fit_transform(df1['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix

cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices_from_title = pd.Series(df1.index, index=df1['title'])
indices_from_food_id = pd.Series(df1.index, index=df1['food_id'])


# Function that takes in food title or food id as input and outputs most similar dishes 
def get_recommendations(title="", cosine_sim=cosine_sim, idx=-1):
    # Get the index of the item that matches the title
    if idx == -1 and title != "":
        idx = indices_from_title[title]

    # Get the pairwsie similarity scores of all dishes with that dish
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the dishes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar dishes
    sim_scores = sim_scores[1:3]

    # Get the food indices
    food_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar dishes
    return food_indices



current_user=st.sidebar.number_input('''Enter your user name''')

a=personalised_recomms(orders, df1, current_user, columns)
run = st.button('Submit')
#current_canteen = get_user_home_canteen(users, current_user)
if run:
    
    st.table(a)
