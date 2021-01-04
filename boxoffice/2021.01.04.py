import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/train.csv")
test = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/test.csv")

train["cast"].fillna("[]", inplace=True)
train["cast"] = train["cast"].apply(eval)

all_actors = []
for movie_cast in train["cast"]:
    for member in movie_cast:
        all_actors.append(member["id"])
all_actors = pd.Series(all_actors)
popular_actors = all_actors.value_counts()[:250].index.values
popular_actors

list_features = ["cast", "genres", "production_companies", "production_countries", 
                 "spoken_languages", "Keywords", "crew"]
for x in list_features[1:]:
    train[x].fillna("[]", inplace=True) 
    train[x] = train[x].apply(eval)


all_directors = [] 
for movie_crew in train["crew"]:
    for member in movie_crew:
        if member["job"] == "Director":
            all_directors.append(member["id"])
all_directors = pd.Series(all_directors)
popular_directors = all_directors.value_counts()[:250].index.values

all_prod_companies = []
for movie in train["production_companies"]:
    for company in movie:
        all_prod_companies.append(company["id"])
all_prod_companies = pd.Series(all_prod_companies)
popular_companies = all_prod_companies.value_counts()[:10].index.values

all_prod_countries = []
for movie in train["production_countries"]:
    for country in movie:
        all_prod_countries.append(country["iso_3166_1"])
all_prod_countries = pd.Series(all_prod_countries)
popular_countries = all_prod_countries.value_counts()[:5].index.values

all_genres = []
for genre_list in train["genres"]: 
    for genre in genre_list:
        all_genres.append(genre["name"])
all_genres = pd.Series(all_genres)
popular_genres = np.append(all_genres.value_counts()[:10].index.values, "Animation")


def count_popular_actors(movie_cast):
    number_of_popular_actors = 0 
    for member in movie_cast:
        if member["id"] in popular_actors: number_of_popular_actors += 1 
    return number_of_popular_actors if number_of_popular_actors < 8 else 8 #movies with number of popular actors > 8 are strange in this dataset
def check_has_popular_director(movie_crew):
    for member in movie_crew: 
        if member["job"] == "Director" and member["id"] in popular_directors: return True     
    return False #if we iterated through all members and haven't found any popular directors
def check_has_popular_company(movie_companies):
    for company in movie_companies: 
        if company["id"] in popular_companies: return True
    return False
def check_has_popular_country(movie_countries):
    for country in movie_countries: 
        if country["iso_3166_1"] in popular_countries: return True   
    return False
def check_genre(movie_genres, target_genre): 
    for genre in movie_genres:
        if target_genre == genre["name"]: return True
    return False

train["has_popular_director"] = train["crew"].apply(check_has_popular_director)
train["number_of_popular_actors"] = train["cast"].apply(count_popular_actors)
train["from_popular_company"] = train["production_companies"].apply(check_has_popular_company)
train["from_popular_country"] = train["production_countries"].apply(check_has_popular_country)
for genre in popular_genres: 
    train["Is"+"".join(genre.split(" "))] = train["genres"].apply(check_genre, target_genre=genre)


train.drop(["belongs_to_collection","homepage", "cast", "crew", "title", "tagline", "spoken_languages", "production_companies", "Keywords", "poster_path", "status",
           "production_countries", "overview", "original_title", "original_language", "imdb_id", "genres"], axis=1, inplace=True)

train.drop("release_season", axis=1, inplace=True) 


plt.scatter(train["release_year"], train["revenue"])