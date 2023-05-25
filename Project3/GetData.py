import pandas as pd
import numpy as np
import time
import torch
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import Utils
from collections import Counter, defaultdict

class CustomDataset(Dataset):
    def __init__(self):
        self.traindata = GetTrainData()
        u, m, y = self.traindata.get_target_rating()
        self.u_data = u
        self.m_data = m
        self.y_data = y

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        user = self.u_data[idx]
        movie = self.m_data[idx]
        y = self.y_data[idx]
        return user, movie, y
    

class GetTrainData():
    def __init__(self):
        self.df_ratings_train = pd.read_csv(Utils.ratings_train_path)
        self.set_user = set(self.df_ratings_train['userId'])
        self.set_movie = set(self.df_ratings_train['movieId'])
        self.max_user = max(self.set_user)
        self.max_movie = max(self.set_movie)
        self.num_ratings = len(set([r for r in self.df_ratings_train['rating']]))
        
        getsim = GetMovieSim()
        simmv_set, second_simmv_set = getsim.define_similar_movie()
        self.simmv_set = simmv_set
        self.second_simmv_set = second_simmv_set ### {movie: set[ (sim_movie, cossim_value)... ]}

    def get_target_rating(self):
        print("length of user, movie : ", len(self.set_user), len(self.set_movie))              ### 547 7357
        print("max of user, movie : ", self.max_user, self.max_movie)                           ### 671 94833
        print("ratings_list : ", sorted(set([r for r in self.df_ratings_train['rating']])))     ### [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        print("num_ratings : ", self.num_ratings)                                               ### 10
        
        ### 한 유저가 내린 평점들의 평균
        user_avg_rating = {}
        user_movieset = set()
        for u in self.set_user:
            u_rating_li = np.array(self.df_ratings_train.loc[(self.df_ratings_train['userId'] == u)]['rating'])
            avg_rating = np.sum(u_rating_li) / u_rating_li.shape[0] # user의 평균 rating
            user_avg_rating[u] = avg_rating
            
        
        ### 유저 - 영화 rating을 dictionary로 {(user, movie): rating}, 길이는 ratings_train.csv의 줄 수
        user_movie_rating = {}
        mv_per_user = defaultdict(set)
        for i, row in self.df_ratings_train.iterrows():
            user = int(row['userId'])
            movie = int(row['movieId'])
            rating = row['rating']
            user_movie_rating[(user, movie)] = rating
            mv_per_user[user].add(movie)
        
        u_list, m_list, y_list = [], [], []
        for u in self.set_user:
            for m in self.set_movie:
                u_list.append(u)
                m_list.append(m)
                if (u, m) in user_movie_rating: ### 유저 u가 영화 m에 rating을 했으면 해당 rating으로
                    y_list.append(user_movie_rating[(u, m)])
                else:  
                    flag = 0
                    if flag == 0:
                        for simmv in self.simmv_set[m]: # rating이 없는 영화는 그 영화와 가장 비슷한 영화의 평점으로
                            if simmv in mv_per_user[u]:
                                y_list.append(user_movie_rating[(u, simmv)])
                                flag = 1
                                break
                    # if flag == 0:
                    #     for simmv in self.second_simmv_set[m]: # rating이 없는 영화는 그 영화와 두 번째로 비슷한 영화의 평점으로
                    #         if simmv in mv_per_user[u]:
                    #             r = user_movie_rating[(u, simmv)]
                    #             y_list.append(r)
                    #             # if r > 2.5:
                    #             #     y_list.append(r * 0.9)
                    #             # else:
                    #             #     y_list.append(r * 1.1)
                    #             flag = 1
                    #             break
                    if flag == 0: # 비슷한 영화도 없으면 그냥 유저가 내린 rating들의 평균으로 대체
                        y_list.append(user_avg_rating[u])
                        flag = 1
                    
                    if flag == 0:
                        print("there are no y value >>> error occured")
                        return
                    # 아닌 경우 유저가 내린 rating들의 평균으로 대체
                    # y_list.append(user_avg_rating[u])
                    # 아닌 경우 0으로 대체
                    # y_list.append(int(0))
        return u_list, m_list, y_list
    
    
class GetValData():
    def __init__(self):
        self.df_ratings_val = pd.read_csv(Utils.ratings_val_path)
        self.max_user = max(self.df_ratings_val['userId'])
        self.max_movie = max(self.df_ratings_val['movieId'])
        self.num_ratings = len(Counter([r for r in self.df_ratings_val['rating']]))
        self.get_train_data = GetTrainData()
    
    def get_val_data(self):
        val_data = []
        for i, row in self.df_ratings_val.iterrows():
            user = int(row['userId'])
            movie = int(row['movieId'])
            rating = row['rating']
            
            ### ratings_train.csv에서 보지 못한 유저나 영화는 고려 안 함
            if user not in self.get_train_data.set_user or movie not in self.get_train_data.set_movie:
                continue
            val_data.append((user, movie, rating))
        return val_data

class GetMovieSim():
    def __init__(self) -> None:
        self.df_movies_w_imgurl = pd.read_csv(Utils.movies_w_imgurl_path)
        self.df_tags = pd.read_csv(Utils.tags_path)
        self.df_ratings_train = pd.read_csv(Utils.ratings_train_path)
        self.sim_k = Utils.sim_k
        self.second_sim_k = Utils.second_sim_k
            
    def get_movie_info(self):
        ### store the movie feature
        movie_info = defaultdict(set) # {movie: [ genres and tags ... ]}
        movie_index = defaultdict(int) # {movie: index}
        index_movie = defaultdict(int) # {index: movie}
        
        ### add genre information
        for i, row in self.df_movies_w_imgurl.iterrows():
            movie = int(row['movieId'])
            genres = row['genres'].split('|')
            for genre in genres:
                ### in case of the (no genres listed)
                # if genre == "(no genres listed)": 
                #     movie_info[movie].append("")
                #     continue
                movie_info[movie].add(genre.replace(" ", "").lower()) # remove blank & lower case just in case
            index_movie[i] = movie
            movie_index[movie] = i
            
        ### add tag information
        for i, row in self.df_tags.iterrows():
            movie = int(row['movieId'])
            tags = row['tag'].split(',')
            for tag in tags:
                movie_info[movie].add(tag.replace(" ", "").lower()) # remove blank & lower case just in case
                
        for m in movie_info:
            movie_info[m] = sorted(list(movie_info[m]))
        # for m in movie_info:
        #     print(m, movie_info[m])
        return movie_info, movie_index, index_movie
            
    def get_movie_cossim(self):
        movie_info, movie_index, index_movie = self.get_movie_info()
        li = []
        for m in movie_info:
            feature = " ".join(movie_info[m])
            li.append(feature)
            
        ### get tf-idf
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(li)
        x_data = tfidf_vectorizer.transform(li)
        
        ### get cossine similarity
        x_cossim = cosine_similarity(x_data, x_data)
        return x_cossim, movie_index, index_movie
        
    def define_similar_movie(self):
        x_cossim, movie_index, index_movie = self.get_movie_cossim()
        simmv_set = defaultdict(set)
        second_simmv_set = defaultdict(set)
        cnt, scnt = 0, 0
        for i, movie_sim in enumerate(x_cossim):
            movie = index_movie[i]
            ### 유사도가 높은 영화들을 비슷한 영화로 간주
            movie_sim_pair = [(index_movie[k], sim) for k, sim in enumerate(movie_sim.tolist()) if i != k and sim > self.sim_k]
            
            # 영화와 비슷한 영화들을 similarity가 큰 순서대로 정렬
            simmv_set[movie] = sorted(movie_sim_pair, reverse=True, key=lambda x: x[1])
            second_simmv_set[movie] = [(index_movie[k], sim) for k, sim in enumerate(movie_sim.tolist()) if i != k and (sim <= self.sim_k and sim > self.second_sim_k)]
            # print(simmv_set[movie], len(simmv_set[movie]))
        return simmv_set, second_simmv_set