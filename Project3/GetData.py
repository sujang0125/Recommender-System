import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import Utils
from collections import Counter

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
        self.num_ratings = len(Counter([r for r in self.df_ratings_train['rating']]))

    
    def get_target_rating(self):
        print("length of user, movie : ", len(self.set_user), len(self.set_movie))
        print("max of user, movie : ", self.max_user, self.max_movie)
        print("num_ratings : ", sorted(Counter([r*2-1 for r in self.df_ratings_train['rating']])))
        print("num_ratings : ", self.num_ratings)
        
        user_avg_rating = {}
        for u in self.set_user:
            u_rating_li = np.array(self.df_ratings_train.loc[(self.df_ratings_train['userId'] == u)]['rating'])
            avg_rating = np.sum(u_rating_li) / u_rating_li.shape[0] # user의 평균 rating
            user_avg_rating[u] = avg_rating
        
        user_movie_rating = {}
        for i, row in self.df_ratings_train.iterrows():
            user_movie_rating[(int(row['userId']), int(row['movieId']))] = row['rating']
        print(len(user_avg_rating), len(user_movie_rating))
        
        u_list, m_list, y_list = [], [], []
        for u in self.set_user:
            for m in self.set_movie:
                u_list.append(u)
                m_list.append(m)
                if (u, m) in user_movie_rating: # 유저 u가 영화 m에 rating을 했으면 해당 rating으로
                    y_list.append(int(user_movie_rating[(u, m)]*2-1))
                else: # 아닌 경우 유저가 내린 rating들의 평균으로 대체
                    y_list.append(int(user_avg_rating[u]*2-1))
        return u_list, m_list, y_list
    

    def get_ratings_train(self):
        print("a >> ", max(self.set_user))
        print("b >> ", max(self.set_movie))
    