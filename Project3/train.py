from Model import NeuMFModel
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import time
import os

from GetData import CustomDataset, GetTrainData
import Utils

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = CustomDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    print(len(train_dataset.u_data), len(train_dataset.y_data))
    print("dataload finished")
    
    max_user = train_dataset.traindata.max_user
    max_movie = train_dataset.traindata.max_movie
    num_ratings = train_dataset.traindata.num_ratings
    
    model = NeuMFModel(num_users=max_user + 1 ,num_items=max_movie + 1, num_ratings=num_ratings)
    model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    save_name = Utils.model_name
    
    # model.train()
    for epoch in range(1000):
        cost = 0.0
        print("epoch", epoch)
        for u, m, y in train_dataloader:
            u = u.to(device)
            m = m.to(device)
            y = y.to(device)
            output = model(u, m)
            
            # y rating을 one-hot으로
            y_one_hot = torch.zeros(output.shape[0], output.shape[1]).to(device)
            y_one_hot.scatter_(1, torch.unsqueeze(y,1), 1).to(device)
            
            # gradient를 0으로 초기화
            optimizer.zero_grad()
            
            loss = criterion(output, y_one_hot)
            # loss = criterion(output, y.to(torch.float32))
            
            # 비용 함수를 미분하여 gradient 계산
            loss.backward()
            # W와 b를 업데이트
            optimizer.step()
            
            cost += loss

        cost = cost / len(train_dataloader)

        loss_state = f"Epoch : {epoch+1:4d}, Cost : {cost:.4f}"
        print(loss_state)
        loss_file = open("./" + save_name + "_loss.txt", 'a')
        loss_file.write(f"{cost:.4f}")
        loss_file.close()
        torch.save(model.state_dict(), "./" + save_name)
    
