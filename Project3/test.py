import torch
from Model import NeuMFModel
import Utils
from GetData import GetValData, GetTrainData
import numpy as np

def test():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using CUDA ", torch.cuda.is_available())
    print("CUDA device number ", torch.cuda.current_device())
    print("CUDA device number ", torch.cuda.get_device_name(0))
    traindata = GetTrainData()
    
    max_user = traindata.max_user
    max_movie = traindata.max_movie
    num_ratings = traindata.num_ratings
    
    model = NeuMFModel(num_users=max_user+1, num_items=max_movie+1, num_ratings=num_ratings)
    load_name = Utils.model_name
    get_val_data = GetValData()
    val_data = get_val_data.get_val_data()
    
    try:
        model.load_state_dict(torch.load("./" + load_name))
    except FileNotFoundError as e:
        print("model load failed")
        print(e)
        return False
    
    mean_rmse = 0
    for data in val_data:
        # data[0]: user, data[1]: movie, data[2]: real rating
        
        user = torch.tensor([data[0]])
        movie = torch.tensor([data[1]])
        y = data[2]
        with torch.no_grad():
            model.eval()
            output = model(user, movie)
            output = output.detach().cpu().numpy()
            # output = np.argmax(output)
            # output = (output + 1)/2
            print(output, y)
            rmse = np.sqrt(np.mean((output - y)**2))
            print("rmse:",rmse)
            mean_rmse += rmse
    mean_rmse = mean_rmse / len(val_data)
    print("mean_rmse:", mean_rmse)