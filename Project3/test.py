import torch
from Model import NeuMFModel
import Utils

def test():
    model = NeuMFModel()
    load_name = Utils.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model.load_state_dict(torch.load("./" + load_name, map_location=device))
    except FileNotFoundError as e:
        print("model load failed")
        print(e)
        return False
    
    ## test
    # with torch.no_grad():
    #     model.eval()
    #     inputs = torch.FloatTensor([[1 ** 2, 1], [5 **2, 5], [11**2, 11]]).to(device)
    #     outputs = model(inputs)
    #     print(outputs)