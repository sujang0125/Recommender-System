from test import test, generate_output_txt
from GetData import *

def main():
    # test()
    gg = GetMovieSim()
    gg.define_similar_movie()
    # generate_output_txt()
    
if __name__=="__main__":
    main()