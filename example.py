from libpredict import train_and_save_neural_network, load_and_run_neural_network

if __name__=='__main__':
    train_and_save_neural_network("bitcoin-close.csv", "./save/temp.sv")
    load_and_run_neural_network("./save/temp.sv-1000",[20,34,3,89,4,20,34,3,89,4,20,34,3,89,4,20,34,3,89,4])