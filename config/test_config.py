# dataset = ["ihc", "sbic", "dynahate", "hateval", "toxigen", "white", "ethos"]
dataset = ["ethos"]
model_path = ["hateval"]

tuning_param  = ["learning_rate","train_batch_size","eval_batch_size","nepoch","SEED","dataset","model_path"] ## list of possible paramters to be tuned

train_batch_size = [16]
eval_batch_size = [16]
hidden_size = 768
nepoch = [6]    
learning_rate = [2e-5]

model_type = "bert-base-uncased"
SEED = [0]
e = 1e-3

param = {"e":e, "dataset":dataset,"model_path":model_path,"learning_rate":learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset, "SEED":SEED,"model_type":model_type}
