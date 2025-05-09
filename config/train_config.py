
dataset = ["dynahate"]

tuning_param  = ["learning_rate","train_batch_size","eval_batch_size","nepoch","SEED","dataset"] ## list of possible paramters to be tuned

train_batch_size = [16]
eval_batch_size = [16]
hidden_size = 768
# hidden_size = 1024
nepoch = [6]    
learning_rate = [2e-5]
# loss = "contrastive-learning"
loss = "cross-entropy"
lambda_loss = 0.5
e = 1

model_type = "bert-base-uncased"
# model_type = "xlm-roberta-base"
SEED = [0]

param = {"e":e, "lambda_loss":lambda_loss,"loss":loss,"dataset":dataset,"learning_rate":learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset, "SEED":SEED,"model_type":model_type}
