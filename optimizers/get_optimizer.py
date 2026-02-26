import torch.optim as optim
from optimizers.SAM import SAM

def get_optimizer(configs, model):
    weight_decay = configs.optimizer.weight_decay if configs.optimizer.weight_decay else 0
    method = configs.general.method
    optimizer_type = configs.optimizer.type
    learning_rate = configs.optimizer.learning_rate
    if_freeze_encoder = configs.model.if_freeze_encoder
    model_name = configs.model.model_name
    
    if method == 'mocov2':
        model = model[0]
    
    parameters = model.parameters()
    
    if if_freeze_encoder:
        parameters = model.classifier.parameters()
    
    if optimizer_type == 'Adam':
        if method == 'LWS':
            parameters = [model.fc.scales]
        elif method == 'MiSLAS':
            parameters = list(model.fc.parameters()) + [model.fc.scales]
        
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    
    elif optimizer_type == 'SAM':
        base_optimizer = optim.Adam
        optimizer = SAM(base_optimizer=base_optimizer, rho=0.05, params=parameters, lr=3e-4)
    
    else:  # Default to SGD
        if method == 'LWS':
            parameters = [model.fc.scales]
        elif method == 'MiSLAS':
            parameters = list(model.fc.parameters()) + [model.fc.scales]
        
        optimizer = optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    
    return optimizer
