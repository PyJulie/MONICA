import torch
import time
import tqdm
from byol_pytorch import BYOL
def train_BYOL(configs, model, dataloaders, optimizer):
    learner = BYOL(
            model,
            image_size = 224,
            hidden_layer = 'global_pool',
            use_momentum = False       # turn off momentum in the target encoder
        )
    learner = learner.cuda()
    for epoch in range(configs.general.train_epochs):
        print('Epoch {}/{}'.format(epoch, configs.general.train_epochs-1))

        running_loss = 0
        for data in tqdm.tqdm(dataloaders['train']):
            img, _ = data
            img = img.cuda()
            loss = learner(img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # learner.update_moving_average()
            running_loss += loss
        # store loss history
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)


        print('train loss: %.6f' %(epoch_loss))



    # save weights
    # torch.save(q_encoder.state_dict(), path2weights);

    return model