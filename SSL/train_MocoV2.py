import torch
import time
import tqdm
def train_MocoV2(configs, dataloaders, model, loss_func, optimizer):
    momentum = 0.999
    q_encoder, k_encoder = model
    loss_func = loss_func['train']
    queue = None
    K = 8192 # K: number of negatives to store in queue

    # fill the queue with negative samples
    flag = 0
    if queue is None:
        while True:

            with torch.no_grad():
                for img, _ in dataloaders['train']:
                    # extract key samples
                    xk = img[1].cuda()
                    k = k_encoder(xk).detach()

                    if queue is None:
                        queue = k
                    else:
                        if queue.shape[0] < K: # queue < 8192
                            queue = torch.cat((queue,k),0)
                        else:
                            flag = 1 # stop filling the queue

                    if flag == 1:
                        break 

            if flag == 1:
                break

    queue = queue[:K]
    q_encoder.train()
    for epoch in range(configs.general.train_epochs):
        print('Epoch {}/{}'.format(epoch, configs.general.train_epochs-1))

        q_encoder.train()
        running_loss = 0
        for data in tqdm.tqdm(dataloaders['train']):
            img, _ = data
            # retrieve query and key
            xq = img[0].cuda()
            xk = img[1].cuda()

            # get model outputs
            q = q_encoder(xq)
            k = k_encoder(xk).detach()

            # normalize representations
            q = torch.div(q, torch.norm(q,dim=1).reshape(-1,1))
            k = torch.div(k, torch.norm(k,dim=1).reshape(-1,1))

            # get loss value
            loss = loss_func(q, k, queue)
            running_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the queue
            queue = torch.cat((queue, k), 0)

            if queue.shape[0] > K:
                queue = queue[256:,:]
            
            # update k_encoder
            for q_params, k_params in zip(q_encoder.parameters(), k_encoder.parameters()):
                k_params.data.copy_(momentum*k_params + q_params*(1.0-momentum))

        # store loss history
        epoch_loss = running_loss / len(dataloaders['train'].dataset)


        print('train loss: %.6f' %(epoch_loss))



    # save weights
    # torch.save(q_encoder.state_dict(), path2weights);

    return q_encoder, k_encoder