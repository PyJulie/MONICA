import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
import tqdm
import datetime
import os
import random
from sampler import get_sampler
from dataset.get_datasets import get_dataloaders, get_datasets
from models.get_model import get_model
from optimizers.get_optimizer import get_optimizer
from losses.get_loss import get_loss_functions, calculate_loss
from losses.MixUp import mixup_data
from utils.log_accuracy import *
from utils.setup_configs import *
from utils.utils import *
from models.KNNClassifier import KNNClassifier
# import wandb



def main(configs):
    
    datasets = get_datasets(configs)
    cls_num_list = get_cls_num_list(datasets['train'])
    configs.datasets.cls_num_list = cls_num_list
    dataloaders = get_dataloaders(datasets, configs)

    model = get_model(configs)
    optimizer = get_optimizer(configs, model)
    
    loss_functions = get_loss_functions(configs, cls_num_list)
    train(configs, dataloaders, model, optimizer, loss_functions)



def train(configs, dataloaders, model, optimizer, loss_functions):
    model.train()
    best_acc = 0
    if configs.optimizer.cos_lr:
        print("[INFORMATION] Using cosine lr_scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, configs.general.train_epochs, eta_min=0.0)
    if configs.general.method == 'KNN':
        knnclassifier = KNNClassifier(feat_dim=configs.model.feature_dim, num_classes=configs.general.num_classes, feat_type='cl2n', dist_type='l2')
        features = get_knncentroids(dataloaders, model)
        knnclassifier.update(features)
        eval(dataloaders, [model, knnclassifier], configs, 0, loss_functions, 0)
        return
    for epoch in range(configs.general.train_epochs):
        running_loss = 0
        correct = list(0. for i in range(configs.general.num_classes))
        total = list(0. for i in range(configs.general.num_classes))
        all_labels = []
        all_outputs = []
        if configs.general.method == 'RSG':
            if epoch == int(configs.general.train_epochs*4/5):
                configs.datasets.sampler = 'RSG'
                datasets = get_datasets(configs)
                sampler, shuffle = get_sampler.get_sampler(configs, datasets)
                dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], 
                                                                        batch_size=configs.datasets.batch_size, 
                                                                        shuffle=shuffle, num_workers=configs.general.num_workers, sampler = sampler)
        if configs.general.method == 'SAM':
            adjust_rho(optimizer, epoch)

        for data in tqdm.tqdm(dataloaders['train']):
            optimizer.zero_grad()
            if configs.general.method == 'BBN':
                iter_dataloader_RS = iter(dataloaders['RS_train'])
                iter_dataloader_OR = iter(dataloaders['train'])
                try:
                    data_OR = next(iter_dataloader_OR)
                except StopIteration:
                    iter_dataloader_OR = iter(dataloaders['train'])
                    data_OR = next(iter_dataloader_OR)

                try:
                    data_RS = next(iter_dataloader_RS)
                except StopIteration:
                    iter_dataloader_RS = iter(dataloaders['RS_train'])
                    data_RS = next(iter_dataloader_RS)
                inputs_RS, target_RS = data_RS
                inputs, labels = data_OR
                if configs.cuda.use_gpu:
                    inputs_RS = Variable(inputs_RS.cuda())
                    target_RS = Variable(target_RS.cuda())
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    inputs_RS, target_RS = Variable(inputs_RS), Variable(target_RS)
            else:
                inputs, labels = data
                if configs.cuda.use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
            
            if configs.general.method == 'MixUp' or 'GCL' in configs.general.method:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                outputs = model(inputs)
                mixup_arguments = [targets_a, targets_b, lam]
                train_loss = calculate_loss(configs, outputs, mixup_arguments, loss_functions['train'], 'train')
                correct, total = calculate_metrics_single(labels, outputs, correct, total)
            elif configs.general.method == 'CenterLoss':
                epoch_info = {'epoch': epoch}
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                features = model.forward_features(inputs)
                features = model.forward_head(features, pre_logits=True)
                outputs = model(inputs)
                mixup_arguments = [features, targets_a, targets_b, lam]
                train_loss = calculate_loss(configs, outputs, mixup_arguments, loss_functions['train'], 'train', **epoch_info)
                correct, total = calculate_metrics_single(labels, outputs, correct, total)

            elif configs.general.method == 'Logits_Adjust_Loss':
                cls_num_list = get_cls_num_list(dataloaders['train'].dataset)
                tau = 1.0
                base_probs = [x/max(cls_num_list) for x in cls_num_list]
                base_probs = torch.Tensor(base_probs).cuda()
                outputs = model(inputs)
                outputs = outputs - torch.log((base_probs**tau) + 1e-12)
                train_loss = calculate_loss(configs, outputs, labels, loss_functions['train'], 'train')
                correct, total = calculate_metrics_single(labels, outputs, correct, total)
            elif configs.general.method == 'BBN':
                feature_a, feature_b = (
                    model.forward_features(inputs),
                    model.forward_features(inputs_RS),
                )
                feature_a, feature_b = (
                    model.forward_head(feature_a, pre_logits=True),
                    model.forward_head(feature_b, pre_logits=True),
                )

                l = 1 - ((epoch - 1) / 100 * (configs.general.train_epochs // 100 + 1)) ** 2
                configs.general.l = l
                mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)
                outputs = model.classifier(mixed_feature)
                clabels = [labels, target_RS]
                train_loss = calculate_loss(configs, outputs, clabels, loss_functions['train'], 'train')
                correct, total = calculate_metrics_single(labels, outputs, correct, total)
            elif configs.general.method == 'RSG':
                outputs, cesc_loss, total_mv_loss, combine_target= model(inputs, epoch, labels)
                epoch_info = {'epoch': epoch}
                ldam_loss = calculate_loss(configs, outputs, combine_target, loss_functions['train'], 'train', **epoch_info)
                train_loss =  ldam_loss + 0.1 * cesc_loss.mean() + 0.01 * total_mv_loss.mean()
                #print(ldam_loss, cesc_loss.mean(), total_mv_loss.mean())
            elif configs.general.method == 'LDAM' or configs.general.method == 'SAM':
                outputs = model(inputs)
                epoch_info = {'epoch': epoch}
                train_loss = calculate_loss(configs, outputs, labels, loss_functions['train'], 'train', **epoch_info)
            elif configs.general.method == 'SADE':
                extra_info = {}
                SADE_outputs = model(inputs)
                outputs = SADE_outputs['output']
                logits = SADE_outputs["logits"]
                extra_info.update({"logits": logits.transpose(0, 1)})
                train_loss = calculate_loss(configs, SADE_outputs, labels, loss_functions['train'], 'train', **extra_info)
                
            else:
                outputs = model(inputs)
                # print(outputs.shape, labels.shape)
                train_loss = calculate_loss(configs, outputs, labels, loss_functions['train'], 'train')
                correct, total = calculate_metrics_single(labels, outputs, correct, total)
            running_loss += train_loss.data.item()
            train_loss.backward()
            if configs.general.method == 'SAM':
                optimizer.first_step(zero_grad=True)
                outputs = model(inputs)
                epoch_info = {'epoch': epoch}
                train_loss = calculate_loss(configs, outputs, labels, loss_functions['train'], 'train', **epoch_info)
                train_loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
        if configs.optimizer.cos_lr:
            scheduler.step()
        train_epoch_loss = running_loss / len(dataloaders['train'])
        all_outputs = torch.cat(all_outputs, dim=0).detach().numpy() 
        all_labels = torch.cat(all_labels, dim=0).detach().numpy()    
        class_acc, accs = calculate_accs(configs, all_outputs, all_labels)
        results = {'epoch': epoch, 'status': 'train', 'loss': train_epoch_loss, 'class_acc': class_acc, 'accuracy': accs}
        # wandb.log({'train_logs': results})
        print(results)
        log_results_train(configs, results)
        best_acc, valid_results = eval(dataloaders,model, configs, epoch, loss_functions, best_acc)

        if epoch>=0 and 'FlexRS' in configs.general.method:
            configs.datasets.sampler = 'FlexRS'
            valid_accuracy = valid_results['class_acc']
            datasets = get_datasets(configs)
            extra_info = {'valid_accuracy': valid_accuracy}
            sampler, shuffle = get_sampler.get_sampler(configs, datasets, **extra_info)
            dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], 
                                                                        batch_size=configs.datasets.batch_size, 
                                                                        shuffle=shuffle, num_workers=configs.general.num_workers, sampler = sampler)


    torch.save(model, 'outputs/%s/%s/last.pt' %(configs.general.dataset_name, configs.general.save_name))

def eval(dataloaders, model, configs, epoch, loss_functions, best_acc):
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()
    
    with torch.no_grad():
        running_loss = 0
        correct = list(0. for i in range(configs.general.num_classes))
        total = list(0. for i in range(configs.general.num_classes))
        all_labels = []
        all_outputs = []
        for data in tqdm.tqdm(dataloaders['val']):
            inputs, labels = data
            if configs.cuda.use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            if configs.general.method == 'BBN':
                feature_a, feature_b = (
                    model.forward_features(inputs),
                    model.forward_features(inputs),
                )
                feature_a, feature_b = (
                    model.forward_head(feature_a, pre_logits=True),
                    model.forward_head(feature_b, pre_logits=True),
                )

                l = 0.5
                configs.general.l = l
                mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)
                outputs = model.classifier(mixed_feature)
                val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'T-Norm':
                feature_x = model.forward_features(inputs)
                feature_x = model.forward_head(feature_x, pre_logits=True)

                weights = model.classifier.weight

                ws = pnorm(weights, 2)
                outputs = forward(ws, feature_x)
                val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'Logits_Adjust_Posthoc':
                cls_num_list = get_cls_num_list(dataloaders['train'].dataset)
                tau = 1.0
                base_probs = [x/max(cls_num_list) for x in cls_num_list]
                base_probs = torch.Tensor(base_probs).cuda()
                outputs = model(inputs)
                outputs = outputs - torch.log((base_probs**tau) + 1e-12)
                val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'KNN':
                model_ft, knn = model[0], model[1]
                feature_x = model_ft.forward_features(inputs)
                feature_x = model_ft.forward_head(feature_x, pre_logits=True)
                outputs = knn(feature_x)[0]
                val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'RSG':
                outputs = model(inputs, phase_train=False)
                val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'SADE':
                outputs = model(inputs)
                outputs = outputs['output']
                val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            else:
                outputs = model(inputs)
                val_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            correct, total = calculate_metrics_single(labels, outputs, correct, total)
            running_loss += val_loss.data.item()

        val_epoch_loss = running_loss / len(dataloaders['val'])
        valid_results = calculate_metrics(configs, all_outputs, all_labels)
        valid_results['epoch'] = epoch
        valid_results['status'] = 'val'
        valid_results['loss'] = val_epoch_loss
        
        # wandb.log({'val_logs': results})
        print(valid_results)
        log_results(configs, valid_results)
        current_acc = valid_results['accuracy'][3]
        if  current_acc> best_acc:
            print('Best acc: %s, current acc: %s. Saving best model...' %(round(best_acc, 4), round(current_acc, 4)))
            best_acc = current_acc
            torch.save(model, 'outputs/%s/%s/best.pt' %(configs.general.dataset_name, configs.general.save_name))
        
    with torch.no_grad():
        running_loss = 0
        correct = list(0. for i in range(configs.general.num_classes))
        total = list(0. for i in range(configs.general.num_classes))
        all_labels = []
        all_outputs = []
        for data in tqdm.tqdm(dataloaders['test']):
            inputs, labels = data
            if configs.cuda.use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            if configs.general.method == 'BBN':
                feature_a, feature_b = (
                    model.forward_features(inputs),
                    model.forward_features(inputs),
                )
                feature_a, feature_b = (
                    model.forward_head(feature_a, pre_logits=True),
                    model.forward_head(feature_b, pre_logits=True),
                )

                l = 0.5
                configs.general.l = l
                mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)
 
                outputs = model.classifier(mixed_feature)
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'T-Norm':
                feature_x = model.forward_features(inputs)
                feature_x = model.forward_head(feature_x, pre_logits=True)
                weights = model.classifier.weight
                ws = pnorm(weights, 2)
                outputs = forward(ws, feature_x)
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'Logits_Adjust_Posthoc':
                cls_num_list = get_cls_num_list(dataloaders['train'].dataset)
                tau = 1.0
                base_probs = [x/max(cls_num_list) for x in cls_num_list]
                base_probs = torch.Tensor(base_probs).cuda()
                outputs = model(inputs)
                outputs = outputs - torch.log((base_probs**tau) + 1e-12)
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'KNN':
                model_ft, knn = model[0], model[1]
                feature_x = model_ft.forward_features(inputs)
                feature_x = model_ft.forward_head(feature_x, pre_logits=True)
                outputs = knn(feature_x)[0]
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'RSG':
                outputs = model(inputs, phase_train=False)
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            elif configs.general.method == 'SADE':
                outputs = model(inputs)
                outputs = outputs['output']
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            else:
                outputs = model(inputs)
                test_loss = calculate_loss(configs, outputs, labels, loss_functions['val'])
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            correct, total = calculate_metrics_single(labels, outputs, correct, total)
            running_loss += test_loss.data.item()

        test_epoch_loss = running_loss / len(dataloaders['test'])
        test_results = calculate_metrics(configs, all_outputs, all_labels)
        test_results['epoch'] = epoch
        test_results['status'] = 'test'
        test_results['loss'] = test_epoch_loss
        
        # wandb.log({'val_logs': results})
        print(test_results)
        log_results(configs, test_results)
    return best_acc, valid_results

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    
    configs = setup_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = configs.cuda.gpu_id
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(configs.general.seed)
    print(configs)
    
    save_name =  current_time = '%s_%s_%s_%s_%s_%s_%s_%s_%s' %(
                                                configs.datasets.imbalance_ratio, 
                                                configs.general.method, 
                                                configs.general.img_size,
                                                configs.model.model_name,
                                                configs.model.pretrained,
                                                configs.datasets.batch_size,
                                                configs.general.seed,
                                                configs.general.train_epochs,
                                                configs.datasets.transforms.train
                                                )
    configs.general.save_name = save_name
    outputs_dir = 'outputs/%s/%s/' %(configs.general.dataset_name, configs.general.save_name)
    if os.path.exists(outputs_dir):
        [os.remove(os.path.join(outputs_dir, file_name)) for file_name in os.listdir(outputs_dir)]
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    # wandb.login()

    # run = wandb.init(
    #     # 此处设置你的项目名
    #     project="Long-tailed ISIC RS",
    #     # 此处配置需要Wandb帮你记录和track的参数
    #     config={
    #         "imbalance ratio": 100,
    #         "method": configs.general.method
    #     })
    main(configs)
    