import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import pandas as pd
from inclearn.convnet.mobilenetv2_baseline import MobileNetV2
from inclearn.lib.icifar10 import iCIFAR10
from inclearn.lib.icifar100 import iCIFAR100


def main():
    parser = argparse.ArgumentParser(description='PyTorch MobileNet V2 CIFAR10 Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--test_data_exposure', action='store_true', help='save pickles according to test loss or not')
    parser.add_argument('--num_classes', default=5, type=int, help='number of classes to be learnt')
    parser.add_argument('--num_classes_old', nargs='?', const=None, default=None, type=int, help='number of old classes')
    parser.add_argument('--num_classes_old_old', nargs='?', const=None, default=None, type=int, help='number of old classes of old classes')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs to run')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--stats_fname', default=None, type=str, help='the name of the pickle file in which we save statistics')

    args = parser.parse_args()

    if args.num_classes_old == None and args.num_classes_old_old == None:
        s = 0
        e = args.num_classes
    else:
        s = args.num_classes_old
        e = args.num_classes

    if args.stats_fname != None:
        if not os.path.isdir('stats'):
            os.mkdir('stats')
        fpath = os.path.join('stats', args.stats_fname)

    num_classes = args.num_classes
    if args.num_classes_old == None:
        num_classes_old = num_classes
    else:
        num_classes_old = args.num_classes_old
    print("Loading training examples for classes", range(s, e))
    batch_size=128
    batch_size_test=128
    datasetGen = iCIFAR100

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasetGen(root='./data',
                        train=True,
                        classes=range(s,e),
                        download=True,
                        transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

    test_set = datasetGen(root='./data',
                        train=False,
                        classes=range(0, num_classes),
                        download=True,
                        transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test,
                                                shuffle=False, num_workers=2)

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    if args.resume:
         # Load checkpoint.
        print(f'==> Resuming from the checkpoint: ckpt_{num_classes_old}_{num_classes}.pth')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'./checkpoint/ckpt_{num_classes_old}_{num_classes}.pth')
        test_loss_min = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        model = MobileNetV2(num_classes=num_classes)
        model.load_state_dict(checkpoint['model'])
        n_inputs = model.linear.weight.shape[1]
        last_layer = nn.Linear(n_inputs, num_classes)
        model.linear = last_layer
    else:
        test_loss_min = float('inf')
        # Start a fresh learning
        if num_classes_old == num_classes:
            print(f'==> Starting learning with {num_classes} classes..')
            model = MobileNetV2(num_classes=num_classes)
            start_epoch = 1
            if args.stats_fname != None:
                df = pd.DataFrame(columns=["num_classes", "num_classes_old", "n_epochs","learning_rate" , "time_epochs", "confusion_matrix", "class_accuracy", "test_accuracy"])
                df.to_pickle(fpath)
        # Start a fresh increamental learning
        else:
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            if args.num_classes_old_old:
                print(f'==> Incremental learning from the checkpoint: ckpt_{args.num_classes_old_old}_{num_classes_old}.pth')
                checkpoint = torch.load(f'./checkpoint/ckpt_{args.num_classes_old_old}_{num_classes_old}.pth')
            else:
                try:
                    print(f'==> Incremental learning from the checkpoint: ckpt_{num_classes_old}_{num_classes_old}.pth')
                    checkpoint = torch.load(f'./checkpoint/ckpt_{num_classes_old}_{num_classes_old}.pth')
                except FileNotFoundError as e:
                    print(e)
                    print('* Use --num_classes_old_old to specify the checkpoint.')
            model = MobileNetV2(num_classes=num_classes_old)
            start_epoch = checkpoint['epoch']
            state_dict = checkpoint['model']
            n_inputs = state_dict['linear.weight'].shape[1]
            new_part = torch.zeros(num_classes - num_classes_old, n_inputs)
            new_part_bias = torch.ones(num_classes - num_classes_old)
            if train_on_gpu:
                new_part = new_part.cuda()
                new_part_bias = new_part_bias.cuda()
            state_dict['linear.weight'] = torch.cat([state_dict['linear.weight'], new_part])
            state_dict['linear.bias'] = torch.cat([state_dict['linear.bias'], new_part_bias])
            last_layer = nn.Linear(n_inputs, num_classes)
            model.linear = last_layer
            model.load_state_dict(state_dict)
            print(model.linear.weight)
            print(f"Model classes increased from {num_classes_old} to {num_classes}.")

    # if GPU is available, move the model to GPU
    # check if CUDA is available
    if train_on_gpu:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [49, 63], gamma=0.2)

    # number of epochs to train the model
    n_epochs = args.epochs
    t_epochs = 0

    stats = []
    stats.append(num_classes)
    stats.append(num_classes_old)
    stats.append(n_epochs)
    stats.append(args.lr)

    for epoch in range(start_epoch, start_epoch+n_epochs):
        
        ###################
        # train the model #
        ###################
        # model is set to train
        model.train()
        # keep track of training loss
        train_loss = 0.0
        
        t_start = time.time()
        for batch_i, (indices, data, target) in enumerate(train_loader):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss 
            train_loss += loss.item()

            if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
                print('Epoch %d, Batch %d. learning rate %.6f. training loss: %.16f' %
                    (epoch, batch_i + 1, optimizer.param_groups[0]['lr'], train_loss / 20))
                train_loss = 0.0
        t_end = time.time()
        t_one_epoch = t_end - t_start
        t_epochs += t_one_epoch

        scheduler.step()


        ##################
        # test the model #
        ##################
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for indices, data, target in test_loader:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update  test loss 
                test_loss += loss.item()*data.size(0)
            
        test_loss = test_loss/len(test_loader.dataset)
        print(f'* Epoch {epoch}. testing loss: {test_loss:.16f}. time used: {t_one_epoch:.3f} s')

        # Save model weights
        if (args.test_data_exposure and test_loss <= test_loss_min) or (not args.test_data_exposure):
            test_loss_min = test_loss
            print(f'Saving the checkpoint: ckpt_{num_classes_old}_{num_classes}.pth')
            state = {
                'model': model.state_dict(),
                'loss': test_loss_min,
                'epoch': epoch+1,
                'num_classes': num_classes,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'./checkpoint/ckpt_{num_classes_old}_{num_classes}.pth')
        else:
            pass
    
    stats.append(t_epochs)

    test_loss = 0.0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    checkpoint = torch.load(f'./checkpoint/ckpt_{num_classes_old}_{num_classes}.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval() # eval mode

    confusion_dict = {'class_'+str(i+1) : [0 for _ in range(num_classes)] for i in range(num_classes)}

    # iterate over test data
    for indices, data, target in test_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update  test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class and update confution dict
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
            # update confution dict
            confusion_dict['class_'+str(label.item()+1)][label.item()] += correct[i].item()
            if not correct[i].item():
                confusion_dict['class_'+str(label.item()+1)][pred[0].item()] += 1

    # calculate avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('\nTest Loss: {:.6f}\n'.format(test_loss))

    test_acc_class_dict = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            test_acc_class = 100 * class_correct[i] / class_total[i]
            print('Test Accuracy of class %5s: %2d%% (%2d/%2d)' % (
                str(i+1), test_acc_class,
                np.sum(class_correct[i]), np.sum(class_total[i])))
            test_acc_class_dict[str(i+1)] = test_acc_class
        else:
            print(f'Test Accuracy of class {str(i+1)}: N/A (no training examples)')

    test_acc_all = 100. * np.sum(class_correct) / np.sum(class_total)
    print('\nTest Accuracy (Overall): %.2f%% (%2d/%2d)\n' % (
        test_acc_all,
        np.sum(class_correct), np.sum(class_total)))
    stats.append(confusion_dict)
    stats.append(test_acc_class_dict)
    stats.append(test_acc_all)

    if args.stats_fname != None:
        df = pd.read_pickle(fpath)
        df.loc[len(df)] = stats
        df.to_pickle(fpath)


if __name__ == '__main__':
    main()