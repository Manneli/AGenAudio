import dataloader as dl
import torch
from model import audio_model_reg
from model import audio_model_clas
import numpy as np

NUM_CLASS = 10
REG = False
MAX_LENGTH = 1132
AGE_RANGE = [7,80]
EMD = True

def main():

    dataset_train = dl.AGenDataset("/home/anneli/AGenAudio/train.dataset")
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True,drop_last = True)

    dataset_test = dl.AGenDataset("/home/anneli/AGenAudio/test.dataset")
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=True, drop_last=False)

    ci_test = makeRealMean(dataloader_test,NumClass=NUM_CLASS)

    if REG :
        model = audio_model_reg(length=MAX_LENGTH)
    else:
        model = audio_model_clas(num_class=NUM_CLASS,length=MAX_LENGTH)
    model = model.type(torch.float64)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    if REG:
        loss = torch.nn.MSELoss()
    else:
        if EMD:
            loss = EMDLoss
        else:
            loss = torch.nn.CrossEntropyLoss()
            loss = loss.cuda()

    num = 0
    num_no_bet = 0
    best_eval_loss = 5000

    print(loss)
    print(model)

    while(True):#num_no_bet < 8):
        print("Epoch ",num)

        train(model,dataloader_train,optimizer,loss,epochs=1,num_classes=NUM_CLASS,pri=False,reg=REG)
        eval_loss = eval(model,dataloader_test,ci_test,reg=REG,num_classes=NUM_CLASS)

        torch.save(model.state_dict(), "/home/anneli/AGenAudio/models/model_reg{}_ep{}_loss{}.pt".format(REG,num,eval_loss))

        num += 1
        num_no_bet += 1

        if (eval_loss < best_eval_loss):
            best_eval_loss = eval_loss
            num_no_bet = 0





def train(model,dataloader,optimizer,loss_function,epochs=5,num_classes=10,pri = False,reg = False):
    num = 0
    for ep in range(epochs):
        if pri:
            print("epochs: ", ep)
        for audio,length, label in dataloader:
            label = label.cuda()

            input = torch.nn.utils.rnn.pack_padded_sequence(audio, lengths=length, batch_first=True, enforce_sorted=False)
            input = input.cuda()

            #input = torch.slice_copy(audio,dim= 2,start=0,end=100,step=1)

            optimizer.zero_grad()


            output = model(input)
            if reg:
                output = torch.reshape(output,[output.shape[0]])
                loss = torch.sqrt(loss_function(output, label.type(torch.float64)))
            else:
                truth = make_truth(label, num_classes, AGE_RANGE[0], AGE_RANGE[1])
                loss = loss_function(output, truth)

            # print("-----------------------------")
            #print(output)
            # print(truth)
            # print(label)
            #outlab = []
            #for i in output:
            #    outlab = outlab + [int(torch.argmax(i))]
            #print(outlab)
            #print(loss)
            loss.backward()
            optimizer.step()

            if pri:
                print("------------",ep)
                print("loss: ", loss)
                print("output: ",output)
                print("label: ",label)

            num += 1
            if num % 100 == 0:
                print(num)
                print(truth)


def eval (model,dataloader,ci,num_classes=10,reg=False):
    loss_function = torch.nn.MSELoss().cuda()

    loss = 0
    num = 0
    for audio,length,label in dataloader:
        label = label.cuda()

        input = torch.nn.utils.rnn.pack_padded_sequence(audio, lengths=length, batch_first=True, enforce_sorted=False)
        input = input.cuda()

        output = model(input)
        output = torch.nn.Softmax(dim = 1) (output)
        if reg:
            output = torch.reshape(output, [output.shape[0]])
        else:
            outlab = []
            for i in output:
                age = 0
                for j in range(num_classes):
                    age += i[j]*ci[j]
                outlab = outlab + [age]
            output = torch.as_tensor(outlab).cuda()

        loss += torch.sqrt(loss_function(output, label.type(torch.float64)))
        loss = loss.detach()
        num += 1

    print("------------------- Eval -------------------")
    print("loss: ", loss/num)
    print("")
    return loss/num

def makeRealMean(dataloader,NumClass=10):
    data = []
    for i in range(NumClass):
        data = data + [[]]

    for audio,length,label in dataloader:
        truth = make_truth(label,NumClass,AGE_RANGE[0],AGE_RANGE[1])
        for i in range(len(label)) :
            data[truth[i]] = data[truth[i]]+[label[i]]

    for i in range(NumClass):
        data[i] = sum(data[i])/len(data[i])
    return data




def make_truth (label,bins,start,end):
    label = label.cpu().numpy()
    truths = np.zeros(label.shape)
    dist = (end-start)/bins
    for i in range(bins):
        truths = truths + ( (label >= ((i*dist)+start)) *  (label < (( (i+1) *dist)+start)) ) * i
    return torch.as_tensor(truths.astype(int)).cuda()

def EMDLoss(outputs, truths):
    batchsize = outputs.shape[0]
    loss = 0
    for i in range(batchsize):
        loss_ = 0
        cdfp = 0
        cdft = 0
        for j in range(outputs[i].shape[0]):
            cdfp += outputs[i][j]
            cdft += 0 if j < truths[i] else 1
            loss_ += torch.square(cdfp-cdft)
        loss += loss_
    return loss





if __name__ == "__main__":
    main()

