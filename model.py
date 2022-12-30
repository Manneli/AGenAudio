import torch
import dataloader as dl

def main():
    dataset = dl.AGenDataset("/home/anneli/AGenAudio/validation.dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = audio_model()
    model = model.type(torch.float64)

    for audio, label in dataloader:
        print( model(audio) )
        break


class audio_model(torch.nn.Module):

    def __init__(self, num_features=13 ,num_hidden=13 ,num_layers = 50, num_class = 10, length = 1132):
        super(audio_model,self).__init__()
        # TODO Mask
        self.lstm1 = torch.nn.LSTM(input_size=num_features,hidden_size=num_hidden,num_layers=num_layers,
                                   batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=num_hidden, hidden_size=num_hidden, num_layers=num_layers,
                                   batch_first=True)
        self.flat = torch.nn.Flatten()
        self.lin = torch.nn.Linear(in_features=num_hidden*length,out_features=num_class)
        self.softmax = torch.nn.Softmax(dim = 1)


    def forward(self, input):
        x, states = self.lstm1(input)
        x, states = self.lstm2(x)
        x = self.flat(x)
        x = self.lin(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    main()