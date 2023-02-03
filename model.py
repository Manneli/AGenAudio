import torch
import dataloader as dl

def main():
    dataset = dl.AGenDataset("/home/anneli/AGenAudio/validation.dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = audio_model_clas()
    model = model.type(torch.float64)

    for audio, label in dataloader:
        print( model(audio) )
        break

class audio_model_clas(torch.nn.Module):

    def __init__(self, num_features=13 ,num_hidden=512 , num_class = 4, length = 1132):
        super(audio_model_clas,self).__init__()
        self.lstm = torch.nn.LSTM(input_size=num_features,hidden_size=num_hidden,num_layers=2,
                                   batch_first=True)

        self.flat = torch.nn.Flatten()
        self.lin = torch.nn.Linear(in_features=num_hidden*length,out_features=num_class)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.length = length


    def forward(self, input):
        x, states = self.lstm(input)
        if isinstance(input,torch.nn.utils.rnn.PackedSequence):
            x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=self.length)[0]
        x = self.flat(x)
        x = self.lin(x)
        #x = self.softmax(x)
        return x

class audio_model_reg(torch.nn.Module):

    def __init__(self, num_features=13 ,num_hidden=512, length = 1132):
        super(audio_model_reg,self).__init__()
        self.lstm = torch.nn.LSTM(input_size=num_features,hidden_size=num_hidden,num_layers=2,
                                   batch_first=True)

        self.flat = torch.nn.Flatten()
        self.lin = torch.nn.Linear(in_features=num_hidden*length,out_features=1)
        self.length = length


    def forward(self, input):
        x, states = self.lstm(input)
        if isinstance(input,torch.nn.utils.rnn.PackedSequence):
            x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=self.length)[0]
        x = self.flat(x)
        x = self.lin(x)
        return x


if __name__ == "__main__":
    main()
