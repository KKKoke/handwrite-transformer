import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions.uniform import Uniform
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WordEmbeddingFromScratch(L.LightningModule):
    def __init__(self):
        super(WordEmbeddingFromScratch, self).__init__()

        min_value = -0.5
        max_value = 0.5

        self.input1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.output1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        input = input.unsqueeze(0)
        inputs_to_top_hidden = ((input[0] * self.input1_w1) + (input[1] * self.input2_w1) + (input[2] * self.input3_w1) + (input[3] * self.input4_w1))
        inputs_to_bottom_hidden = ((input[0] * self.input1_w2) + (input[1] * self.input2_w2) + (input[2] * self.input3_w2) + (input[3] * self.input4_w2))

        output1 = ((inputs_to_top_hidden * self.output1_w1) + (inputs_to_bottom_hidden * self.output1_w2))
        output2 = ((inputs_to_top_hidden * self.output2_w1) + (inputs_to_bottom_hidden * self.output2_w2))
        output3 = ((inputs_to_top_hidden * self.output3_w1) + (inputs_to_bottom_hidden * self.output3_w2))
        output4 = ((inputs_to_top_hidden * self.output4_w1) + (inputs_to_bottom_hidden * self.output4_w2))

        """
        tips: torch.stack(), We just returned a list of the output values by wrapping them up in square brackets.
        Then the gradients would get stripped off and we would not be able to do backpropagation.
        So by using torch.stack(), we can return a list that preserves the order of gradients.
        """
        output_presoftmax = torch.stack([output1, output2, output3, output4])

        return output_presoftmax

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.1)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i[0])
        return loss


modelFromScratch = WordEmbeddingFromScratch()

print("Before optimization, the parameters are...")
for name, param in modelFromScratch.named_parameters():
    print(name, param.data)

data = {
    "w1": [modelFromScratch.input1_w1.item(), modelFromScratch.input2_w1.item(),
           modelFromScratch.input3_w1.item(), modelFromScratch.input4_w1.item()],
    "w2": [modelFromScratch.input1_w2.item(), modelFromScratch.input2_w2.item(),
           modelFromScratch.input3_w2.item(), modelFromScratch.input4_w2.item()],
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
print(df)

sns.scatterplot(data=df, x="w1", y="w2")
plt.text(df.w1[0], df.w2[0], df.token[0], horizontalalignment="left", size='medium', color='black', weight='semibold')
plt.text(df.w1[1], df.w2[1], df.token[1], horizontalalignment="left", size='medium', color='black', weight='semibold')
plt.text(df.w1[2], df.w2[2], df.token[2], horizontalalignment="left", size='medium', color='black', weight='semibold')
plt.text(df.w1[3], df.w2[3], df.token[3], horizontalalignment="left", size='medium', color='black', weight='semibold')
plt.show()

trainer = L.Trainer(max_epochs=100)
trainer.fit(modelFromScratch, train_dataloaders=DataLoader)

data = {
    "w1": [modelFromScratch.input1_w1.item(), modelFromScratch.input2_w1.item(),
           modelFromScratch.input3_w1.item(), modelFromScratch.input4_w1.item()],
    "w2": [modelFromScratch.input1_w2.item(), modelFromScratch.input2_w2.item(),
           modelFromScratch.input3_w2.item(), modelFromScratch.input4_w2.item()],
    "token": ["Troll2", "is", "great", "Gymkata"],
    "input": ["input1", "input2", "input3", "input4"]
}
df = pd.DataFrame(data)
print(df)

sns.scatterplot(data=df, x="w1", y="w2")
plt.text(df.w1[0], df.w2[0], df.token[0], horizontalalignment="left", size='medium', color='black', weight='semibold')
plt.text(df.w1[1], df.w2[1], df.token[1], horizontalalignment="left", size='medium', color='black', weight='semibold')
plt.text(df.w1[2], df.w2[2], df.token[2], horizontalalignment="left", size='medium', color='black', weight='semibold')
plt.text(df.w1[3], df.w2[3], df.token[3], horizontalalignment="left", size='medium', color='black', weight='semibold')
plt.show()
