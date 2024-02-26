import torch
import torch.nn as nn

# direction = 'vertical' # 'horizontal' or 'vertical
# if direction == 'horizontal':
#     print(' ======================= Horizontal ======================= ')
# elif direction == 'vertical':
#     print(' ======================= Vertical ======================= ')
# elif direction == 'combined':
#     print(' ======================= Combined ======================= ')

class BiLSTMImage(nn.Module):
    def __init__(self, lstm_hidden_size, direction):
        super(BiLSTMImage, self).__init__()

        self.direction = direction
        self.horizontal_lstm = nn.LSTM(3, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.vertical_lstm = nn.LSTM(3, lstm_hidden_size, batch_first=True, bidirectional=True)

        # FC layer to map back to the original shape
        self.fc = nn.Linear(4 * lstm_hidden_size, 1)

        self.fc_single = nn.Linear(2 * lstm_hidden_size, 1)

    def forward(self, x):
        # Input shape：[batch_size, 3, 256, 512]





        # # Method 1
        # combined_features = (horizontal_output + vertical_output) / 2.0
        # output = combined_features.mean(dim=-1)

        # Method 2 - Concatenate features and apply FC layer
        # combined_features = torch.cat((horizontal_output, vertical_output), dim=-1)
        # output = self.fc(combined_features).squeeze(-1)

        # Method 3 - Apply FC layer to single direction and add the outputs
        if self.direction == 'horizontal':
            # Horizontal BiLSTM
            horizontal_input = x.permute(0, 2, 3, 1).contiguous().view(-1, 256, 3)
            horizontal_output, _ = self.horizontal_lstm(horizontal_input)
            horizontal_output = horizontal_output.view(-1, 256, 256,
                                                       2 * lstm_hidden_size)  # 2*lstm_hidden_size due to bidirectional
            output = self.fc_single(horizontal_output).squeeze(-1)
        elif self.direction == 'vertical':
            # Vetical BiLSTM
            vertical_input = x.permute(0, 3, 2, 1).contiguous().view(-1, 256, 3)
            vertical_output, _ = self.vertical_lstm(vertical_input)
            vertical_output = vertical_output.view(-1, 256, 256, 2 * lstm_hidden_size).permute(0, 2, 1, 3)
            output = self.fc_single(vertical_output).squeeze(-1)
        elif self.direction == 'combined':
            # Horizontal BiLSTM
            horizontal_input = x.permute(0, 2, 3, 1).contiguous().view(-1, 256, 3)
            horizontal_output, _ = self.horizontal_lstm(horizontal_input)
            horizontal_output = horizontal_output.view(-1, 256, 256,
                                                       2 * lstm_hidden_size)  # 2*lstm_hidden_size due to bidirectional

            # Vetical BiLSTM
            vertical_input = x.permute(0, 3, 2, 1).contiguous().view(-1, 256, 3)
            vertical_output, _ = self.vertical_lstm(vertical_input)
            vertical_output = vertical_output.view(-1, 256, 256, 2 * lstm_hidden_size).permute(0, 2, 1, 3)

            combined_features = torch.cat((horizontal_output, vertical_output), dim=-1)
            output = self.fc(combined_features).squeeze(-1)

        return output  # output shape：[batch_size, 256, 512]


lstm_hidden_size = 128
# model = BiLSTMImage(lstm_hidden_size, direction='combined')
# # input_tensor = torch.randn(4, 3, 256, 512)
# input_tensor = torch.randn(4, 3, 256, 256)
#
# output_tensor = model(input_tensor)
#
# # output shape should be [1, 256, 512]
# print(output_tensor.shape)
