import numpy as np
import torch
import math

"""
LogSparse Transformer, which only needs to calculate O(log L) dot products
for each cell in each layer.
"""


class LogSparseMask(torch.nn.Module):
    def __init__(self, hist_steps, sub_len):
        super(LogSparseMask, self).__init__()
        mask = torch.zeros((hist_steps, hist_steps), dtype=torch.float)
        for i in range(hist_steps):
            mask[i] = self.log_row_mask(i, sub_len, hist_steps)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def log_row_mask(self, index, sub_len, hist_steps):

        log_l = math.ceil(np.log2(sub_len))
        row_mask = torch.zeros((hist_steps), dtype=torch.float)
        if((hist_steps // sub_len) * 2 * (log_l) > index):
            row_mask[:(index + 1)] = 1
        else:
            while(index >= 0):
                if((index - log_l + 1) < 0):
                    row_mask[:index] = 1
                    break
                row_mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if((index - new_index) <= sub_len and new_index >= 0):
                        row_mask[new_index] = 1
                index -= sub_len
        return row_mask
