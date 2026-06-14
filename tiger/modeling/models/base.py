import torch
import torch.nn as nn


class TorchModel(nn.Module):
    @torch.no_grad()
    def _init_weights(self, initializer_range):
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.normal_(value.data, mean=0.0, std=initializer_range)
            elif 'bias' in key:
                nn.init.zeros_(value.data)
            elif 'codebook' in key:
                nn.init.normal_(value.data, mean=0.0, std=initializer_range)
            elif 'bos_embedding' in key:
                nn.init.normal_(value.data, mean=0.0, std=initializer_range)
            else:
                raise ValueError(f'Unknown transformer weight: {key}')