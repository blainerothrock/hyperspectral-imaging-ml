import torch
from . import TransformException, BaseTransform

class ToTensor(BaseTransform):
    """
    convert obspy Trace to Tensor
    Args:
        device (torch.device)
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: tensor
        inplace: (Bool): optional, will overwrite the source data with the trim
    Raises:
        TransformException
    Returns:
        data: a modified dictionary with filters applied
    """

    def __init__(self, device=torch.device('cpu'), source='raw', output='tensor', inplace=False):
        super().__init__(source, output, inplace)

        if not isinstance(device, torch.device):
            raise TransformException(f'device must be a torch.device, got {type(device)}')

        self.device = device

    def __call__(self, data):
        super().__call__(data)

        img, labels = data[self.source]
        img = img.copy()
        labels = labels.copy()

        img_tensor = torch.tensor(img, dtype=torch.float64, device=self.device)
        label_tensor = torch.tensor(labels, dtype=torch.float64, device=self.device)

        super().update(data, (img_tensor, label_tensor))

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'device: {self.device})'
        )