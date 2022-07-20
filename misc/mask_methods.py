import torch


class RandomBoardShuffle:
    def __init__(self, x_num=4, y_num=4, size=8):
        self._len = x_num * y_num
        self._size = size
        self._x_num = x_num
        self._y_num = y_num

    def __call__(self, in_x):
        B, C, H, W = in_x.shape
        assert H == 32 and W == 32
        out_x = in_x.clone()
        out_x = out_x.view(
            B, C, self._x_num, self._size, self._y_num, self._size
        ).permute(2, 4, 0, 1, 3, 5)
        out_x = out_x.reshape(-1, B, C, self._size, self._size)
        out_x = out_x[torch.randperm(self._len)]
        out_x = out_x.view(
            self._x_num, self._y_num, B, C, self._size, self._size
        ).permute(2, 3, 0, 4, 1, 5) 
        out_x = out_x.reshape(B, C, H, W)
        return out_x


class MosaicMask:
    def __init__(self, ratio, x_num=4, y_num=4, size=8):
        self._len = x_num * y_num
        self._ratio = ratio
        self._size = size
        self._x_num = x_num
        self._y_num = y_num
        self._mask = torch.ones(self._len)
        self._mask[int(ratio * self._len):] = 0
    
    def permute_idx(self):
        self._mask = self._mask[torch.randperm(self._len)]

    def __call__(self, in_x, random=True):
        assert in_x.dim() == 4
        B, C, H, W = in_x.shape
        out_x = in_x.clone()
        out_x = out_x.view(
            B, C, self._x_num, self._size, self._y_num, self._size
        ).permute(0, 1, 3, 5, 2, 4)
        out_x = out_x.reshape(B, C, self._size, self._size, -1)
        if random:
            self.permute_idx()
        out_x = out_x * self._mask.to(out_x.device)
        out_x = out_x.reshape(
            B, C, self._size, self._size, self._x_num, self._y_num
        ).permute(0, 1, 4, 2, 5, 3)
        out_x = out_x.reshape(B, C, H, W)
        return out_x


class FixCenterMask:
    def __init__(self, x_start, x_end, y_start, y_end):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        ratio = (x_end - x_start) * (y_end - y_start) / 32**2
        print(f"This equal to {ratio} on 32x32 inputs.")

    def __call__(self, in_x):
        out_x = in_x.clone()
        out_x[:, :, self.x_start:self.x_end, self.y_start:self.y_end] = 0 
        return out_x