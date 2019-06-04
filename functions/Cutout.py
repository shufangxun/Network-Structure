import numpy as np
import torch

class Cutout:
     """https://github.com/raeidsaqur/Cutout
        
        Args:
            holes (int): Number of patches to cut out of each image. default = 1 
            patchsize (int): The length (in pixels) of each square patch. default = 16
    """
    def __init__(self, holes, patchsize):
        self.holes = holes
        self.patchsize = patchsize
    
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h,w), no.float32)

        for n in range(holes):
            # 随机取中心点
            y = np.random.randint(h)
            x = np.random.randint(w)

            # 固定尺寸
            y1 = np.clip(y - self.patchsize // 2, 0, h)
            y2 = np.clip(y + self.patchsize // 2, 0, h)
            x1 = np.clip(x - self.patchsize // 2, 0, w)
            x2 = np.clip(x + self.patchsize // 2, 0, w)

            mask[y1:y2, x1:x2] = 0
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

        