from PIL import Image
import numpy as np
with Image.open("Risk.png") as im:
    a = np.asarray(im)
    target_size = 12
    count = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            target = True
            for k in range(-target_size // 2, target_size // 2):
                limit = np.sqrt((target_size ** 2) // 4 - k ** 2)
                for l in range(-limit, limit):
                    if any([a[i + k, j + l][c] != 255 for c in range(3)]):
                        target = False
            if target:
                Image.Image.show(Image.fromarray(a[(i - target_size // 2 - 1):(i - target_size // 2 + 1),
                                                   (j - target_size // 2 - 1):(j - target_size // 2 + 1)]))
    print(count)