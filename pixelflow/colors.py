# RBG
# BGR

DEFAULT_COLOR_PALETTE = [
    (3, 154, 30),
    (36, 139, 251),
    (20, 100, 227),
    (92, 76, 15),
    (64, 15, 95),
    (30, 3, 154),
    (64, 15, 95),
    (36, 139, 251),
    (20, 100, 227),
    (92, 76, 15)
]


class ColorManager:
    def __init__(self, palette=None):
        self.palette = palette if palette is not None else DEFAULT_COLOR_PALETTE
        self.used_colors = {}

    def get_color(self, idx):
        if idx not in self.used_colors:
            current_index = len(self.used_colors)
            self.used_colors[idx] = self.palette[current_index + 1]
            return self.used_colors[idx]
        else:
            return self.used_colors[idx]