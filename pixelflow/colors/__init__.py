import random

DEFAULT_COLOR_PALETTE = [
    "5F0F40", "9A031E", "FB8B24", "E36414", "0F4C5C"
]


class ColorManager:
    def __init__(self, palette=None):
        self.palette = palette if palette is not None else DEFAULT_COLOR_PALETTE
        self.object_colors = {}
        self.random_color_start_index = len(self.palette)

    def hex_to_rgb(self, hex_color):
        """
        Converts a hex color string to an RGB tuple.
        Example: "FF4040" -> (255, 64, 64)
        """
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def generate_random_color(self):
        """
        Generates a random RGB color.
        """
        return tuple(random.randint(0, 255) for _ in range(3))

    def get_color_for_object(self, object_id):
        """
        Returns a color (RGB tuple) for the given object_id.
        If it's a new object_id, assign a color from the palette or generate a random color.
        """
        if object_id not in self.object_colors:
            # If there are still colors in the palette, assign the next one
            if len(self.object_colors) < len(self.palette):
                hex_color = self.palette[len(self.object_colors)]
                rgb_color = self.hex_to_rgb(hex_color)
            else:
                # Generate a random color if we run out of the predefined palette
                rgb_color = self.generate_random_color()
            self.object_colors[object_id] = rgb_color
        return self.object_colors[object_id]
