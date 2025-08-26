import cv2
import numpy as np
import os

# Import the palettes from colors.py
from pixelflow.colors import DEFAULT_PALETTE, PALETTES, UI_COLORS

def create_palette_image(colors, filename, title=""):
    """Create a PNG image showing colors as side-by-side columns."""
    column_width = 50
    column_height = 250
    
    # Create image with enough width for all colors
    image_width = len(colors) * column_width
    image = np.zeros((column_height, image_width, 3), dtype=np.uint8)
    
    # Fill each column with its color
    for i, color in enumerate(colors):
        x_start = i * column_width
        x_end = (i + 1) * column_width
        image[:, x_start:x_end] = color  # BGR format
    
    # Save the image
    cv2.imwrite(filename, image)
    print(f"Created {filename} with {len(colors)} colors")

def main():
    # Create output directory if it doesn't exist
    output_dir = "palette_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate default palette image
    create_palette_image(
        DEFAULT_PALETTE, 
        os.path.join(output_dir, "default_palette.png"),
        "Default Palette"
    )
    
    # Generate images for all named palettes
    for palette_name, colors in PALETTES.items():
        if palette_name == 'default':
            continue  # Already created above
            
        filename = f"{palette_name}_palette.png"
        create_palette_image(
            colors,
            os.path.join(output_dir, filename),
            f"{palette_name.title()} Palette"
        )
    
    # Generate UI colors image
    ui_colors_list = list(UI_COLORS.values())
    create_palette_image(
        ui_colors_list,
        os.path.join(output_dir, "ui_colors.png"),
        "UI Colors"
    )
    
    print(f"\nAll palette images created in '{output_dir}' directory:")
    print("- default_palette.png")
    print("- vibrant_palette.png") 
    print("- pastel_palette.png")
    print("- ui_colors.png")

if __name__ == "__main__":
    main()