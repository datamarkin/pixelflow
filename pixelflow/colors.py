# BGR format for OpenCV compatibility
# Scientifically chosen for maximum perceptual distinction

# 20 perceptually distinct colors optimized for visibility
# Ordered to maximize distinction between adjacent colors

DEFAULT_PALETTE = [
    (255, 127, 14),   # Vivid Orange
    (31, 119, 180),   # Strong Blue  
    (44, 160, 44),    # Green
    (214, 39, 40),    # Vivid Red
    (148, 103, 189),  # Purple
    (140, 86, 75),    # Brown
    (227, 119, 194),  # Pink
    (127, 127, 127),  # Gray
    (188, 189, 34),   # Olive
    (23, 190, 207),   # Cyan
    (255, 187, 120),  # Light Orange
    (174, 199, 232),  # Light Blue
    (152, 223, 138),  # Light Green
    (255, 152, 150),  # Light Red
    (197, 176, 213),  # Light Purple
    (196, 156, 148),  # Light Brown
    (247, 182, 210),  # Light Pink
    (199, 199, 199),  # Light Gray
    (219, 219, 141),  # Light Olive
    (158, 218, 229),  # Light Cyan
]

# Preset UI colors for consistent styling across the library
UI_COLORS = {
    'text': (255, 255, 255),           # White text (default)
    'text_dark': (0, 0, 0),            # Black text
    'text_accent': (0, 255, 255),      # Yellow accent text
    'background': (0, 0, 0),           # Black background
    'background_light': (255, 255, 255), # White background
    'background_overlay': (40, 40, 40),  # Dark overlay background
    'shadow': (50, 50, 50),            # Shadow color
    'success': (0, 255, 0),            # Green for success
    'warning': (0, 165, 255),          # Orange for warnings
    'error': (0, 0, 255),              # Red for errors
    'info': (255, 255, 0),             # Cyan for info
    'accent': (255, 0, 255),           # Magenta accent
    'grid': (128, 128, 128),           # Grid lines
    'border': (200, 200, 200),         # Border color
    'fps': (0, 255, 0),                # FPS counter color
}

# Alternative palette styles
PALETTES = {
    'default': DEFAULT_PALETTE,
    'vibrant': [
        (0, 0, 255),      # Pure Red
        (0, 255, 0),      # Pure Green  
        (255, 0, 0),      # Pure Blue
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (0, 128, 255),    # Orange
        (255, 0, 128),    # Purple
        (128, 255, 0),    # Lime
        (0, 128, 128),    # Olive
        (128, 0, 128),    # Maroon
        (128, 128, 0),    # Navy
        (64, 224, 208),   # Turquoise
        (250, 128, 114),  # Salmon
        (255, 215, 0),    # Gold
        (255, 105, 180),  # Hot Pink
        (0, 191, 255),    # Deep Sky Blue
        (50, 205, 50),    # Lime Green
        (255, 20, 147),   # Deep Pink
        (255, 140, 0),    # Dark Orange
    ],
    'pastel': [
        (203, 195, 255),  # Pastel Red
        (195, 255, 203),  # Pastel Green
        (255, 203, 195),  # Pastel Blue
        (195, 255, 255),  # Pastel Yellow
        (255, 195, 255),  # Pastel Magenta
        (255, 255, 195),  # Pastel Cyan
        (195, 225, 255),  # Pastel Orange
        (255, 195, 225),  # Pastel Purple
        (225, 255, 195),  # Pastel Lime
        (214, 234, 248),  # Pastel Sky
        (250, 219, 216),  # Pastel Rose
        (253, 235, 208),  # Pastel Peach
        (222, 234, 210),  # Pastel Mint
        (239, 224, 255),  # Pastel Lavender
        (255, 245, 215),  # Pastel Cream
        (230, 244, 241),  # Pastel Teal
        (255, 239, 213),  # Pastel Apricot
        (241, 238, 252),  # Pastel Periwinkle
        (255, 250, 230),  # Pastel Beige
        (240, 255, 240),  # Pastel Honeydew
    ],
}


class ColorManager:
    """
    Manages color assignment for classes and UI elements.
    
    Features:
    - Deterministic color assignment for class IDs
    - Semantic UI color presets
    - Support for multiple palette styles
    - Automatic color generation beyond palette size
    """
    
    def __init__(self, palette='default', seed=None):
        """
        Initialize ColorManager with specified palette.
        
        Args:
            palette: Palette name ('default', 'vibrant', 'pastel') or custom list of BGR tuples
            seed: Optional seed for reproducible color generation (not used in current implementation)
        """
        if isinstance(palette, str):
            self.palette = PALETTES.get(palette, DEFAULT_PALETTE).copy()
        elif isinstance(palette, list):
            self.palette = palette.copy()
        else:
            self.palette = DEFAULT_PALETTE.copy()
            
        self.assigned_colors = {}  # class_id -> color mapping
        self.ui_colors = UI_COLORS.copy()  # Mutable UI colors
        self.seed = seed
    
    def get_color(self, class_id):
        """
        Get deterministic color for a class ID.
        
        Args:
            class_id: Integer class identifier
            
        Returns:
            BGR color tuple for the class
        """
        if class_id not in self.assigned_colors:
            # Calculate color index
            idx = len(self.assigned_colors)
            
            if idx < len(self.palette):
                # Use color from palette
                color = self.palette[idx]
            else:
                # Generate additional color when palette is exhausted
                # Use modulo to cycle through palette with slight variations
                base_idx = idx % len(self.palette)
                base_color = self.palette[base_idx]
                
                # Apply brightness variation based on cycle number
                cycle = idx // len(self.palette)
                factor = 1.0 - (cycle * 0.2)  # Darken by 20% each cycle
                factor = max(0.3, factor)  # Don't go below 30% brightness
                
                color = tuple(int(c * factor) for c in base_color)
            
            self.assigned_colors[class_id] = color
            
        return self.assigned_colors[class_id]
    
    def ui(self, name):
        """
        Get UI color by semantic name.
        
        Args:
            name: UI element name ('text', 'background', 'shadow', etc.)
            
        Returns:
            BGR color tuple for the UI element
        """
        return self.ui_colors.get(name, (128, 128, 128))
    
    def set_ui(self, name, color):
        """
        Override a UI color.
        
        Args:
            name: UI element name to override
            color: BGR color tuple
        """
        self.ui_colors[name] = color
    
    def reset_assignments(self):
        """Reset class color assignments while keeping UI colors."""
        self.assigned_colors = {}
    
    def get_palette_info(self):
        """Get information about current palette."""
        return {
            'palette_size': len(self.palette),
            'assigned_classes': len(self.assigned_colors),
            'ui_colors': list(self.ui_colors.keys())
        }
