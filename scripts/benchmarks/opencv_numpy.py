import cv2
import time
import numpy as np

overlay = cv2.imread("../../examples/data/fireworks.png", cv2.IMREAD_UNCHANGED)

def opencv_rectangle_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return image

def numpy_rectangle_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = box
        image[y1:y1+thickness, x1:x2] = color
        image[y2-thickness:y2, x1:x2] = color
        image[y1:y2, x1:x1+thickness] = color
        image[y1:y2, x2-thickness:x2] = color
    return image

def opencv_filled_boxes(image, boxes, color=(0, 255, 0)):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, -1)
    return image

def numpy_filled_boxes(image, boxes, color=(0, 255, 0)):
    for box in boxes:
        x1, y1, x2, y2 = box
        image[y1:y2, x1:x2] = color
    return image

def opencv_rounded_rectangle_boxes(image, boxes, color=(0, 255, 0), thickness=2, radius=30):
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Create a temporary image for smooth blending
        temp = np.zeros_like(image)
        
        # Draw main rectangle edges
        cv2.rectangle(temp, (x1 + radius, y1), (x2 - radius, y1 + thickness), color, -1)
        cv2.rectangle(temp, (x1 + radius, y2 - thickness), (x2 - radius, y2), color, -1)
        cv2.rectangle(temp, (x1, y1 + radius), (x1 + thickness, y2 - radius), color, -1)
        cv2.rectangle(temp, (x2 - thickness, y1 + radius), (x2, y2 - radius), color, -1)
        
        # Draw square corners (top-right and bottom-left)
        cv2.rectangle(temp, (x2 - radius, y1), (x2, y1 + radius), color, -1)
        cv2.rectangle(temp, (x1, y2 - radius), (x1 + radius, y2), color, -1)
        
        # Draw smooth rounded corners using filled circles
        cv2.circle(temp, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(temp, (x2 - radius, y2 - radius), radius, color, -1)
        
        # Create inner circles to make it hollow
        inner_radius = radius - thickness
        if inner_radius > 0:
            cv2.circle(temp, (x1 + radius, y1 + radius), inner_radius, (0, 0, 0), -1)
            cv2.circle(temp, (x2 - radius, y2 - radius), inner_radius, (0, 0, 0), -1)
        
        # Apply anti-aliasing by blending with original
        mask = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
        mask = mask.astype(np.float32) / 255.0
        
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - mask) + temp[:, :, c] * mask
    
    return image

def numpy_rounded_rectangle_boxes(image, boxes, color=(0, 255, 0), thickness=2, radius=30):
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Draw main rectangle edges
        image[y1:y1+thickness, x1+radius:x2-radius] = color
        image[y2-thickness:y2, x1+radius:x2-radius] = color
        image[y1+radius:y2-radius, x1:x1+thickness] = color
        image[y1+radius:y2-radius, x2-thickness:x2] = color
        
        # Draw square corners (top-right and bottom-left)
        image[y1:y1+radius, x2-radius:x2] = color
        image[y2-radius:y2, x1:x1+radius] = color
        
        # Draw smooth rounded corners using anti-aliased circles
        cy1, cx1 = y1 + radius, x1 + radius  # Top-left center
        cy2, cx2 = y2 - radius, x2 - radius  # Bottom-right center
        
        # Create coordinate grids for anti-aliasing
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Calculate distance from center with sub-pixel precision
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Anti-aliasing: calculate coverage for each pixel
                if dist <= radius:
                    # Calculate how much of the pixel is covered by the ring
                    outer_coverage = min(1.0, max(0.0, radius - dist + 0.5))
                    inner_coverage = min(1.0, max(0.0, (radius - thickness) - dist + 0.5)) if radius > thickness else 0.0
                    coverage = max(0.0, outer_coverage - inner_coverage)
                    
                    if coverage > 0:
                        # Top-left corner
                        py1, px1 = cy1 + dy, cx1 + dx
                        if 0 <= py1 < image.shape[0] and 0 <= px1 < image.shape[1]:
                            for c in range(3):
                                image[py1, px1, c] = int(image[py1, px1, c] * (1 - coverage) + color[c] * coverage)
                        
                        # Bottom-right corner
                        py2, px2 = cy2 + dy, cx2 + dx
                        if 0 <= py2 < image.shape[0] and 0 <= px2 < image.shape[1]:
                            for c in range(3):
                                image[py2, px2, c] = int(image[py2, px2, c] * (1 - coverage) + color[c] * coverage)
    
    return image

def opencv_blend_transparent(image, overlay_path, positions):

    if overlay is None:
        return image
    
    for pos in positions:
        x, y = pos
        h, w = overlay.shape[:2]
        
        # Check bounds
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            continue
            
        # Extract alpha channel if exists
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
            overlay_rgb = overlay[:, :, :3]
        else:
            alpha = np.ones((h, w))
            overlay_rgb = overlay
            
        # Blend using OpenCV
        roi = image[y:y+h, x:x+w]
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_rgb[:, :, c] * alpha
        
        image[y:y+h, x:x+w] = roi
    
    return image

def numpy_blend_transparent(image, overlay_path, positions):
    if overlay is None:
        return image
    
    for pos in positions:
        x, y = pos
        h, w = overlay.shape[:2]
        
        # Check bounds
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            continue
            
        # Extract alpha channel if exists
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
            overlay_rgb = overlay[:, :, :3].astype(np.float32)
        else:
            alpha = np.ones((h, w, 1), dtype=np.float32)
            overlay_rgb = overlay.astype(np.float32)
            
        # Pure numpy blending
        roi = image[y:y+h, x:x+w].astype(np.float32)
        blended = roi * (1 - alpha) + overlay_rgb * alpha
        image[y:y+h, x:x+w] = blended.astype(np.uint8)
    
    return image


image = cv2.imread("../../examples/data/delhi.jpg")
boxes = [(100, 100, 200, 200), (300, 300, 400, 400), (50, 250, 150, 350), (450, 50, 550, 150), (200, 450, 350, 550), (500, 200, 600, 300), (150, 150, 250, 250)]

# OpenCV benchmark
opencv_image = image.copy()
start = time.time()
for _ in range(1000):
    opencv_rectangle_boxes(opencv_image, boxes)
end = time.time()
opencv_time = end - start
opencv_fps = 1000 / opencv_time
print(f"OpenCV Draw Boxes: {opencv_time:.6f} seconds ({opencv_fps:.2f} FPS)")

# Save OpenCV result
cv2.imwrite("opencv_numpy_opencv_modified.png", opencv_image)

# Numpy benchmark
numpy_image = image.copy()
start = time.time()
for _ in range(1000):
    numpy_rectangle_boxes(numpy_image, boxes)
end = time.time()
numpy_time = end - start
numpy_fps = 1000 / numpy_time
print(f"Numpy Draw Boxes: {numpy_time:.6f} seconds ({numpy_fps:.2f} FPS)")

# Save Numpy result
cv2.imwrite("opencv_numpy_numpy_modified.png", numpy_image)


# OpenCV benchmark
opencv_image = image.copy()
start = time.time()
for _ in range(1000):
    opencv_filled_boxes(opencv_image, boxes)
end = time.time()
opencv_time = end - start
opencv_fps = 1000 / opencv_time
print(f"OpenCV Filled Boxes: {opencv_time:.6f} seconds ({opencv_fps:.2f} FPS)")

# Save OpenCV result
cv2.imwrite("opencv_numpy_opencv_filled_modified.png", opencv_image)

# Numpy benchmark
numpy_image = image.copy()
start = time.time()
for _ in range(1000):
    numpy_filled_boxes(numpy_image, boxes)
end = time.time()
numpy_time = end - start
numpy_fps = 1000 / numpy_time
print(f"Numpy Filled Draw Boxes: {numpy_time:.6f} seconds ({numpy_fps:.2f} FPS)")

# Save Numpy result
cv2.imwrite("opencv_numpy_numpy_filled_modified.png", numpy_image)


# OpenCV rounded benchmark
# opencv_image = image.copy()
# start = time.time()
# for _ in range(100):
#     opencv_rounded_rectangle_boxes(opencv_image, boxes)
# end = time.time()
# opencv_time = end - start
# opencv_fps = 1000 / opencv_time
# print(f"OpenCV Rounded Boxes: {opencv_time:.6f} seconds ({opencv_fps:.2f} FPS)")
#
# # Save OpenCV rounded result
# cv2.imwrite("opencv_numpy_opencv_rounded_modified.png", opencv_image)
#
# # Numpy rounded benchmark
# numpy_image = image.copy()
# start = time.time()
# for _ in range(100):
#     numpy_rounded_rectangle_boxes(numpy_image, boxes)
# end = time.time()
# numpy_time = end - start
# numpy_fps = 1000 / numpy_time
# print(f"Numpy Rounded Boxes: {numpy_time:.6f} seconds ({numpy_fps:.2f} FPS)")
#
# # Save Numpy rounded result
# cv2.imwrite("opencv_numpy_numpy_rounded_modified.png", numpy_image)


# Blend positions for fireworks overlay
blend_positions = [(50, 50), (300, 100), (150, 300), (400, 200), (200, 400), 
                   (100, 150), (350, 350), (75, 400), (450, 100), (250, 50), 
                   (175, 225), (425, 275), (25, 200), (375, 425), (225, 325)]

# OpenCV blend benchmark
opencv_image = image.copy()
start = time.time()
for _ in range(100):
    opencv_blend_transparent(opencv_image, "../../examples/data/fireworks.png", blend_positions)
end = time.time()
opencv_time = end - start
opencv_fps = 1000 / opencv_time
print(f"OpenCV Blend Transparent: {opencv_time:.6f} seconds ({opencv_fps:.2f} FPS)")

# Save OpenCV blend result
cv2.imwrite("opencv_numpy_opencv_blend_modified.png", opencv_image)

# Numpy blend benchmark
numpy_image = image.copy()
start = time.time()
for _ in range(100):
    numpy_blend_transparent(numpy_image, "../../examples/data/fireworks.png", blend_positions)
end = time.time()
numpy_time = end - start
numpy_fps = 1000 / numpy_time
print(f"Numpy Blend Transparent: {numpy_time:.6f} seconds ({numpy_fps:.2f} FPS)")

# Save Numpy blend result
cv2.imwrite("opencv_numpy_numpy_blend_modified.png", numpy_image)