from typing import List, Tuple


def convert_points(points_str: str) -> List[Tuple[int, int]]:
    """Convert space-separated coordinate pairs to list of integer tuples.
    
    Args:
        points_str: String with space-separated x,y coordinate pairs
                   Example: "867.87,720 413.59,378.44 482.07,363 578.66,407.33"
    
    Returns:
        List of tuples with integer coordinates
        Example: [(867, 720), (413, 378), (482, 363), (578, 407)]
    """
    if not points_str.strip():
        return []
    
    points = []
    for pair in points_str.strip().split():
        x_str, y_str = pair.split(',')
        x = int(round(float(x_str)))
        y = int(round(float(y_str)))
        points.append((x, y))
    
    return points


if __name__ == "__main__":
    # Test with the example from the user
    test_input = "551.6,357.85 700.48,411.76 863.15,473.62 1280,638.42 1280,532.07 870.34,419.41 583.27,343.72"
    result = convert_points(test_input)
    print(f"Input: {test_input}")
    print(f"Output: {result}")
    
    # Expected: [(867, 720), (414, 378), (482, 363), (579, 407)]