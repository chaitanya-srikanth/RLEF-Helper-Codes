def resize_polygon_annotations(original_size, new_size, polygons):
    """
    Resize polygons to fit a new image size.

    Parameters:
        original_size (tuple): Original image size as (width, height).
        new_size (tuple): New image size as (width, height).
        polygons (list of lists): List of polygons where each polygon is a list of tuples
                                  [(x, y), (x, y), ...].

    Returns:
        list of lists: Resized polygons as a list of lists of tuples.
    """
    original_width, original_height = original_size
    new_width, new_height = new_size
    
    resized_polygons = []
    
    for polygon in polygons:
        resized_polygon = []
        
        # Calculate scaling factors
        x_scale = new_width / original_width
        y_scale = new_height / original_height
        for (x, y) in polygon:
            # Resize the polygon vertex coordinates
            new_x = x * x_scale
            new_y = y * y_scale
            resized_polygon.append((new_x, new_y))
        
        # Append resized polygon
        resized_polygons.append(resized_polygon)
    
    return resized_polygons

# Example usage:
# original_size = (4656, 3496)
# new_size = (640, 480)
# polygons = [
#     [(100, 150), (200, 150), (200, 250), (100, 250)],  # Example polygon
#     [(300, 400), (400, 400), (400, 500), (300, 500)]   # Another example polygon
# # ]

# resized_polygons = resize_polygon_annotations(original_size, new_size, polygons)
# print(resized_polygons)
