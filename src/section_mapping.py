def create_grid(frame_width, frame_height, n_cols, n_rows):
    """
    Creates a grid of dictionaries describing each cell:
    {
      'id': numeric_index,
      'x_min': float,
      'x_max': float,
      'y_min': float,
      'y_max': float
    }
    """
    grid = []
    cell_width = frame_width / n_cols
    cell_height = frame_height / n_rows
    for row in range(n_rows):
        for col in range(n_cols):
            cell = {
                'id': row*n_cols + col,
                'x_min': col*cell_width,
                'x_max': (col+1)*cell_width,
                'y_min': row*cell_height,
                'y_max': (row+1)*cell_height
            }
            grid.append(cell)
    return grid

def get_region_for_point(x, y, grid):
    """
    Given (x, y) and a grid of sections,
    returns the 'id' of the grid cell containing that point.
    If out of bounds, returns None.
    """
    for cell in grid:
        if (cell['x_min'] <= x < cell['x_max'] and
            cell['y_min'] <= y < cell['y_max']):
            return cell['id']
    return None
