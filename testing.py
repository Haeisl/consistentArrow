import numpy as np

def generate_grid_points(bounds, cols, rows):
        min_x, max_x, min_y, max_y = bounds

        # Generate the linearly spaced points between min and max for x and y
        x = np.linspace(min_x + (max_x - min_x) / (cols + 1), max_x - (max_x - min_x) / (cols + 1), cols)
        y = np.linspace(min_y + (max_y - min_y) / (rows + 1), max_y - (max_y - min_y) / (rows + 1), rows)

        # Generate grid points using meshgrid
        xv, yv = np.meshgrid(x, y)

        # Combine x and y coordinates into a single 2D array
        grid_points = np.column_stack([xv.ravel(), yv.ravel()])

        return grid_points


def main() -> None:
    grid = generate_grid_points([-2, 2, -2, 2], 2, 2)
    print(grid)

if __name__ == '__main__':
    main()