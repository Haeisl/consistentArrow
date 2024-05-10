def generate_grid_points(bounds, cols, rows):
        min_x, max_x, min_y, max_y = bounds
        grid_points = []

        # Compute the spacing between the points
        dx = (max_x - min_x) / (cols + 1)
        dy = (max_y - min_y) / (rows + 1)

        # Generate the grid points
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                x = min_x + j * dx
                y = min_y + i * dy
                grid_points.append((x, y))

        return grid_points


def main() -> None:
    grid = generate_grid_points([-2, 2, -2, 2], 3, 3)
    print(grid)

if __name__ == '__main__':
    main()