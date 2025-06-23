import numpy as np
from PIL import Image, ImageDraw


def bbox_highlight(image, bboxes):
    """
    Highlights grid cells that overlap with bounding boxes.

    Parameters:
    - image: PIL.Image object representing the image.
    - bboxes: List of bounding boxes, where each bbox is a tuple (left, top, right, bottom).

    Returns:
    - A 24x24 numpy array where cells overlapping with any bbox are marked as 1, others as 0.
    """
    # Get image dimensions
    width, height = image.size

    # Calculate the size of each grid cell
    cell_width = width / 24.0
    cell_height = height / 24.0

    # Initialize the grid with zeros
    grid = np.zeros((24, 24), dtype=int)

    # Iterate over each cell in the grid
    for i in range(24):
        for j in range(24):
            # Define the boundaries of the current grid cell
            cell_left = i * cell_width
            cell_right = (i + 1) * cell_width
            cell_top = j * cell_height
            cell_bottom = (j + 1) * cell_height

            # Check if this cell overlaps with any bounding box
            for bbox in bboxes:
                bbox_left, bbox_top, bbox_right, bbox_bottom = bbox

                # Calculate the overlapping rectangle
                overlap_left = max(cell_left, bbox_left)
                overlap_top = max(cell_top, bbox_top)
                overlap_right = min(cell_right, bbox_right)
                overlap_bottom = min(cell_bottom, bbox_bottom)

                # If the overlapping area is positive, mark the cell as 1
                if overlap_left < overlap_right and overlap_top < overlap_bottom:
                    grid[j][i] = 1  # Note: row index is 'j', column index is 'i'
                    break  # No need to check other bounding boxes

    return grid.flatten().tolist()


def render_grid(image, grid, output_filename):
    """
    Renders the grid over the image, highlighting cells marked as 1.

    Parameters:
    - image: PIL.Image object representing the image.
    - grid: 24x24 numpy array with cells marked as 1 or 0.
    - output_filename: String, the filename to save the output image.
    """
    # Create a copy of the image to draw on
    image_with_grid = image.convert("RGBA")
    draw = ImageDraw.Draw(image_with_grid, "RGBA")

    width, height = image.size
    cell_width = width / 24.0
    cell_height = height / 24.0

    # Draw grid lines and highlight cells
    for i in range(24):
        for j in range(24):
            cell_left = i * cell_width
            cell_top = j * cell_height
            cell_right = (i + 1) * cell_width
            cell_bottom = (j + 1) * cell_height

            if grid[j][i] == 1:
                # Highlight the cell with a semi-transparent color
                draw.rectangle(
                    [cell_left, cell_top, cell_right, cell_bottom],
                    fill=(255, 0, 0, 100),  # Semi-transparent red
                )

    # Draw the grid lines over the entire image
    for i in range(25):
        # Vertical lines
        x = i * cell_width
        draw.line([(x, 0), (x, height)], fill=(128, 128, 128, 150))
    for j in range(25):
        # Horizontal lines
        y = j * cell_height
        draw.line([(0, y), (width, y)], fill=(128, 128, 128, 150))

    # Save the image
    image_with_grid.save(output_filename)

    print(f"Grid image saved as '{output_filename}'.")


def txt_highlight(tokenizer, txt_prompt, highlighted_idx_range=[[]]):
    # Convert text to tokens
    tokens = tokenizer.tokenize(txt_prompt)

    # Initialize the mask
    mask = [0] * len(tokens)

    # Convert highlighted_idx_range to integer ranges
    ranges = []
    for idx_range in highlighted_idx_range:
        if isinstance(idx_range, str):
            # Add a space before the string to avoid partial matches
            if idx_range[0] != " ":
                idx_range = " " + idx_range
            start_idx = txt_prompt.find(idx_range)
            if start_idx == -1:
                start_idx = txt_prompt.find(
                    idx_range[1:]
                )  # remove the space and try again
                if start_idx == -1:
                    continue  # Skip if the string is not found
            end_idx = start_idx + len(idx_range)
            ranges.append((start_idx, end_idx))
        elif isinstance(idx_range, list) and len(idx_range) == 2:
            ranges.append((idx_range[0], idx_range[1]))

    # Mark the highlighted ranges in the mask
    for start_idx, end_idx in ranges:
        start_token_idx = len(tokenizer.tokenize(txt_prompt[:start_idx]))
        end_token_idx = len(tokenizer.tokenize(txt_prompt[:end_idx]))
        # TODO: Include the start and end tokens that partially overlap with the highlighted area
        for i in range(start_token_idx, end_token_idx):
            mask[i] = 1

    return [0] + mask


# Usage Example:
if __name__ == "__main__":
    from PIL import Image

    # Load your image
    image = Image.open(
        "./storage/pointingqa/Datasets/LookTwiceQA/vg/VG_100K_2/2414753.jpg"
    )  # Replace with your image file

    # Define your bounding boxes (left, top, right, bottom)
    bboxes = [[72, 61, 463, 216]]
    points = [[268, 138]]

    # Get the grid highlighting the bounding boxes
    grid = bbox_highlight(image, bboxes)

    # Render and save the grid image
    render_grid(image, grid, "grid_image.png")

    # Now 'grid_image.png' contains the original image with the grid overlaid,
    # highlighting the cells overlapping with the bounding boxes.
