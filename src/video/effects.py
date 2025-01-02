
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_stat_box_gradient(width: int, height: int, is_player_one: bool = True, background_image: Image = None) -> Image:
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    base_rgb, range_rgb = (
        ([0, 82, 212], [67, 100, 43]) if is_player_one else ([212, 0, 0], [43, 67, 67])
    )
    for y in range(height):
        progress = y / height  # Calculate progress as a ratio (0 to 1)
        gradient[y, :] = [
            int(base_rgb[2] + range_rgb[2] * progress),  # Blue channel
            int(base_rgb[1] + range_rgb[1] * progress),  # Green channel
            int(base_rgb[0] + range_rgb[0] * progress)   # Red channel
        ]
    gradient_image = Image.fromarray(gradient)
    if not background_image:
        background_image = Image.new("RGB", (width, height), "green")
        draw = ImageDraw.Draw(background_image)
        field_margin = 20
        line_width = 5
        draw.rectangle(
            [
                (field_margin, field_margin),
                (width - field_margin, height - field_margin)
            ],
            outline="white",
            width=line_width
        )

        # Draw penalty boxes
        penalty_box_margin = 100
        penalty_box_height = height // 4
        draw.rectangle(
            [
                (field_margin, height // 2 - penalty_box_height),
                (field_margin + penalty_box_margin, height // 2 + penalty_box_height)
            ],
            outline="white",
            width=line_width
        )
        draw.rectangle(
            [
                (width - field_margin - penalty_box_margin, height // 2 - penalty_box_height),
                (width - field_margin, height // 2 + penalty_box_height)
            ],
            outline="white",
            width=line_width
        )

        # Draw the center circle
        center_x, center_y = width // 2, height // 2
        circle_radius = min(width, height) // 6
        draw.ellipse(
            [
                (center_x - circle_radius, center_y - circle_radius),
                (center_x + circle_radius, center_y + circle_radius)
            ],
            outline="white",
            width=line_width
        )

        # Draw the center line
        draw.line(
            [(center_x, field_margin), (center_x, height - field_margin)],
            fill="white",
            width=line_width
        )

        # Add smaller field markings (e.g., goal areas, penalty spots)
        goal_area_margin = 50
        goal_area_height = height // 8
        draw.rectangle(
            [
                (field_margin, height // 2 - goal_area_height),
                (field_margin + goal_area_margin, height // 2 + goal_area_height)
            ],
            outline="white",
            width=line_width
        )
        draw.rectangle(
            [
                (width - field_margin - goal_area_margin, height // 2 - goal_area_height),
                (width - field_margin, height // 2 + goal_area_height)
            ],
            outline="white",
            width=line_width
        )

        # Draw penalty spots
        penalty_spot_offset = penalty_box_margin - 20
        draw.ellipse(
            [
                (field_margin + penalty_spot_offset - 3, height // 2 - 3),
                (field_margin + penalty_spot_offset + 3, height // 2 + 3)
            ],
            fill="white"
        )
        draw.ellipse(
            [
                (width - field_margin - penalty_spot_offset - 3, height // 2 - 3),
                (width - field_margin - penalty_spot_offset + 3, height // 2 + 3)
            ],
            fill="white"
        )
    background_image = background_image.resize((width, height))
    gradient_image = Image.blend(background_image.convert("RGB"), gradient_image, alpha=0.5)

    return gradient_image
