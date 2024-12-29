from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os


class VideoGenerator:
    def __init__(self, width=1080, height=1920):
        self.width = width
        self.height = height
        self.fps = 30

        # Try different system fonts
        possible_fonts = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'
        ]

        self.font_path = next((font for font in possible_fonts if os.path.exists(font)), None)
        if self.font_path is None:
            raise Exception("No suitable font found. Please install dejavu-sans-fonts package.")

        # Match font sizes from HTML
        self.title_font = ImageFont.truetype(self.font_path, 80)  # main-title size
        self.vs_font = ImageFont.truetype(self.font_path, 120)  # vs-section size
        self.player_font = ImageFont.truetype(self.font_path, 60)  # player-name size
        self.stat_font = ImageFont.truetype(self.font_path, 100)  # stat-number size
        self.additional_stat_label_font = ImageFont.truetype(self.font_path, 24)  # stat-label size
        self.additional_stat_value_font = ImageFont.truetype(self.font_path, 36)  # stat-value size

    def create_gradient_background(self):
        # Refined background gradient with smoother transition
        gradient = np.zeros((self.height, self.width, 3), np.uint8)
        for y in range(self.height):
            progress = y / self.height
            r = int(10 * (1 - progress))
            g = int(17 * (1 - progress))
            b = int(40 * (1 - progress))
            gradient[y, :] = [b, g, r]  # OpenCV uses BGR
        return Image.fromarray(gradient)

    def create_stat_box_gradient(self, width, height, is_player_one=True):
        # Refined stat box gradient with smoother transitions
        gradient = np.zeros((height, width, 3), np.uint8)
        for y in range(height):
            progress = y / height
            if is_player_one:
                r = int(0 + (67 * progress))
                g = int(82 + (100 * progress))
                b = int(212 + (43 * progress))
            else:
                r = int(212 + (43 * progress))
                g = int(0 + (67 * progress))
                b = int(0 + (67 * progress))
            gradient[y, :] = [b, g, r]
        return Image.fromarray(gradient)

    def create_frame(self, title, player1, player2, number1, number2, additional_stats=None, winner=None,
                     winner_color='white'):
        image = self.create_gradient_background()
        draw = ImageDraw.Draw(image)

        # Title with text shadow (margin-top: 60px from HTML)
        title_text = title.upper()
        title_bbox = draw.textbbox((0, 0), title_text, font=self.title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_y = 60  # Match HTML margin-top

        # Add text shadow effect
        shadow_color = (255, 255, 255, 77)  # 0.3 opacity white
        for offset in range(1, 21):  # text-shadow: 0 0 20px
            draw.text(((self.width - title_width) // 2, title_y), title_text,
                      font=self.title_font, fill=shadow_color)
        draw.text(((self.width - title_width) // 2, title_y), title_text,
                  font=self.title_font, fill='white')

        # VS text with red glow (matches vs-section style)
        vs_text = "VS"
        vs_bbox = draw.textbbox((0, 0), vs_text, font=self.vs_font)
        vs_width = vs_bbox[2] - vs_bbox[0]
        vs_x = (self.width - vs_width) // 2
        vs_y = (self.height // 2) - 100  # Adjust to match HTML positioning

        # Add red glow effect (text-shadow: 0 0 30px rgba(255,0,0,0.5))
        for offset in range(1, 31):
            alpha = int(128 * (1 - offset / 30))  # Fade out effect
            draw.text((vs_x, vs_y), vs_text, font=self.vs_font,
                      fill=(255, 52, 52, alpha))  # #ff3434 with alpha
        draw.text((vs_x, vs_y), vs_text, font=self.vs_font, fill='#ff3434')

        # Player sections (margin-top: 200px from HTML)
        player_section_y = 200
        for idx, (player, number) in enumerate([(player1, number1), (player2, number2)]):
            x_base = self.width // 4 if idx == 0 else 3 * self.width // 4

            # Player name
            player_text = player.upper()
            player_bbox = draw.textbbox((0, 0), player_text, font=self.player_font)
            player_width = player_bbox[2] - player_bbox[0]
            player_y = player_section_y + 300  # Account for image height + margin
            draw.text((x_base - player_width // 2, player_y), player_text,
                      font=self.player_font, fill='white')

            # Load and resize player image
            player_image_path = f"images/{player.lower()}.jpg"  # Example image path (adjust according to your image files)
            player_image = Image.open(player_image_path)
            player_image = player_image.resize((150, 150))  # Resize image to fit

            # Player image position
            image_x = x_base - 75  # Center the image horizontally
            image_y = player_section_y + 100  # Positioning the image vertically

            # Paste player image onto the frame
            image.paste(player_image, (image_x, image_y))

            # Stat box with gradient background (margin-top: 30px from HTML)
            stat_text = str(number)
            stat_bbox = draw.textbbox((0, 0), stat_text, font=self.stat_font)
            stat_width = stat_bbox[2] - stat_bbox[0]

            # Create gradient stat box
            box_padding = 30  # padding: 30px from HTML
            box_height = 150  # Approximate height from HTML
            box_width = stat_width + (box_padding * 2)
            stat_box = self.create_stat_box_gradient(box_width, box_height, idx == 0)

            box_x = x_base - (box_width // 2)
            box_y = player_y + 100  # Position below player name

            # Add box shadow
            shadow_color = (0, 82, 212, 77) if idx == 0 else (212, 0, 0, 77)
            for offset in range(1, 31):
                shadow_y = box_y + offset
                draw.rectangle([box_x, shadow_y, box_x + box_width, shadow_y + box_height],
                               fill=shadow_color)

            # Paste the gradient box
            image.paste(stat_box, (box_x, box_y))

            # Draw stat number with text shadow
            stat_x = x_base - (stat_width // 2)
            stat_y = box_y + ((box_height - stat_bbox[3]) // 2)

            # Mark the winner with green color
            if winner == 1 and idx == 0 and winner_color == 'green':
                stat_color = 'green'
            elif winner == 2 and idx == 1 and winner_color == 'green':
                stat_color = 'green'
            else:
                stat_color = 'white'

            draw.text((stat_x, stat_y), stat_text, font=self.stat_font, fill=stat_color)

        # Additional stats at bottom (bottom: 100px from HTML)
        if additional_stats:
            stat_y = self.height - 200
            num_stats = len(additional_stats)
            stat_spacing = self.width // (num_stats + 1)

            for idx, (label, value) in enumerate(additional_stats):
                x_pos = stat_spacing * (idx + 1)

                # Create semi-transparent background (background: rgba(255,255,255,0.1))
                stat_box_padding = 20
                label_bbox = draw.textbbox((0, 0), label, font=self.additional_stat_label_font)
                value_bbox = draw.textbbox((0, 0), str(value), font=self.additional_stat_value_font)
                box_width = max(label_bbox[2] - label_bbox[0], value_bbox[2] - value_bbox[0]) + (stat_box_padding * 2)

                # Draw rounded rectangle background
                draw.rectangle([x_pos - box_width // 2, stat_y - stat_box_padding,
                                x_pos + box_width // 2, stat_y + 80 + stat_box_padding],
                               fill=(255, 255, 255, 26))  # 0.1 opacity white

                # Draw label and value
                draw.text((x_pos - (label_bbox[2] - label_bbox[0]) // 2, stat_y),
                          label, font=self.additional_stat_label_font, fill='#aaaaaa')
                draw.text((x_pos - (value_bbox[2] - value_bbox[0]) // 2, stat_y + 40),
                          str(value), font=self.additional_stat_value_font, fill='white')

        return np.array(image)

    def generate_video(self, title, player1, player2, final_num1, final_num2,
                       output_filename, additional_stats=None):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, self.fps, (self.width, self.height))

        total_frames = self.fps * 10  # 10 seconds video

        for frame_num in range(total_frames):
            second = frame_num / self.fps

            # Animate the numbers
            if second < 5:
                num1, num2 = 0, 0
            elif second < 8:
                progress = (second - 5) / 3
                num1 = int(final_num1 * progress)
                num2 = int(final_num2 * progress)
            else:
                num1, num2 = final_num1, final_num2

            winner = 1 if num1 > num2 else (2 if num2 > num1 else None)
            winner_color = 'green' if winner else 'white'

            # Generate the frame
            frame = self.create_frame(title, player1, player2, num1, num2, additional_stats, winner, winner_color)

            out.write(frame)

        out.release()



# Example usage
if __name__ == "__main__":
    try:
        generator = VideoGenerator()
        generator.generate_video(
            title="Red Cards in Career",
            player1="PELE",
            player2="CR7",
            final_num1=213,
            final_num2=1121,
            output_filename="football_comparison.mp4",
        )
        print("Video generated successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
