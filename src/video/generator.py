
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip
from moviepy.video.fx.mask_color import mask_color
from src.config.settings import VIDEO_SETTINGS


class VideoGenerator:
    def __init__(self, width=VIDEO_SETTINGS.get('width'), height=VIDEO_SETTINGS.get('height')):
        self.width = width
        self.height = height
        self.fps = VIDEO_SETTINGS.get('fps')
        self.audio_segments = []

        possible_fonts = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'
        ]

        self.font_path = next((font for font in possible_fonts if os.path.exists(font)), None)
        if self.font_path is None:
            raise Exception("No suitable font found. Please install dejavu-sans-fonts package.")

        self.title_font = ImageFont.truetype(self.font_path, 80)
        self.vs_font = ImageFont.truetype(self.font_path, 120)
        self.player_font = ImageFont.truetype(self.font_path, 60)
        self.stat_font = ImageFont.truetype(self.font_path, 100)
        self.additional_stat_label_font = ImageFont.truetype(self.font_path, 24)
        self.additional_stat_value_font = ImageFont.truetype(self.font_path, 36)

    def create_gradient_background(self):
        # Create a green gradient
        gradient = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            progress = y / self.height
            r = int(34 * (1 - progress))
            g = int(139 * (1 - progress)) + int(71 * progress)
            b = int(34 * (1 - progress))
            gradient[y, :] = [r, g, b]

        gradient_image = Image.fromarray(gradient)

        # Draw football field markings
        draw = ImageDraw.Draw(gradient_image)

        field_margin = 50
        line_width = 10

        # Field boundary
        draw.rectangle(
            [(field_margin, field_margin),
             (self.width - field_margin, self.height - field_margin)],
            outline="white", width=line_width
        )

        # Center line
        center_x = self.width // 2
        draw.line(
            [(center_x, field_margin), (center_x, self.height - field_margin)],
            fill="white", width=line_width
        )

        # Center circle
        center_y = self.height // 2
        circle_radius = 100
        draw.ellipse(
            [(center_x - circle_radius, center_y - circle_radius),
             (center_x + circle_radius, center_y + circle_radius)],
            outline="white", width=line_width
        )

        # Penalty boxes
        penalty_box_width = 200
        penalty_box_height = 400
        draw.rectangle(
            [(field_margin, center_y - penalty_box_height // 2),
             (field_margin + penalty_box_width, center_y + penalty_box_height // 2)],
            outline="white", width=line_width
        )
        draw.rectangle(
            [(self.width - field_margin - penalty_box_width, center_y - penalty_box_height // 2),
             (self.width - field_margin, center_y + penalty_box_height // 2)],
            outline="white", width=line_width
        )

        return gradient_image

    def create_frame(self, title, player1, player2, number1, number2, additional_stats=None, winner=None,
                     winner_color='white'):
        image = self.create_gradient_background()
        image = image.convert('RGBA')
        draw = ImageDraw.Draw(image)
        title_text = title.upper()
        title_bbox = draw.textbbox((0, 0), title_text, font=self.title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_y = 60

        # Add a shadow effect behind the title text
        shadow_color = (255, 255, 255, 77)
        for offset in range(1, 21):
            draw.text(((self.width - title_width) // 2, title_y), title_text,
                      font=self.title_font, fill=shadow_color)
        draw.text(((self.width - title_width) // 2, title_y), title_text,
                  font=self.title_font, fill='white')

        # Draw the "VS" text in the middle
        vs_text = "VS"
        vs_bbox = draw.textbbox((0, 0), vs_text, font=self.vs_font)
        vs_width = vs_bbox[2] - vs_bbox[0]
        vs_x = (self.width - vs_width) // 2
        vs_y = (self.height // 2) - 100

        # Add a gradient shadow effect for the "VS" text
        for offset in range(1, 31):
            alpha = int(128 * (1 - offset / 30))
            draw.text((vs_x, vs_y), vs_text, font=self.vs_font,
                      fill=(255, 52, 52, alpha))
        draw.text((vs_x, vs_y), vs_text, font=self.vs_font, fill='#ff3434')

        # Position player names, images, and stats
        player_section_y = 200
        for idx, (player, number) in enumerate([(player1, number1), (player2, number2)]):
            x_base = self.width // 4 if idx == 0 else 3 * self.width // 4

            # Draw player names below their respective images
            player_text = player.upper()
            player_bbox = draw.textbbox((0, 0), player_text, font=self.player_font)
            player_width = player_bbox[2] - player_bbox[0]
            player_y = player_section_y + 300
            draw.text((x_base - player_width // 2, player_y), player_text,
                      font=self.player_font, fill='white')

            # Load and process player images with color correction
            player_image_path = f"images/{player.lower()}.jpg"
            player_image = Image.open(player_image_path).convert('RGB')

            # Color correction
            # Convert to numpy array for easier processing
            img_array = np.array(player_image)

            # Calculate the average color balance
            avg_color = np.mean(img_array, axis=(0, 1))

            # If blue channel is dominant, adjust color balance
            if avg_color[2] > avg_color[0] * 1.2 and avg_color[2] > avg_color[1] * 1.2:
                # Reduce blue channel and enhance red/green channels
                correction_factor = avg_color[2] / ((avg_color[0] + avg_color[1]) / 2)
                img_array[:, :, 2] = np.clip(img_array[:, :, 2] / correction_factor, 0, 255)
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.2, 0, 255)  # Enhance red
                img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.2, 0, 255)  # Enhance green

            # Convert back to PIL Image
            player_image = Image.fromarray(img_array.astype('uint8'))

            # Convert to RGBA and resize
            player_image = player_image.convert('RGBA')
            player_image = player_image.resize((150, 150), Image.Resampling.LANCZOS)

            # Create a mask for the player image
            if player_image.mode == 'RGBA':
                mask = player_image.split()[3]
            else:
                mask = None

            # Position and paste the image
            image_x = x_base - 75
            image_y = player_section_y + 100
            image.paste(player_image, (image_x, image_y), mask)

            # Display stats in boxes with gradient effects
            stat_text = str(number)
            stat_bbox = draw.textbbox((0, 0), stat_text, font=self.stat_font)
            stat_width = stat_bbox[2] - stat_bbox[0]

            box_padding = 30
            box_height = 150
            box_width = stat_width + (box_padding * 2)

            box_x = x_base - (box_width // 2)
            box_y = player_y + 100

            # Add shadow for the stat box
            shadow_color = (0, 82, 212, 77) if idx == 0 else (212, 0, 0, 77)
            for offset in range(1, 31):
                shadow_y = box_y + offset
                draw.rectangle([box_x, shadow_y, box_x + box_width, shadow_y + box_height],
                               fill=shadow_color)

            # Determine color for winner's stat
            stat_color = 'green' if winner == idx + 1 and winner_color == 'green' else 'white'
            draw.text((box_x + box_padding, box_y + box_padding), stat_text,
                      font=self.stat_font, fill=stat_color)

        # Display additional stats (if any) at the bottom
        if additional_stats:
            stat_y = self.height - 200
            num_stats = len(additional_stats)
            stat_spacing = self.width // (num_stats + 1)

            for idx, (label, value) in enumerate(additional_stats):
                x_pos = stat_spacing * (idx + 1)

                # Draw background rectangle for additional stats
                stat_box_padding = 20
                label_bbox = draw.textbbox((0, 0), label, font=self.additional_stat_label_font)
                value_bbox = draw.textbbox((0, 0), str(value), font=self.additional_stat_value_font)
                box_width = max(label_bbox[2] - label_bbox[0], value_bbox[2] - value_bbox[0]) + (stat_box_padding * 2)

                draw.rectangle([x_pos - box_width // 2, stat_y - stat_box_padding,
                                x_pos + box_width // 2, stat_y + 80 + stat_box_padding],
                               fill=(255, 255, 255, 26))

                draw.text((x_pos - (label_bbox[2] - label_bbox[0]) // 2, stat_y),
                          label, font=self.additional_stat_label_font, fill='#aaaaaa')
                draw.text((x_pos - (value_bbox[2] - value_bbox[0]) // 2, stat_y + 40),
                          str(value), font=self.additional_stat_value_font, fill='white')

        # Convert back to RGB before returning numpy array
        image = image.convert('RGB')
        return np.array(image)


    def create_title_audio(self, title):
        tts = gTTS(text=title, lang='en')
        filename = 'title.mp3'
        tts.save(filename)
        self.audio_segments.append((filename, 0))

    def generate_video(self, title, player1, player2, final_num1, final_num2,
                       output_filename, additional_stats=None):
        temp_video = "temp_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, self.fps, (1080, 1920))

        total_duration = VIDEO_SETTINGS.get('total_duration')
        marking_start = VIDEO_SETTINGS.get('marking_start')

        # Determine winner based on final stats
        winner = 1 if final_num1 > final_num2 else (2 if final_num2 > final_num1 else None)

        # Generate video frames
        for frame_num in range(self.fps * total_duration):
            second = frame_num / self.fps
            if second < 5:
                num1, num2 = 0, 0
            elif second < 8:
                progress = (second - 5) / 3
                num1 = int(final_num1 * progress)
                num2 = int(final_num2 * progress)
            else:
                num1, num2 = final_num1, final_num2

            # Highlight winner in the last 2 seconds
            winner_color = 'green' if second >= marking_start and winner else 'white'

            frame = self.create_frame(title, player1, player2, num1, num2,
                                      additional_stats, winner, winner_color)
            out.write(frame)

        out.release()

        # Load base and overlay videos
        base_video = VideoFileClip(temp_video)
        insert_video = VideoFileClip("output_video_with_audio.mp4").subclip(0, 4.5)
        marking_video = VideoFileClip("marked_without_green.mp4")
        sui = VideoFileClip("celebration_without_green.mp4")

        # Check the duration of the celebration clip
        print(f"Duration of celebration clip: {sui.duration} seconds")

        # Remove black background from the overlay videos
        insert_video = mask_color(insert_video, color=(0, 0, 0), thr=50, s=10)
        marking_video = mask_color(marking_video, color=(0, 0, 0), thr=50, s=10)
        sui = mask_color(sui, color=(0, 0, 0), thr=50, s=10)

        # Zoom-out effect for sui (celebration) video
        zoom_factor = 0.8  # Set the zoom-out factor (80% of the original size)
        sui = sui.resize(zoom_factor)

        # Position the overlay videos
        x_center = (1080 - insert_video.w) // 2
        y_center = (1920 - insert_video.h) // 1.75
        insert_video = insert_video.set_position((x_center, y_center)).set_start(2)

        marking_y = 850  # Adjusted position
        if winner == 1:
            marking_x = 1080 // 4 - marking_video.w // 2
        else:
            marking_x = (3 * 1080 // 4) - marking_video.w // 2

        marking_video = marking_video.set_position((marking_x, marking_y)).set_start(marking_start)

        # Ensure the celebration clip doesn't exceed its duration
        start_time = 9
        end_time = min(sui.duration, total_duration - start_time)  # Adjust end time based on clip duration

        # Position the celebration video and set its duration
        sui_x_center = (1080 - sui.w) // 2
        sui_y_center = (1920 - sui.h) // 2
        sui = sui.set_position((sui_x_center, sui_y_center)).set_start(start_time).set_end(start_time + end_time)

        # Combine all clips
        video_clips = [base_video, insert_video]
        if winner:
            video_clips.append(marking_video)
        video_clips.append(sui)  # Add celebration video to the sequence

        final_video = CompositeVideoClip(video_clips)

        # Handle audio
        title_text = f'Who has most {title}'
        self.create_title_audio(title_text)
        title_audio = [AudioFileClip(audio_file) for audio_file, _ in self.audio_segments]

        video_audios = [insert_video.audio.set_start(2)]
        if winner:
            video_audios.append(marking_video.audio.set_start(marking_start))

        if sui.audio:
            video_audios.append(sui.audio.set_start(start_time))  # Adjust start time as needed

        all_audio = title_audio + video_audios
        final_video.audio = CompositeAudioClip(all_audio)
        final_video.write_videofile(output_filename, codec='libx264', audio_codec='aac')
        os.remove(temp_video)
        for audio_file, _ in self.audio_segments:
            os.remove(audio_file)
        self.audio_segments = []
        base_video.close()
        insert_video.close()
        marking_video.close()
        sui.close()
