import json
import logging
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import google.generativeai as genai
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip



used_titles = set()


def get_ai_comparison(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        unique_title = False
        while not unique_title:
            prompt = f"""
            Generate a unique football statistics comparison. Choose from these categories:
            - Career Goals
            - Champions League Goals
            - International Goals
            - Goals in a Single Season
            - League Titles Won
            - Ballon d'Or Awards
            - Hat-tricks Scored
            - World Cup Goals
            - Assists in Career
            - Free-kick Goals

            Return in this exact JSON format:
            {{
                "title": "your chosen category",
                "player1": "footballer from [MESSI, CR7, PELE, MARADONA, NEYMAR]",
                "player2": "different footballer from the same list",
                "stat1": corresponding statistic for player1,
                "stat2": corresponding statistic for player2
            }}

            Requirements:
            - Generate random but accurate statistics
            - player1 must not be always winner. player2 should win sometimes 50/50%
            - Use different players each time
            - Return only the JSON object
            """

            response = model.generate_content(prompt, safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ])

            response_text = response.text.strip()
            json_str = response_text[response_text.find('{'):response_text.rfind('}') + 1]
            data = json.loads(json_str)
            uniq_name = generate_unique_filename(data['title'], data['player1'], data['player2'], data['stat1'], data['stat2'])
            if os.path.exists(uniq_name):
                print("File already exists.... Regenerating")
                get_ai_comparison(api_key)
            print(data)

            title = data['title']
            if title not in used_titles:
                used_titles.add(title)
                unique_title = True

        return (
            title,
            *sorted([data['player1'], data['player2']]),
            int(data['stat1']),
            int(data['stat2'])
        )

    except Exception as e:
        logging.error(f"Error getting AI response: {str(e)}")
        return None


def create_stat_box_gradient(width, height, is_player_one=True):
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


class VideoGenerator:
    def __init__(self, width=1080, height=1920):
        self.width = width
        self.height = height
        self.fps = 30
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
        gradient = np.zeros((self.height, self.width, 3), np.uint8)
        for y in range(self.height):
            progress = y / self.height
            r = int(10 * (1 - progress))
            g = int(17 * (1 - progress))
            b = int(40 * (1 - progress))
            gradient[y, :] = [b, g, r]
        return Image.fromarray(gradient)

    def create_frame(self, title, player1, player2, number1, number2, additional_stats=None, winner=None,
                     winner_color='white'):
        image = self.create_gradient_background()
        draw = ImageDraw.Draw(image)

        title_text = title.upper()
        title_bbox = draw.textbbox((0, 0), title_text, font=self.title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_y = 60

        shadow_color = (255, 255, 255, 77)
        for offset in range(1, 21):
            draw.text(((self.width - title_width) // 2, title_y), title_text,
                      font=self.title_font, fill=shadow_color)
        draw.text(((self.width - title_width) // 2, title_y), title_text,
                  font=self.title_font, fill='white')

        vs_text = "VS"
        vs_bbox = draw.textbbox((0, 0), vs_text, font=self.vs_font)
        vs_width = vs_bbox[2] - vs_bbox[0]
        vs_x = (self.width - vs_width) // 2
        vs_y = (self.height // 2) - 100

        for offset in range(1, 31):
            alpha = int(128 * (1 - offset / 30))
            draw.text((vs_x, vs_y), vs_text, font=self.vs_font,
                      fill=(255, 52, 52, alpha))
        draw.text((vs_x, vs_y), vs_text, font=self.vs_font, fill='#ff3434')

        player_section_y = 200
        for idx, (player, number) in enumerate([(player1, number1), (player2, number2)]):
            x_base = self.width // 4 if idx == 0 else 3 * self.width // 4

            player_text = player.upper()
            player_bbox = draw.textbbox((0, 0), player_text, font=self.player_font)
            player_width = player_bbox[2] - player_bbox[0]
            player_y = player_section_y + 300
            draw.text((x_base - player_width // 2, player_y), player_text,
                      font=self.player_font, fill='white')

            player_image_path = f"images/{player.lower()}.jpg"
            player_image = Image.open(player_image_path)
            player_image = player_image.resize((150, 150))

            image_x = x_base - 75
            image_y = player_section_y + 100

            image.paste(player_image, (image_x, image_y))

            stat_text = str(number)
            stat_bbox = draw.textbbox((0, 0), stat_text, font=self.stat_font)
            stat_width = stat_bbox[2] - stat_bbox[0]

            box_padding = 30
            box_height = 150
            box_width = stat_width + (box_padding * 2)
            stat_box = create_stat_box_gradient(box_width, box_height, idx == 0)

            box_x = x_base - (box_width // 2)
            box_y = player_y + 100

            shadow_color = (0, 82, 212, 77) if idx == 0 else (212, 0, 0, 77)
            for offset in range(1, 31):
                shadow_y = box_y + offset
                draw.rectangle([box_x, shadow_y, box_x + box_width, shadow_y + box_height],
                               fill=shadow_color)

            image.paste(stat_box, (box_x, box_y))

            stat_x = x_base - (stat_width // 2)
            stat_y = box_y + ((box_height - stat_bbox[3]) // 2)

            if winner == 1 and idx == 0 and winner_color == 'green':
                stat_color = 'green'
            elif winner == 2 and idx == 1 and winner_color == 'green':
                stat_color = 'green'
            else:
                stat_color = 'white'

            draw.text((stat_x, stat_y), stat_text, font=self.stat_font, fill=stat_color)

        if additional_stats:
            stat_y = self.height - 200
            num_stats = len(additional_stats)
            stat_spacing = self.width // (num_stats + 1)

            for idx, (label, value) in enumerate(additional_stats):
                x_pos = stat_spacing * (idx + 1)

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

        total_duration = 10  # 10 seconds total
        marking_start = 8  # Start marking effect at 8 seconds

        # Determine winner once
        winner = 1 if final_num1 > final_num2 else (2 if final_num2 > final_num1 else None)

        # Generate base frames
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

            # Only change color to green in the last 2 seconds for the winner
            winner_color = 'green' if second >= marking_start and winner else 'white'

            frame = self.create_frame(title, player1, player2, num1, num2,
                                      additional_stats, winner, winner_color)
            out.write(frame)

        out.release()

        # Load videos
        base_video = VideoFileClip(temp_video)
        insert_video = VideoFileClip("output_video_with_audio.mp4").subclip(0, 4.5)
        marking_video = VideoFileClip("marked_without_green.mp4")

        # Position the insert video (middle animation)
        x_center = (1080 - insert_video.w) // 2
        y_center = (1920 - insert_video.h) // 1.75

        insert_video = (insert_video
                        .set_position((x_center, y_center))
                        .set_start(2))

        # Position the marking video based on winner
        # Assuming the marking video should be positioned around the stat box area
        marking_y = 850  # Increased from 700 to 850 to move down

        if winner == 1:
            # Position for player 1 (left side)
            marking_x = 1080 // 4 - marking_video.w // 2
        else:
            # Position for player 2 (right side)
            marking_x = (3 * 1080 // 4) - marking_video.w // 2

        marking_video = (marking_video
                         .set_position((marking_x, marking_y))
                         .set_start(marking_start))  # Start at 8 seconds

        # Only add marking video if there's a winner
        video_clips = [base_video, insert_video]
        if winner:
            video_clips.append(marking_video)

        final_video = CompositeVideoClip(video_clips)

        # Handle audio
        title_text = f'Who has most {title}'
        self.create_title_audio(title_text)
        title_audio = [AudioFileClip(audio_file) for audio_file, _ in self.audio_segments]

        # Combine audio from both insert_video and marking_video
        video_audios = [insert_video.audio.set_start(2)]
        if winner:
            video_audios.append(marking_video.audio.set_start(marking_start))

        all_audio = title_audio + video_audios
        final_video.audio = CompositeAudioClip(all_audio)

        final_video.write_videofile(output_filename, codec='libx264', audio_codec='aac')

        # Cleanup
        os.remove(temp_video)
        for audio_file, _ in self.audio_segments:
            os.remove(audio_file)
        self.audio_segments = []
        base_video.close()
        insert_video.close()
        marking_video.close()


def generate_unique_filename(title, player1, player2, stat1, stat2):
    sanitized_title = title.replace(" ", "_").replace("-", "_")
    sanitized_player1 = player1.replace(" ", "_").replace("-", "_")
    sanitized_player2 = player2.replace(" ", "_").replace("-", "_")

    filename = f"{sanitized_title}_{sanitized_player1}_vs_{sanitized_player2}_{stat1}_{stat2}.mp4"
    return filename


if __name__ == "__main__":
    try:
        generator = VideoGenerator()
        result = get_ai_comparison(api_key="AIzaSyB0RF_0w_ailsBfbDjClMN5jSdk4xQjRlQ")

        # Generate unique filename based on the stats and players
        unique_filename = generate_unique_filename(result[0], result[1], result[2], result[3], result[4])

        generator.generate_video(
            title=result[0],
            player1=result[1],
            player2=result[2],
            final_num1=result[3],
            final_num2=result[4],
            output_filename=unique_filename,
        )

        print("Video generated successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error: {str(e)}")