from moviepy.editor import VideoFileClip, AudioFileClip
import cv2
import numpy as np


def remove_green_bg_with_audio(input_video, output_video, bg_color=(0, 0, 0)):
    """
    Removes the green background from a video while retaining its audio.

    Args:
        input_video (str): Path to the input video with green background.
        output_video (str): Path to save the output video with the green background removed.
        bg_color (tuple): RGB tuple for the background color (default is black).
    """
    # Load video
    video = cv2.VideoCapture(input_video)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Temp output video (without audio)
    temp_output_video = "marked_green.mp4"
    out = cv2.VideoWriter(temp_output_video, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Convert to HSV and isolate green
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 55, 55])  # Adjust as needed
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)

        # Replace green background
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        background = np.full(frame.shape, bg_color, dtype=np.uint8)
        background = cv2.bitwise_and(background, background, mask=mask)
        final_frame = cv2.add(frame_bg, background)

        out.write(final_frame)

    # Release resources
    video.release()
    out.release()

    # Combine processed video with original audio
    video_clip = VideoFileClip(temp_output_video)
    original_audio = AudioFileClip(input_video)
    final_clip = video_clip.set_audio(original_audio)
    final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")

    # Clean up temporary file
    video_clip.close()
    original_audio.close()
    import os
    os.remove(temp_output_video)


# Example usage
input_video_path = "green.mp4"
output_video_path = "marked_without_green.mp4"
remove_green_bg_with_audio(input_video_path, output_video_path, bg_color=(0, 0, 0))


import requests
import os
from bs4 import BeautifulSoup
import unicodedata

# def slugify(text):
#     """Converts a string to a URL-friendly slug (ASCII only)."""
#     text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
#     text = text.lower()
#     return text

# def download_images(player_name, save_dir="images"):
#     """Downloads one image of a football player and names it after their surname."""
#
#     surname = player_name.split()[-1]
#     surname = slugify(surname) # Slugify the surname
#
#     search_query = f"{player_name} football player"
#     search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
#
#     try:
#         response = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'})
#         response.raise_for_status()
#
#         soup = BeautifulSoup(response.content, "html.parser")
#
#         img_tags = soup.find_all("img")
#         for img in img_tags:
#             src = img.get('src') or img.get('data-src')
#             if src and not src.startswith('data:'):
#                 try:
#                     img_data = requests.get(src, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
#                     img_data.raise_for_status()
#
#                     file_extension = os.path.splitext(src)[1] or ".jpg"
#                     filename = os.path.join(save_dir, f"{surname}{file_extension}")
#
#                     if not os.path.exists(save_dir):
#                         os.makedirs(save_dir)
#
#                     with open(filename, "wb") as f:
#                         for chunk in img_data.iter_content(1024):
#                             f.write(chunk)
#                     print(f"Downloaded: {filename}")
#                     return
#
#                 except requests.exceptions.RequestException as e:
#                     print(f"Error downloading image {src}: {e}")
#                 except Exception as e:
#                     print(f"An unexpected error occurred: {e}")
#         print(f"No suitable image found for {player_name}.")
#
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching search results: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#
# # Example usage:
# players = ["Lionel Messi", "Cristiano Ronaldo", "Neymar Jr", "Kylian Mbappé", "Erling Haaland", "Zlatan Ibrahimović", 'maradona']
# for player in players:
#     download_images(player, save_dir="/home/sandro/PycharmProjects/Shorts/images")