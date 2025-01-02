from src.utils.file_handling import generate_unique_filename, download_images
from src.video.generator import VideoGenerator
from src.ai.comparison import get_ai_comparison

if __name__ == "__main__":
    try:
        generator = VideoGenerator()
        result = get_ai_comparison(api_key="AIzaSyB0RF_0w_ailsBfbDjClMN5jSdk4xQjRlQ")

        unique_filename = generate_unique_filename(result[0], result[1], result[2], result[3], result[4])
        players = [result[1], result[2]]
        download_images(players, save_dir="/home/sandro/PycharmProjects/Shorts/images")
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