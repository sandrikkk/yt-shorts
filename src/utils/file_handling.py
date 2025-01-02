import unicodedata
import requests
import os
from bs4 import BeautifulSoup
import unicodedata

def slugify(text):
    """Converts a string to a URL-friendly slug (ASCII only)."""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    return text

def generate_unique_filename(title, player1, player2, stat1, stat2):
    players = sorted([(player1, stat1), (player2, stat2)], key=lambda x: x[0].lower())
    sanitized_title = title.replace(" ", "_").replace("-", "_")
    sanitized_player1 = players[0][0].replace(" ", "_").replace("-", "_")
    sanitized_player2 = players[1][0].replace(" ", "_").replace("-", "_")
    stat1, stat2 = players[0][1], players[1][1]
    filename = f"{sanitized_title}_{sanitized_player1}_vs_{sanitized_player2}_{stat1}_{stat2}.mp4"
    return filename

def download_images(players, save_dir="images"):
    """Downloads images for multiple football players and names them after their surname."""

    for player_name in players:
        surname = player_name.split()[-1]
        surname = slugify(surname)  # Slugify the surname

        search_query = f"{player_name} football player"
        search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"

        try:
            response = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            img_tags = soup.find_all("img")
            for img in img_tags:
                src = img.get('src') or img.get('data-src')
                if src and not src.startswith('data:'):
                    try:
                        img_data = requests.get(src, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
                        img_data.raise_for_status()

                        file_extension = os.path.splitext(src)[1] or ".jpg"
                        filename = os.path.join(save_dir, f"{surname}{file_extension}")

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        with open(filename, "wb") as f:
                            for chunk in img_data.iter_content(1024):
                                f.write(chunk)
                        print(f"Downloaded: {filename}")
                        break  # Stop after downloading the first suitable image for this player

                    except requests.exceptions.RequestException as e:
                        print(f"Error downloading image {src}: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
            else:
                print(f"No suitable image found for {player_name}.")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching search results: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
