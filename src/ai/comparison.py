import json
import logging
import os

import google.generativeai as genai

from src.utils.file_handling import generate_unique_filename

used_comparisons = set()
def is_comparison_unique(title, player1, player2, stat1, stat2):
    players = sorted([(player1, stat1), (player2, stat2)], key=lambda x: x[0].lower())
    comparison_key = (title, players[0][0], players[1][0], players[0][1], players[1][1])
    if comparison_key in used_comparisons:
        return False
    used_comparisons.add(comparison_key)
    return True

def get_ai_comparison(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        unique_comparison = False
        while not unique_comparison:
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
                "player1": "Top 100 footballer only surnames",
                "player2": "different footballer from the same list",
                "stat1": corresponding statistic for player1,
                "stat2": corresponding statistic for player2
            }}

            Requirements:
            - Generate random but accurate statistics
            - player1 and player2 should have a 50/50 chance of being the winner, meaning they should win equally as often.
            - player1 should not always be the winner; player2 should win in about half of the comparisons.
            - Use different players each time.
            - Return only the JSON object.
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

            title, player1, player2, stat1, stat2 = data['title'], data['player1'], data['player2'], int(
                data['stat1']), int(data['stat2'])

            if is_comparison_unique(title, player1, player2, stat1, stat2):
                uniq_name = generate_unique_filename(title, player1, player2, stat1, stat2)
                if not os.path.exists(uniq_name):
                    unique_comparison = True
                else:
                    print("File already exists.... Regenerating")
            else:
                print("Duplicate content.... Regenerating")

        return title, player1, player2, stat1, stat2

    except Exception as e:
        logging.error(f"Error getting AI response: {str(e)}")
        return None