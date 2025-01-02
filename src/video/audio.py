from gtts import gTTS


def create_title_audio(self, title):
    tts = gTTS(text=title, lang='en')
    filename = 'title.mp3'
    tts.save(filename)
    self.audio_segments.append((filename, 0))