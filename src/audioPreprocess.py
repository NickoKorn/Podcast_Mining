from pydub import AudioSegment

# methods for splitting mp3

#methods for transforming mp3 to wav

def convert_mp3_to_wav_pydub(mp3_filepath, wav_filepath, target_sr=16000):
    """
    Konvertiert eine MP3-Datei in eine Mono-WAV-Datei mit der angegebenen Abtastrate unter Verwendung von pydub.

    Args:
        mp3_filepath (str): Der Pfad zur MP3-Datei.
        wav_filepath (str): Der Pfad, unter dem die WAV-Datei gespeichert werden soll.
        target_sr (int): Die gewünschte Abtastrate in Hz (z.B. 16000).
    """
    try:
        audio = AudioSegment.from_mp3(mp3_filepath)
        audio = audio.set_channels(1)  # Mono machen
        audio = audio.set_frame_rate(target_sr)
        audio.export(wav_filepath, format="wav")
        print(f"MP3 erfolgreich konvertiert nach: {wav_filepath}")

    except Exception as e:
        print(f"Fehler bei der Konvertierung: {e}")
