from pydub import AudioSegment
import os
#from dataScraping import * 

"""
Function Collection for Audio-Preprocessing 
"""
shortened_audios_for_transcriptions = list()
shortened_audios_for_diarization = list()

def split_audio(file_path, max_duration):

    """Teilt eine Audiodatei in Segmente mit max. Dauer für Whisper."""
    print(file_path)
    audio = AudioSegment.from_mp3(file_path)
    segment_length_ms = max_duration * 1000  # Sekunden in Millisekunden
    #os.chdir("shortened_audios")
    # Erstelle Ausgabeordner falls nicht vorhanden
    os.makedirs("shortened_audios", exist_ok=True)
    os.makedirs("shortened_audios_wav", exist_ok=True)
    episode_name = file_path.split(".", 1)[0]
    # Schneide die Datei in Segmente
    for i in range(0, len(audio), int(segment_length_ms)):
        segment = audio[i:i+int(segment_length_ms)]
        os.chdir("shortened_audios")
        if os.path.isfile(f"{episode_name}_segment_{i}.mp3") and os.path.isfile(f"../shortened_audios_wav/{episode_name}_segment_{i}.wav"):
            
            output_filename = f"{episode_name}_segment_{i}.mp3"
            wav_output_filename = f"{episode_name}_segment_{i}.wav"
            shortened_audios_for_transcriptions.append(output_filename)
            shortened_audios_for_diarization.append(wav_output_filename)
            print(f"appended:{output_filename}")
            print("Split existiert bereits")
            os.chdir("..")
        else:
            output_filename = f"{episode_name}_segment_{i}.mp3"
            wav_output_filename = f"{episode_name}_segment_{i}.wav"

            shortened_audios_for_transcriptions.append(output_filename)
            print(f"appended:{output_filename}")
            segment.export(os.getcwd() + "/" + output_filename, format="mp3")
            os.chdir("..")
            os.chdir("shortened_audios_wav")
            shortened_audios_for_diarization.append(wav_output_filename)
            segment.export(os.getcwd() + "/" + wav_output_filename, format="wav")
            os.chdir("..")

    print(f" Datei '{file_path}' wurde erfolgreich gesplittet!")

"""
Preprocess Pipeline
Example of Usage: 
Episodes from StreamLineGUI are selected with chosen tasks combinded like: 
-Give me only Audios from Episodes X
-Give me only Text from Episodes X
-Give me Speaker Diarization of Episodes X

If some of these are already in Database than no preprocess again
"""

def audioPreprocessPipeline(episodes: dict)->tuple:

    shortened_audios_for_transcriptions.clear()
    shortened_audios_for_diarization.clear()
    print(episodes)
    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps")
    print(os.listdir())
    current_dir = None
    try:

    # Wechseln in das vorherige Verzeichnis
        os.chdir("..")
        print(f"Aktuelles Verzeichnis nach Wechsel nach oben: {os.getcwd()}")

        # Wechseln in das Unterverzeichnis 'audiodata'
        current_dir = os.chdir("audioData")
        print(f"Aktuelles Verzeichnis nach Wechsel nach 'audiodata': {os.getcwd()}")

    except FileNotFoundError:

        print(f"Fehler: Das Verzeichnis 'audiodata' konnte im übergeordneten Verzeichnis nicht gefunden werden.")
    except OSError as e:

        print(f"Ein Fehler ist beim Wecshseln des Verzeichnisses aufgetreten: {e}")

    for key in episodes:

        split_audio(episodes[key]['title'].split(":", 1)[0] + ".mp3", max_duration=166)
        #os.chdir("..")
    
    os.chdir("..")
    os.chdir("src")
    print(os.getcwd())
    return shortened_audios_for_transcriptions, shortened_audios_for_diarization

if __name__ == "__main__":

    audioPreprocessPipeline([2,5])
