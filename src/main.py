import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
#from datasets import load_dataset
import os
from pydub import AudioSegment
from whispersModel import *
from audioPreprocess import *
from pyannote.audio import Pipeline
#from diarizers import SegmentationModel  # Importiere die Klasse zum Laden des feingetunten Modells
import soundfile as sf
import numpy as np
import whisperx
import gc

def get_speech_segments(audio_filepath):
    """
    Extrahiert Sprechsegmente mit Zeitstempeln aus einer Audiodatei unter Verwendung der modifizierten pyannote Pipeline.

    Args:
        audio_filepath (str): Der Pfad zur Audiodatei (sollte von pyannote.audio unterstützt werden oder vorher in ein passendes Format konvertiert werden).

    Returns:
        pyannote.core.Annotation: Ein Annotation-Objekt, das die Sprechsegmente mit Zeitstempeln enthält.
    """
    try:
        # Die Pipeline erwartet entweder einen Pfad zur Audiodatei oder ein vorverarbeitetes Dictionary
        diarization = pipeline(audio_filepath)
        return diarization
    except Exception as e:
        print(f"Fehler bei der Verarbeitung von {audio_filepath}: {e}")
        return None

    # Beispielhafte Anwendung auf eine deiner Audiodateien:
    audio_file = 'audio.wav'  # Ersetze durch den Pfad zu deiner WAV-Datei
    diarization_result = get_speech_segments(audio_file)

    if diarization_result:
    # Gib die Sprechsegmente mit Zeitstempeln aus
        for segment, _, label in diarization_result.itertracks(yield_label=True):

            print(f"Segment: {segment.start:.3f} - {segment.end:.3f}, Label: {label}")

            # Optional: Speichere die Ergebnisse im RTTM-Format
            with open("sprechsegmente.rttm", "w") as rttm:
                diarization_result.write_rttm(rttm)

def main():

    try:
    # Wechseln in das vorherige Verzeichnis
        os.chdir("..")
        print(f"Aktuelles Verzeichnis nach Wechsel nach oben: {os.getcwd()}")

        # Wechseln in das Unterverzeichnis 'audiodata'
        current_dir = os.chdir("audiodata")
        print(f"Aktuelles Verzeichnis nach Wechsel nach 'audiodata': {os.getcwd()}")

    except FileNotFoundError:
        print(f"Fehler: Das Verzeichnis 'audiodata' konnte im übergeordneten Verzeichnis nicht gefunden werden.")
    except OSError as e:
        print(f"Ein Fehler ist beim Wechseln des Verzeichnisses aufgetreten: {e}")

    arr = os.listdir(current_dir)
    print(arr)
    print(arr[0])

    #!!!!!!!! Whsipers can only load 25 mb files, so we have to split the files into shorter files. !!!!!!!!!
    #Possible problems: what if the sentences are cut, but maybe its no probelm, well see

    episode = AudioSegment.from_mp3(arr[1])

    end_time_ms = 106000

    # Slice das Audio bis zu dieser Zeit
    shortened_episode = episode[:end_time_ms]
    print(os.getcwd())
    # Optional: Speichere die gekürzte Version
    shortened_episode.export(os.getcwd() + "/" + "shortened_audio.mp3", format="mp3")

    try:

        transcription = pipe(arr[3], return_timestamps=True)
        print(transcription["text"])

    except Exception as e:

        print(f"Fehler bei der Transkription: {e}")
        print("Stelle sicher, dass die Audiodatei existiert und ein unterstütztes Format hat.")
        print("Möglicherweise musst du zusätzliche Abhängigkeiten wie 'ffmpeg' installieren.")

    mp3_datei = 'shortened_audio.mp3'  # Ersetze durch den Pfad zu deiner MP3-Datei
    wav_datei = 'audio.wav'  # Der Pfad, wo die WAV-Datei gespeichert werden soll
    convert_mp3_to_wav_pydub(mp3_datei, wav_datei)

    device = "cuda"
    audio_file = "shortened_audio.mp3"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["de"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"]) # segments are now assigned speaker IDs


if __name__ == "__main__":

    main()