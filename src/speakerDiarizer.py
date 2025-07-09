#from pyannote.audio import *
#from diarizers import *
import torch 
import torchaudio
#from pydub import AudioSegment
import os
from nemo.collections.asr.models import SortformerEncLabelModel

def convert_mp3_to_wav_pydub(mp3_filepath, wav_filepath):
    """
    Converts an MP3 file to a WAV file using pydub.

    Args:
        mp3_filepath (str): Path to the input MP3 file.
        wav_filepath (str): Path for the output WAV file.
    """
    if not os.path.exists(mp3_filepath):
        print(f"Error: MP3 file not found at {mp3_filepath}")
        return

    try:
        # Load the MP3 file
        audio = AudioSegment.from_mp3(mp3_filepath)

        # Export as WAV
        audio.export(wav_filepath, format="wav")
        print(f"Successfully converted '{mp3_filepath}' to '{wav_filepath}'")
    except Exception as e:
        print(f"Error converting '{mp3_filepath}' to WAV: {e}")
        print("Ensure ffmpeg is installed and accessible in your system's PATH.")

def nemo_cpu(audio_input_path : str):
    
    mono_audio_output_path = audio_input_path

    # 1. Load the stereo audio
    waveform, sample_rate = torchaudio.load(audio_input_path)
    print(f"Original waveform shape: {waveform.shape}, Sample Rate: {sample_rate}")

    # 2. Convert to mono if it's stereo
    if waveform.shape[0] > 1: # Check if number of channels > 1
        # Summing across the channel dimension to get mono
        waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        print(f"Converted mono waveform shape: {waveform_mono.shape}")
        # Save the mono audio to a temporary file
        torchaudio.save(mono_audio_output_path, waveform_mono, sample_rate)
        audio_input_for_nemo = mono_audio_output_path
    else:
        # Already mono
        audio_input_for_nemo = audio_input_path
        print("Audio is already mono.")


    # load model from Hugging Face model card directly (You need a Hugging Face token)
    # Make sure you have your Hugging Face token configured (e.g., via `huggingface-cli login`)
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    
    device = torch.device("cpu")
    diar_model = diar_model.to(device)

    # switch to inference mode
    diar_model.eval()

    #predicted_segments = diar_model.diarize(audio=audio_input_for_nemo, batch_size=1)
    
    predicted_segments, predicted_probs = diar_model.diarize(audio=audio_input_for_nemo, batch_size=1, include_tensor_outputs=True)

    print(predicted_segments)
    print(predicted_probs)

    # Clean up the temporary mono file if created
    if audio_input_for_nemo == mono_audio_output_path and os.path.exists(mono_audio_output_path):
        os.remove(mono_audio_output_path)
    torch.cuda.empty_cache()
    del diar_model
    return predicted_segments

def nemo(audio_input_path : str):
    
    mono_audio_output_path = audio_input_path + "_mono.wav"

    # 1. Load the stereo audio
    waveform, sample_rate = torchaudio.load(audio_input_path)
    #print(f"Original waveform shape: {waveform.shape}, Sample Rate: {sample_rate}")

    # 2. Convert to mono if it's stereo
    if waveform.shape[0] > 1: # Check if number of channels > 1
        # Summing across the channel dimension to get mono
        waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        print(f"Converted mono waveform shape: {waveform_mono.shape}")
        # Save the mono audio to a temporary file
        torchaudio.save(mono_audio_output_path, waveform_mono, sample_rate)
        audio_input_for_nemo = mono_audio_output_path
    else:
        # Already mono
        audio_input_for_nemo = audio_input_path
        print("Audio is already mono.")


    # load model from Hugging Face model card directly (You need a Hugging Face token)
    # Make sure you have your Hugging Face token configured (e.g., via `huggingface-cli login`)
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")

    # switch to inference mode
    diar_model.eval()

    #predicted_segments = diar_model.diarize(audio=audio_input_for_nemo, batch_size=1)
    
    predicted_segments, predicted_probs = diar_model.diarize(audio=audio_input_for_nemo, batch_size=1, include_tensor_outputs=True)

    #print(predicted_segments)
    #print(predicted_probs)

    # Clean up the temporary mono file if created
    if audio_input_for_nemo == mono_audio_output_path and os.path.exists(mono_audio_output_path):
        os.remove(mono_audio_output_path)
    torch.cuda.empty_cache()
    del diar_model
    return predicted_segments

if __name__ == "__main__":

    #convert_mp3_to_wav_pydub("../audioData/shortened_audios/episodeGAG01_segment_0.mp3", "../audioData/shortened_audios/episodeGAG01_segment_0.wav")
    #speakerDiarizerization()
    #speakerDiarizerizatio_German()
    nemo_cpu("../audioData/shortened_audios_wav/GAG09_segment_664000")
