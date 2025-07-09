import chromadb
from chromadb.utils import embedding_functions
import torch
import numpy as np
import os
# Beispiel-Daten (stellen Sie sich vor, diese kommen aus Ihrer Pipeline)
# Jedes Element ist ein dict für ein Segment
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import tempfile
import chromadb
from chromadb.utils import embedding_functions
from scipy.spatial.distance import euclidean # Für die Distanzberechnung
import collections # Für defaultdict in der neuen Funktion
from sklearn.metrics.pairwise import cosine_similarity
import re

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
#speaker_model = speaker_model.to(torch.device('cpu'))

#kann nicht ohne menschliche Überprüfung automatisiert werden
def split_audio_for_speaker_embeddings():

    os.chdir("..")
    os.chdir("audioData")
    os.chdir("shortened_audios_wav")
    #Folge 1 Richard Embeddings
    diarizations = [[['0.240 4.320 speaker_0', '11.840 14.800 speaker_0', '34.800 35.760 speaker_0', '54.720 57.440 speaker_0', '58.080 63.920 speaker_0', '64.800 67.920 speaker_0', '68.080 70.880 speaker_0', '72.960 76.160 speaker_0', '76.880 79.120 speaker_0', '80.160 80.240 speaker_0', '80.400 81.120 speaker_0', '82.240 85.040 speaker_0', '86.000 88.560 speaker_0', '93.520 103.040 speaker_0', '103.600 109.280 speaker_0', '109.440 116.400 speaker_0', '117.040 121.360 speaker_0', '121.760 134.560 speaker_0', '135.920 146.800 speaker_0', '147.520 155.360 speaker_0', '155.920 160.000 speaker_0', '161.040 165.040 speaker_0', '165.120 166.070 speaker_0', '4.560 11.520 speaker_1', '29.200 34.720 speaker_1', '35.920 50.960 speaker_1', '51.840 52.400 speaker_1', '52.720 54.560 speaker_1', '71.360 73.040 speaker_1', '88.640 90.880 speaker_1', '91.200 94.080 speaker_1', '17.920 19.120 speaker_2', '20.400 24.800 speaker_2', '25.760 27.200 speaker_2']], [['0.000 3.520 speaker_0', '4.240 5.520 speaker_0', '6.000 18.960 speaker_0', '19.840 22.800 speaker_0', '23.920 24.080 speaker_0', '26.640 28.720 speaker_0', '29.280 39.040 speaker_0', '39.120 45.360 speaker_0', '46.320 46.560 speaker_0', '46.720 53.760 speaker_0', '54.640 55.760 speaker_0', '65.040 65.200 speaker_0', '75.040 86.080 speaker_0', '86.160 91.360 speaker_0', '93.040 93.120 speaker_0', '93.200 118.240 speaker_0', '118.720 123.600 speaker_0', '126.720 126.880 speaker_0', '127.680 129.520 speaker_0', '129.680 130.400 speaker_0', '131.120 135.200 speaker_0', '136.000 136.880 speaker_0', '144.080 152.480 speaker_0', '152.800 157.600 speaker_0', '158.080 158.240 speaker_0', '158.400 163.280 speaker_0', '163.360 163.520 speaker_0', '163.600 166.070 speaker_0', '23.840 28.320 speaker_1', '56.720 58.240 speaker_1', '58.480 61.200 speaker_1', '61.680 61.840 speaker_1', '62.480 70.640 speaker_1', '70.960 74.880 speaker_1', '123.920 125.360 speaker_1', '125.680 127.520 speaker_1', '136.400 142.880 speaker_1', '143.280 144.080 speaker_1']], [['0.000 0.880 speaker_0', '5.840 6.320 speaker_0', '8.240 11.520 speaker_0', '12.960 23.760 speaker_0', '62.640 69.120 speaker_0', '73.200 79.040 speaker_0', '80.400 83.600 speaker_0', '90.160 93.200 speaker_0', '93.760 99.680 speaker_0', '100.240 115.280 speaker_0', '116.400 124.000 speaker_0', '124.880 137.520 speaker_0', '138.320 140.320 speaker_0', '140.800 148.560 speaker_0', '148.880 150.880 speaker_0', '151.520 154.560 speaker_0', '155.040 166.070 speaker_0', '1.920 7.760 speaker_1', '24.400 30.640 speaker_1', '69.680 72.160 speaker_1', '72.240 73.520 speaker_1', '79.840 80.800 speaker_1', '83.680 85.840 speaker_1', '86.320 90.160 speaker_1', '31.520 35.680 speaker_2', '36.160 38.960 speaker_2', '39.760 41.520 speaker_2', '42.000 47.200 speaker_2', '47.760 53.520 speaker_2', '54.080 58.080 speaker_2', '58.560 59.600 speaker_2', '59.840 61.840 speaker_2']], [['0.000 5.440 speaker_0', '6.400 7.360 speaker_0', '7.840 16.400 speaker_0', '22.560 28.000 speaker_0', '28.480 33.680 speaker_0', '34.640 40.480 speaker_0', '41.120 41.520 speaker_0', '41.600 41.760 speaker_0', '42.160 45.200 speaker_0', '46.000 62.560 speaker_0', '62.800 66.400 speaker_0', '66.480 74.240 speaker_0', '74.800 78.560 speaker_0', '79.600 83.520 speaker_0', '84.000 88.640 speaker_0', '88.800 93.040 speaker_0', '94.400 95.120 speaker_0', '101.760 116.560 speaker_0', '117.200 118.080 speaker_0', '118.720 128.880 speaker_0', '129.200 129.760 speaker_0', '130.080 132.480 speaker_0', '132.960 133.600 speaker_0', '134.080 135.520 speaker_0', '136.400 141.520 speaker_0', '141.680 146.080 speaker_0', '147.280 147.920 speaker_0', '148.240 154.320 speaker_0', '155.360 166.070 speaker_0', '16.560 22.400 speaker_1', '93.440 98.960 speaker_1', '99.040 101.360 speaker_1']], [['0.000 0.080 speaker_0', '0.800 2.560 speaker_0', '5.680 11.600 speaker_0', '12.400 19.200 speaker_0', '19.840 28.960 speaker_0', '29.680 35.920 speaker_0', '36.800 56.640 speaker_0', '57.280 60.240 speaker_0', '60.800 61.200 speaker_0', '62.720 67.120 speaker_0', '68.080 69.680 speaker_0', '70.640 79.920 speaker_0', '80.080 83.520 speaker_0', '85.440 93.280 speaker_0', '94.000 102.560 speaker_0', '105.440 117.120 speaker_0', '117.200 120.880 speaker_0', '121.120 124.720 speaker_0', '125.040 128.000 speaker_0', '128.560 136.400 speaker_0', '136.880 147.760 speaker_0', '148.560 160.160 speaker_0', '160.800 166.070 speaker_0', '2.720 5.920 speaker_1', '6.480 6.640 speaker_1', '56.640 57.360 speaker_1', '61.200 62.240 speaker_1', '85.120 85.600 speaker_1', '102.480 105.360 speaker_1', '136.000 138.320 speaker_1']], [['0.000 0.240 speaker_0', '1.120 5.440 speaker_0', '6.480 17.120 speaker_0', '17.280 22.240 speaker_0', '23.280 25.120 speaker_0', '25.200 38.000 speaker_0', '38.400 43.280 speaker_0', '43.840 62.960 speaker_0', '63.600 64.800 speaker_0', '64.880 69.440 speaker_0', '69.680 70.720 speaker_0', '71.200 71.840 speaker_0', '72.880 74.480 speaker_0', '74.640 75.520 speaker_0', '76.320 82.960 speaker_0', '85.920 91.200 speaker_0', '92.000 93.280 speaker_0', '94.320 98.720 speaker_0', '99.200 102.960 speaker_0', '103.040 104.080 speaker_0', '114.320 114.560 speaker_0', '114.720 114.960 speaker_0', '119.600 119.760 speaker_0', '120.960 121.280 speaker_0', '131.680 145.680 speaker_0', '146.080 154.720 speaker_0', '155.040 158.640 speaker_0', '159.520 161.440 speaker_0', '162.000 162.560 speaker_0', '163.280 166.070 speaker_0', '70.720 71.280 speaker_1', '83.920 85.920 speaker_1', '105.040 111.600 speaker_1', '111.920 112.800 speaker_1', '113.120 116.320 speaker_1', '116.560 118.000 speaker_1', '118.560 120.640 speaker_1', '121.600 126.000 speaker_1', '126.160 131.520 speaker_1']], [['0.000 0.720 speaker_0', '1.360 2.240 speaker_0', '2.800 8.160 speaker_0', '8.800 9.680 speaker_0', '10.080 20.960 speaker_0', '21.680 22.160 speaker_0', '22.320 27.520 speaker_0', '28.320 30.320 speaker_0', '32.560 32.880 speaker_0', '35.680 37.680 speaker_0', '42.160 42.560 speaker_0', '44.320 48.400 speaker_0', '30.640 32.560 speaker_1', '32.640 35.680 speaker_1', '38.080 41.840 speaker_1', '42.960 44.240 speaker_1', '48.960 58.000 speaker_1', '59.040 65.520 speaker_1', '66.560 69.680 speaker_1', '70.000 73.600 speaker_1', '74.160 76.320 speaker_1', '76.640 77.440 speaker_1', '80.400 81.600 speaker_2', '82.880 87.280 speaker_2', '88.240 89.680 speaker_2']]]
    shortened_audios_for_diarization = ['GAG01_segment_0.wav', 'GAG01_segment_166000.wav', 'GAG01_segment_332000.wav', 'GAG01_segment_498000.wav', 'GAG01_segment_664000.wav', 'GAG01_segment_830000.wav', 'GAG01_segment_996000.wav']
    #Folge 2 Daniel
    #diarizations = [[['0.240 11.520 speaker_0', '11.760 14.800 speaker_0', '29.600 37.840 speaker_0', '38.320 40.320 speaker_0', '40.400 42.800 speaker_0', '44.160 44.720 speaker_0', '45.280 45.760 speaker_0', '46.080 54.480 speaker_0', '55.200 57.760 speaker_0', '58.000 61.440 speaker_0', '62.640 69.520 speaker_0', '69.760 69.840 speaker_0', '69.920 79.360 speaker_0', '79.520 107.360 speaker_0', '107.440 136.560 speaker_0', '137.120 166.070 speaker_0', '17.920 19.200 speaker_1', '20.400 24.720 speaker_1', '25.760 27.200 speaker_1', '61.360 62.640 speaker_2']], [['0.000 23.840 speaker_0', '29.920 32.000 speaker_0', '33.360 34.720 speaker_0', '36.480 37.360 speaker_0', '41.440 50.880 speaker_0', '51.760 53.120 speaker_0', '53.680 54.640 speaker_0', '55.680 58.000 speaker_0', '58.160 60.240 speaker_0', '60.880 61.120 speaker_0', '64.400 68.400 speaker_0', '73.840 75.600 speaker_0', '76.480 76.560 speaker_0', '77.120 77.520 speaker_0', '78.000 78.160 speaker_0', '78.480 78.640 speaker_0', '78.720 79.920 speaker_0', '80.480 86.400 speaker_0', '86.560 88.400 speaker_0', '89.120 90.720 speaker_0', '96.000 99.120 speaker_0', '100.160 100.880 speaker_0', '101.280 109.040 speaker_0', '109.920 112.240 speaker_0', '117.120 123.600 speaker_0', '124.080 126.640 speaker_0', '127.600 128.400 speaker_0', '128.720 133.120 speaker_0', '133.920 137.200 speaker_0', '143.360 143.440 speaker_0', '144.800 145.040 speaker_0', '149.120 149.360 speaker_0', '150.880 155.680 speaker_0', '156.640 162.000 speaker_0', '164.000 166.070 speaker_0', '24.160 29.600 speaker_1', '31.920 32.160 speaker_1', '32.800 32.880 speaker_1', '32.960 33.040 speaker_1', '60.720 61.120 speaker_2', '78.400 78.800 speaker_2', '115.680 115.760 speaker_2', '137.120 137.200 speaker_2', '37.760 41.200 speaker_3', '60.240 64.160 speaker_3', '68.640 72.640 speaker_3', '91.200 95.440 speaker_3', '112.720 117.200 speaker_3', '137.120 144.800 speaker_3', '145.040 148.720 speaker_3', '149.280 150.800 speaker_3']], [['0.000 2.640 speaker_0', '2.800 4.480 speaker_0', '5.520 15.200 speaker_0', '15.760 26.240 speaker_0', '26.320 28.560 speaker_0', '28.720 31.360 speaker_0', '32.320 34.880 speaker_0', '35.280 42.880 speaker_0', '43.440 48.560 speaker_0', '49.440 55.520 speaker_0', '66.400 73.360 speaker_0', '73.760 76.320 speaker_0', '76.800 81.200 speaker_0', '81.600 84.320 speaker_0', '84.960 88.320 speaker_0', '88.560 95.360 speaker_0', '95.840 99.040 speaker_0', '103.280 108.960 speaker_0', '111.680 113.680 speaker_0', '115.680 117.360 speaker_0', '117.680 120.160 speaker_0', '120.720 124.640 speaker_0', '125.280 127.120 speaker_0', '127.760 146.800 speaker_0', '150.800 158.560 speaker_0', '159.520 166.070 speaker_0', '55.760 65.600 speaker_1', '147.600 150.720 speaker_1', '99.440 100.640 speaker_2', '101.280 102.800 speaker_2', '109.040 111.360 speaker_2']], [['0.000 3.440 speaker_0', '4.320 4.640 speaker_0', '5.680 6.000 speaker_0', '6.320 6.800 speaker_0', '14.880 24.720 speaker_0', '24.960 26.400 speaker_0', '27.040 27.680 speaker_0', '28.480 30.160 speaker_0', '30.720 36.880 speaker_0', '37.520 41.680 speaker_0', '41.840 53.520 speaker_0', '53.680 63.200 speaker_0', '63.920 65.120 speaker_0', '65.440 69.360 speaker_0', '69.840 70.960 speaker_0', '71.040 79.040 speaker_0', '79.200 92.000 speaker_0', '92.960 97.440 speaker_0', '98.160 105.200 speaker_0', '105.440 111.600 speaker_0', '112.240 112.640 speaker_0', '113.120 117.600 speaker_0', '118.160 125.520 speaker_0', '127.680 129.120 speaker_0', '130.000 134.240 speaker_0', '136.000 140.720 speaker_0', '141.200 159.600 speaker_0', '159.840 163.280 speaker_0', '163.440 166.070 speaker_0', '4.400 6.320 speaker_1', '6.640 15.120 speaker_1', '126.080 128.560 speaker_1']], [['0.000 3.200 speaker_0', '3.280 3.360 speaker_0', '3.920 15.200 speaker_0', '15.520 22.880 speaker_0', '23.120 26.720 speaker_0', '26.960 39.760 speaker_0', '40.480 43.200 speaker_0', '43.680 45.920 speaker_0', '46.000 55.760 speaker_0', '55.840 56.320 speaker_0', '56.720 62.880 speaker_0', '66.080 75.360 speaker_0', '75.520 83.200 speaker_0', '83.680 84.800 speaker_0', '85.280 91.760 speaker_0', '98.720 106.240 speaker_0', '106.880 113.520 speaker_0', '113.680 125.920 speaker_0', '126.720 130.240 speaker_0', '130.880 137.360 speaker_0', '137.840 149.520 speaker_0', '150.000 155.680 speaker_0', '155.840 160.320 speaker_0', '161.200 166.070 speaker_0', '63.200 65.520 speaker_1', '92.160 98.400 speaker_1']], [['0.000 7.760 speaker_0', '8.560 12.240 speaker_0', '12.880 17.120 speaker_0', '17.200 23.920 speaker_0', '24.640 24.720 speaker_0', '25.040 26.320 speaker_0', '26.880 32.080 speaker_0', '32.960 40.240 speaker_0', '40.320 46.720 speaker_0', '47.760 48.160 speaker_0', '48.240 48.960 speaker_0', '50.000 60.880 speaker_0', '61.680 73.600 speaker_0', '74.480 77.360 speaker_0', '77.840 80.000 speaker_0', '80.640 84.000 speaker_0', '85.200 86.880 speaker_0', '88.320 88.960 speaker_0', '90.400 90.720 speaker_0', '91.360 92.800 speaker_0', '93.280 94.960 speaker_0', '107.040 107.440 speaker_0', '111.840 112.480 speaker_0', '116.880 118.960 speaker_0', '120.480 121.040 speaker_0', '125.040 125.120 speaker_0', '125.280 125.840 speaker_0', '129.440 130.400 speaker_0', '131.760 132.400 speaker_0', '133.120 134.720 speaker_0', '135.440 137.440 speaker_0', '141.600 142.000 speaker_0', '143.840 148.720 speaker_0', '87.600 88.320 speaker_1', '88.960 90.320 speaker_1', '96.320 100.560 speaker_1', '101.360 104.480 speaker_1', '104.960 106.960 speaker_1', '107.680 116.800 speaker_1', '120.000 120.480 speaker_1', '121.040 122.080 speaker_1', '122.880 126.560 speaker_1', '127.040 128.800 speaker_1', '129.200 129.680 speaker_1', '130.240 131.440 speaker_1', '132.240 134.160 speaker_1', '135.600 135.920 speaker_1', '138.960 139.360 speaker_1', '140.160 141.600 speaker_1', '142.240 143.280 speaker_1', '149.280 151.760 speaker_1', '152.160 153.760 speaker_1', '154.480 155.200 speaker_1', '158.080 159.280 speaker_2', '160.640 164.960 speaker_2', '166.000 166.070 speaker_2']], [['0.000 1.520 speaker_0']]]
    #shortened_audios_for_diarization = ['GAG02_segment_0.wav', 'GAG02_segment_166000.wav', 'GAG02_segment_332000.wav', 'GAG02_segment_498000.wav', 'GAG02_segment_664000.wav', 'GAG02_segment_830000.wav', 'GAG02_segment_996000.wav']

    speakerDanielAudios = list()
    speakerRichardAudios = list()
    speaker_times = dict()
    # Durchlaufen der verschachtelten Dict
    # speaker_times['segment'] = dict()
    # speaker_times['segment']
    diaCount= 0
    for sublist_level1 in diarizations:
        for sublist_level2 in sublist_level1:
            current_segment = shortened_audios_for_diarization[diaCount]
            diaCount+=1
            speaker_times[current_segment] = dict()
            for entry_string in sublist_level2:
                # Teilen des Strings in Startzeit, Endzeit und Sprecher-ID
                parts = entry_string.split()
                start_time = float(parts[0])
                end_time = float(parts[1])
                speaker_id = str(parts[2])

                #speaker_times[]
                # Prüfe, ob der Hauptschlüssel existiert, ansonsten initialisiere ihn
                if speaker_id not in speaker_times[current_segment]:
                    speaker_times[current_segment][speaker_id] = {} # Initialisiere den Wert als leeres Dictionary

                if 'start_time' not in speaker_times[current_segment][speaker_id]:
                    speaker_times[current_segment][speaker_id]['start_time'] = list()
                
                if 'end_time' not in speaker_times[current_segment][speaker_id]:
                    speaker_times[current_segment][speaker_id]['end_time'] = list()

                #sicher Elemente anhängen
                speaker_times[current_segment][speaker_id]['start_time'].append(start_time)
                speaker_times[current_segment][speaker_id]['end_time'].append(end_time)

    #Audiosegmente mit Sprechanteile rausschneiden, um daraus embeddings zu machen: 

    for key in speaker_times.keys():

        for key2 in speaker_times[key].keys():

            print(key2)
            print(speaker_times[key][key2]['start_time'])
            print(speaker_times[key][key2]['end_time']) 

    print(speaker_times)

        # Define the target sample rate for the NeMo model
    TARGET_SAMPLE_RATE = 16000 # Most ASR/Speaker Recognition models expect 16kHz mono

    richardEmbeddings = list()
    danielEmbeddings = list()
    current_audio_segment = list()
    richard_audio_segments = list()
    segment_index = -1
    segmentOne = None
    for key in speaker_times.keys():

        segment_index += 1
        print(f"Processing main audio file: {key}")
        audio_wav = AudioSegment.from_wav(key)
        print("len(audio_wav)")
        print(len(audio_wav))
        for key2 in speaker_times[key].keys():
            if key2 == 'speaker_2':
                    
                continue
            
            print(f"  Processing speaker: {key2}")
            print(f"  Start times: {speaker_times[key][key2]['start_time']}")
            print(f"  End times: {speaker_times[key][key2]['end_time']}") 

            for i in range(0, len(speaker_times[key][key2]['start_time']), 1):
                start_ms = speaker_times[key][key2]['start_time'][i]*1000
                end_ms = speaker_times[key][key2]['end_time'][i]*1000
                if key2 == 'speaker_0': 
                    if key == "GAG01_segment_0.wav" and segmentOne == None:
                        segmentOne = audio_wav[start_ms:end_ms]
                        print("len(segmentOne)")
                        print(len(segmentOne))
                    if key == "GAG01_segment_0.wav" and segmentOne != None:
                        segmentOne += audio_wav[start_ms:end_ms]
                        print("len(segmentOne)")
                        print(len(segmentOne))
                    if current_audio_segment:
                        current_audio_segment[0] += audio_wav[start_ms:end_ms]
                        print("len(current_audio_segment[0])")
                        print(len(current_audio_segment[0]))
                    else:
                        current_audio_segment.append(audio_wav[start_ms:end_ms])

                    if richard_audio_segments and len(richard_audio_segments)==1 and segment_index==0: 

                        richard_audio_segments[segment_index] += audio_wav[start_ms:end_ms]
                        print("len(richard_audio_segments[segment_index])")
                        print(len(richard_audio_segments[segment_index]))
                    elif len(richard_audio_segments)<segment_index+1 and segment_index!=0:

                        richard_audio_segments.append(audio_wav[start_ms:end_ms])
                    elif len(richard_audio_segments)==segment_index+1 and segment_index!=0:

                        richard_audio_segments[segment_index] += audio_wav[start_ms:end_ms]
                    else:
                        richard_audio_segments.append(audio_wav[start_ms:end_ms])


                if key2 == 'speaker_1': 
                    continue
                    if len(current_audio_segment)>2:
                        current_audio_segment[1] += audio_wav[start_ms:end_ms]
                        print(len(current_audio_segment[1]))
                    else:
                        current_audio_segment.append(audio_wav[start_ms:end_ms])
                    
                    
                    # --- Key changes below ---
                    
                    # 1. Ensure audio is mono and at the target sample rate
                    # NeMo models often expect mono 16kHz audio.
                    # Pydub's .set_channels(1) ensures mono.
                    # Pydub's .set_frame_rate() resamples if needed.

    if segmentOne.channels != 1:
            segmentOne = segmentOne.set_channels(1)
                        
    if segmentOne.frame_rate != TARGET_SAMPLE_RATE:
        segmentOne = segmentOne.set_frame_rate(TARGET_SAMPLE_RATE)

                        # Create a temporary file to save the audio segment
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        temp_audio_path = tmpfile.name
        segmentOne.export(temp_audio_path, format="wav")
                        
        try:
                            # Get the embedding using the temporary file path
            embedding_segmentOne = speaker_model.get_embedding(temp_audio_path)
                            
                            # The embedding returned by get_embedding is usually already a PyTorch tensor.
                            # You can check its shape if you want:
                            # print(f"  Embedding shape: {embedding.shape}")

            print("embedding_segmentOne:")
            print(embedding_segmentOne)

        finally:
                            # Clean up the temporary file
            os.remove(temp_audio_path)

    print("länge Richard Segmente: ")
    print(len(current_audio_segment[0]))
    output_filename = "saved_audio_segment_richard.wav"        
    # Export the audio segment to a WAV file
    current_audio_segment[0].export(output_filename, format="wav")

    for i in range(0, len(current_audio_segment), 1):
        
        if i==1:
            continue
        if current_audio_segment[i].channels != 1:
            current_audio_segment[i] = current_audio_segment[i].set_channels(1)
                        
        if current_audio_segment[i].frame_rate != TARGET_SAMPLE_RATE:
            current_audio_segment[i] = current_audio_segment[i].set_frame_rate(TARGET_SAMPLE_RATE)

                        # Create a temporary file to save the audio segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_audio_path = tmpfile.name
            current_audio_segment[i].export(temp_audio_path, format="wav")
                        
            try:
                            # Get the embedding using the temporary file path
                embedding = speaker_model.get_embedding(temp_audio_path)
                            
                            # The embedding returned by get_embedding is usually already a PyTorch tensor.
                            # You can check its shape if you want:
                            # print(f"  Embedding shape: {embedding.shape}")

                if i == 0:
                    richardEmbeddings.append(embedding)
                else:
                    danielEmbeddings.append(embedding)
            finally:
                            # Clean up the temporary file
                os.remove(temp_audio_path)

    print("Embeddings extraction complete.")
    print(f"Number of Richard embeddings: {len(richardEmbeddings)}")
    print(f"Number of Daniel embeddings: {len(danielEmbeddings)}")
    print(richardEmbeddings)
    print(danielEmbeddings)
    
    embedding_list = richardEmbeddings[0].cpu().squeeze().tolist()

    print(embedding_list)

    # ChromaDB Client initialisieren (Für eine persistente Datenbank)
    # client = chromadb.PersistentClient(path="/path/to/your/chromadb_data") # Lokaler Pfad zum Speichern
    #client = chromadb.Client() # In-Memory Client für schnelles Testen
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Definiere den Ordnernamen für die ChromaDB-Daten
    db_directory = os.path.join(script_dir, "chroma_db_speakers")
    client = chromadb.PersistentClient(path=db_directory)
    # Eine Collection erstellen oder abrufen
    collection_name = "speaker_embeddings"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except:
        collection = client.create_collection(name=collection_name)
        print(f"Collection '{collection_name}' created.")

    # Metadaten hinzufügen, um das Embedding zu identifizieren
    # Sie sollten hier sinnvolle IDs und Metadaten verwenden
    # z.B. welcher Sprecher, aus welcher Datei, Start-/Endzeit
    embedding_id = "Richard_audio_embedding" # Eindeutige ID
    metadata = {
        "speaker_id": "Richard_Speaker_Embedding",
        #"audio_file": "my_meeting.wav",
        #"start_time_ms": 12345,
        #"end_time_ms": 15678
    }

    # Embedding hinzufügen
    collection.add(
        embeddings=embedding_list,  # Muss eine Liste von Embeddings sein, auch wenn es nur eins ist
        metadatas=[metadata],         # Liste von Metadaten, passend zu Embeddings
        ids=[embedding_id]            # Liste von IDs, passend zu Embeddings
    )

    print(f"Embedding '{embedding_id}' erfolgreich in ChromaDB gespeichert.")

    
    result = collection.get(
        ids=[embedding_id]
    )
    print(result)
    
    results = collection.get(
        ids=[embedding_id], 
        include=['embeddings']
    )

    print(results)

    # Beispiel: Abfragen des Embeddings
    results = collection.query(
        query_embeddings=[embedding_list], # Query mit dem gleichen Embedding
        n_results=1,
        include=['embeddings']
    )

    print(results)

    richeard_seperate_segments_embeddings = list()
    #Seperate Richard Embeddings: 
    for i in range(0, len(richard_audio_segments), 1):

        output_filename_list = f"saved_audio_segment_richard_{i}.wav"        
        # Export the audio segment to a WAV file
        richard_audio_segments[i].export(output_filename_list, format="wav")

    for i in range(0, len(richard_audio_segments), 1):
        
        if richard_audio_segments[i].channels != 1:
            richard_audio_segments[i] = richard_audio_segments[i].set_channels(1)
                        
        if richard_audio_segments[i].frame_rate != TARGET_SAMPLE_RATE:
            richard_audio_segments[i] = richard_audio_segments[i].set_frame_rate(TARGET_SAMPLE_RATE)

                        # Create a temporary file to save the audio segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_audio_path = tmpfile.name
            richard_audio_segments[i].export(temp_audio_path, format="wav")
                        
            try:
                            # Get the embedding using the temporary file path
                embedding = speaker_model.get_embedding(temp_audio_path)
                            
                            # The embedding returned by get_embedding is usually already a PyTorch tensor.
                            # You can check its shape if you want:
                            # print(f"  Embedding shape: {embedding.shape}")

                richeard_seperate_segments_embeddings.append(embedding.cpu().squeeze().tolist())
            except Exception as e:
                continue
            finally:
                            # Clean up the temporary file
                os.remove(temp_audio_path)
    
    for i in range(0, len(richeard_seperate_segments_embeddings), 1): 

        print(f"richeard_seperate_segments_embeddings{i}")
        print(richeard_seperate_segments_embeddings[i])
        embedding_id = f"richeard_seperate_segments_embeddings{i}" # Eindeutige ID
        metadata = {
            "speaker_id": "Richard_Speaker_Embedding",
            #"audio_file": "my_meeting.wav",
            #"start_time_ms": 12345,
            #"end_time_ms": 15678
        }

        # Embedding hinzufügen
        collection.add(
            embeddings=richeard_seperate_segments_embeddings[i],  # Muss eine Liste von Embeddings sein, auch wenn es nur eins ist
            metadatas=[metadata],         # Liste von Metadaten, passend zu Embeddings
            ids=[embedding_id]            # Liste von IDs, passend zu Embeddings
        )

        print(f"Embedding '{embedding_id}' erfolgreich in ChromaDB gespeichert.")
    
    for i in range(0, len(richeard_seperate_segments_embeddings), 1):

        embedding_id = f"richeard_seperate_segments_embeddings{i}"
        results = collection.get(
            
            ids=[embedding_id], 
            include=['embeddings']
        )

        print("GET RESULT FROM CHROMADB")
        print(results)
    #print("Embeddings extraction complete.")
    #print(f"Number of Richard embeddings: {len(richardEmbeddings)}")
    #print(f"Number of Daniel embeddings: {len(danielEmbeddings)}")
    #print(richardEmbeddings)
    #print(danielEmbeddings)
    
    #embedding_list = richardEmbeddings[0].cpu().squeeze().tolist()

def split_audio_for_speaker_embeddings_Daniel():

    #os.chdir("..")
    #os.chdir("audioData")
    #os.chdir("shortened_audios_wav")
    #Folge 1 Richard Embeddings
    #diarizations = [[['0.240 4.320 speaker_0', '11.840 14.800 speaker_0', '34.800 35.760 speaker_0', '54.720 57.440 speaker_0', '58.080 63.920 speaker_0', '64.800 67.920 speaker_0', '68.080 70.880 speaker_0', '72.960 76.160 speaker_0', '76.880 79.120 speaker_0', '80.160 80.240 speaker_0', '80.400 81.120 speaker_0', '82.240 85.040 speaker_0', '86.000 88.560 speaker_0', '93.520 103.040 speaker_0', '103.600 109.280 speaker_0', '109.440 116.400 speaker_0', '117.040 121.360 speaker_0', '121.760 134.560 speaker_0', '135.920 146.800 speaker_0', '147.520 155.360 speaker_0', '155.920 160.000 speaker_0', '161.040 165.040 speaker_0', '165.120 166.070 speaker_0', '4.560 11.520 speaker_1', '29.200 34.720 speaker_1', '35.920 50.960 speaker_1', '51.840 52.400 speaker_1', '52.720 54.560 speaker_1', '71.360 73.040 speaker_1', '88.640 90.880 speaker_1', '91.200 94.080 speaker_1', '17.920 19.120 speaker_2', '20.400 24.800 speaker_2', '25.760 27.200 speaker_2']], [['0.000 3.520 speaker_0', '4.240 5.520 speaker_0', '6.000 18.960 speaker_0', '19.840 22.800 speaker_0', '23.920 24.080 speaker_0', '26.640 28.720 speaker_0', '29.280 39.040 speaker_0', '39.120 45.360 speaker_0', '46.320 46.560 speaker_0', '46.720 53.760 speaker_0', '54.640 55.760 speaker_0', '65.040 65.200 speaker_0', '75.040 86.080 speaker_0', '86.160 91.360 speaker_0', '93.040 93.120 speaker_0', '93.200 118.240 speaker_0', '118.720 123.600 speaker_0', '126.720 126.880 speaker_0', '127.680 129.520 speaker_0', '129.680 130.400 speaker_0', '131.120 135.200 speaker_0', '136.000 136.880 speaker_0', '144.080 152.480 speaker_0', '152.800 157.600 speaker_0', '158.080 158.240 speaker_0', '158.400 163.280 speaker_0', '163.360 163.520 speaker_0', '163.600 166.070 speaker_0', '23.840 28.320 speaker_1', '56.720 58.240 speaker_1', '58.480 61.200 speaker_1', '61.680 61.840 speaker_1', '62.480 70.640 speaker_1', '70.960 74.880 speaker_1', '123.920 125.360 speaker_1', '125.680 127.520 speaker_1', '136.400 142.880 speaker_1', '143.280 144.080 speaker_1']], [['0.000 0.880 speaker_0', '5.840 6.320 speaker_0', '8.240 11.520 speaker_0', '12.960 23.760 speaker_0', '62.640 69.120 speaker_0', '73.200 79.040 speaker_0', '80.400 83.600 speaker_0', '90.160 93.200 speaker_0', '93.760 99.680 speaker_0', '100.240 115.280 speaker_0', '116.400 124.000 speaker_0', '124.880 137.520 speaker_0', '138.320 140.320 speaker_0', '140.800 148.560 speaker_0', '148.880 150.880 speaker_0', '151.520 154.560 speaker_0', '155.040 166.070 speaker_0', '1.920 7.760 speaker_1', '24.400 30.640 speaker_1', '69.680 72.160 speaker_1', '72.240 73.520 speaker_1', '79.840 80.800 speaker_1', '83.680 85.840 speaker_1', '86.320 90.160 speaker_1', '31.520 35.680 speaker_2', '36.160 38.960 speaker_2', '39.760 41.520 speaker_2', '42.000 47.200 speaker_2', '47.760 53.520 speaker_2', '54.080 58.080 speaker_2', '58.560 59.600 speaker_2', '59.840 61.840 speaker_2']], [['0.000 5.440 speaker_0', '6.400 7.360 speaker_0', '7.840 16.400 speaker_0', '22.560 28.000 speaker_0', '28.480 33.680 speaker_0', '34.640 40.480 speaker_0', '41.120 41.520 speaker_0', '41.600 41.760 speaker_0', '42.160 45.200 speaker_0', '46.000 62.560 speaker_0', '62.800 66.400 speaker_0', '66.480 74.240 speaker_0', '74.800 78.560 speaker_0', '79.600 83.520 speaker_0', '84.000 88.640 speaker_0', '88.800 93.040 speaker_0', '94.400 95.120 speaker_0', '101.760 116.560 speaker_0', '117.200 118.080 speaker_0', '118.720 128.880 speaker_0', '129.200 129.760 speaker_0', '130.080 132.480 speaker_0', '132.960 133.600 speaker_0', '134.080 135.520 speaker_0', '136.400 141.520 speaker_0', '141.680 146.080 speaker_0', '147.280 147.920 speaker_0', '148.240 154.320 speaker_0', '155.360 166.070 speaker_0', '16.560 22.400 speaker_1', '93.440 98.960 speaker_1', '99.040 101.360 speaker_1']], [['0.000 0.080 speaker_0', '0.800 2.560 speaker_0', '5.680 11.600 speaker_0', '12.400 19.200 speaker_0', '19.840 28.960 speaker_0', '29.680 35.920 speaker_0', '36.800 56.640 speaker_0', '57.280 60.240 speaker_0', '60.800 61.200 speaker_0', '62.720 67.120 speaker_0', '68.080 69.680 speaker_0', '70.640 79.920 speaker_0', '80.080 83.520 speaker_0', '85.440 93.280 speaker_0', '94.000 102.560 speaker_0', '105.440 117.120 speaker_0', '117.200 120.880 speaker_0', '121.120 124.720 speaker_0', '125.040 128.000 speaker_0', '128.560 136.400 speaker_0', '136.880 147.760 speaker_0', '148.560 160.160 speaker_0', '160.800 166.070 speaker_0', '2.720 5.920 speaker_1', '6.480 6.640 speaker_1', '56.640 57.360 speaker_1', '61.200 62.240 speaker_1', '85.120 85.600 speaker_1', '102.480 105.360 speaker_1', '136.000 138.320 speaker_1']], [['0.000 0.240 speaker_0', '1.120 5.440 speaker_0', '6.480 17.120 speaker_0', '17.280 22.240 speaker_0', '23.280 25.120 speaker_0', '25.200 38.000 speaker_0', '38.400 43.280 speaker_0', '43.840 62.960 speaker_0', '63.600 64.800 speaker_0', '64.880 69.440 speaker_0', '69.680 70.720 speaker_0', '71.200 71.840 speaker_0', '72.880 74.480 speaker_0', '74.640 75.520 speaker_0', '76.320 82.960 speaker_0', '85.920 91.200 speaker_0', '92.000 93.280 speaker_0', '94.320 98.720 speaker_0', '99.200 102.960 speaker_0', '103.040 104.080 speaker_0', '114.320 114.560 speaker_0', '114.720 114.960 speaker_0', '119.600 119.760 speaker_0', '120.960 121.280 speaker_0', '131.680 145.680 speaker_0', '146.080 154.720 speaker_0', '155.040 158.640 speaker_0', '159.520 161.440 speaker_0', '162.000 162.560 speaker_0', '163.280 166.070 speaker_0', '70.720 71.280 speaker_1', '83.920 85.920 speaker_1', '105.040 111.600 speaker_1', '111.920 112.800 speaker_1', '113.120 116.320 speaker_1', '116.560 118.000 speaker_1', '118.560 120.640 speaker_1', '121.600 126.000 speaker_1', '126.160 131.520 speaker_1']], [['0.000 0.720 speaker_0', '1.360 2.240 speaker_0', '2.800 8.160 speaker_0', '8.800 9.680 speaker_0', '10.080 20.960 speaker_0', '21.680 22.160 speaker_0', '22.320 27.520 speaker_0', '28.320 30.320 speaker_0', '32.560 32.880 speaker_0', '35.680 37.680 speaker_0', '42.160 42.560 speaker_0', '44.320 48.400 speaker_0', '30.640 32.560 speaker_1', '32.640 35.680 speaker_1', '38.080 41.840 speaker_1', '42.960 44.240 speaker_1', '48.960 58.000 speaker_1', '59.040 65.520 speaker_1', '66.560 69.680 speaker_1', '70.000 73.600 speaker_1', '74.160 76.320 speaker_1', '76.640 77.440 speaker_1', '80.400 81.600 speaker_2', '82.880 87.280 speaker_2', '88.240 89.680 speaker_2']]]
    #shortened_audios_for_diarization = ['GAG01_segment_0.wav', 'GAG01_segment_166000.wav', 'GAG01_segment_332000.wav', 'GAG01_segment_498000.wav', 'GAG01_segment_664000.wav', 'GAG01_segment_830000.wav', 'GAG01_segment_996000.wav']
    #Folge 2 Daniel
    diarizations = [[['0.240 11.600 speaker_0', '11.760 14.800 speaker_0', '29.680 37.840 speaker_0', '38.640 43.120 speaker_0', '44.480 45.040 speaker_0', '45.600 46.080 speaker_0', '46.400 54.880 speaker_0', '55.600 58.000 speaker_0', '58.480 58.880 speaker_0', '59.040 59.120 speaker_0', '59.200 61.200 speaker_0', '61.280 61.920 speaker_0', '62.880 69.840 speaker_0', '70.240 79.760 speaker_0', '80.080 107.680 speaker_0', '107.840 136.880 speaker_0', '137.440 166.070 speaker_0', '17.920 19.200 speaker_1', '20.400 24.800 speaker_1', '25.760 27.200 speaker_1', '61.840 62.320 speaker_2', '96.000 96.240 speaker_2']], [['0.000 5.600 speaker_0', '5.840 24.160 speaker_0', '30.240 32.320 speaker_0', '33.760 35.120 speaker_0', '24.480 29.920 speaker_1', '58.240 59.120 speaker_1', '60.000 60.880 speaker_1', '61.040 66.720 speaker_1', '67.200 69.840 speaker_1', '70.480 74.400 speaker_1', '103.600 105.280 speaker_1', '105.360 119.200 speaker_1', '119.760 126.480 speaker_1', '126.960 129.760 speaker_1', '130.560 134.720 speaker_1', '138.080 151.120 speaker_1', '165.520 166.070 speaker_1', '36.800 37.680 speaker_2', '37.680 37.760 speaker_3', '38.640 44.480 speaker_3', '45.040 52.480 speaker_3', '52.960 54.400 speaker_3', '55.600 58.000 speaker_3', '74.560 76.160 speaker_3', '76.720 86.240 speaker_3', '86.640 89.760 speaker_3', '90.240 95.120 speaker_3', '95.520 102.960 speaker_3', '119.280 119.600 speaker_3', '134.400 136.160 speaker_3', '136.560 137.760 speaker_3', '142.640 143.040 speaker_3', '150.480 150.800 speaker_3', '152.080 165.200 speaker_3']], [['0.000 23.920 speaker_0', '25.280 38.080 speaker_0', '39.280 72.720 speaker_0', '73.440 75.280 speaker_0', '75.600 81.440 speaker_0', '81.760 95.440 speaker_0', '156.080 166.070 speaker_0', '13.200 13.600 speaker_1', '23.920 24.320 speaker_1', '35.680 36.160 speaker_1', '38.000 38.880 speaker_1', '47.920 48.160 speaker_1', '59.040 59.600 speaker_1', '73.200 73.520 speaker_1', '75.200 76.080 speaker_1', '81.360 81.760 speaker_1', '128.640 146.720 speaker_1', '147.280 154.800 speaker_1', '96.880 104.560 speaker_2', '105.280 127.440 speaker_2']], [['0.000 8.960 speaker_0', '9.520 15.040 speaker_0', '16.240 23.600 speaker_0', '24.320 27.600 speaker_0', '27.840 39.680 speaker_0', '39.840 41.680 speaker_0', '41.760 59.520 speaker_0', '59.920 66.640 speaker_0', '67.280 70.160 speaker_0', '70.560 73.840 speaker_0', '73.920 75.040 speaker_0', '75.840 77.200 speaker_0', '77.280 77.680 speaker_0', '78.400 79.200 speaker_0', '79.360 81.200 speaker_0', '81.600 86.560 speaker_0', '87.920 93.920 speaker_0', '94.480 101.920 speaker_0', '102.640 111.680 speaker_0', '112.000 116.640 speaker_0', '118.480 119.280 speaker_0', '121.280 127.600 speaker_0', '127.840 131.440 speaker_0', '132.800 133.280 speaker_0', '133.520 136.640 speaker_0', '136.800 140.000 speaker_0', '140.160 143.760 speaker_0', '153.760 161.280 speaker_0', '161.760 166.070 speaker_0', '8.960 9.360 speaker_1', '14.240 16.320 speaker_1', '16.640 17.520 speaker_1', '17.600 18.000 speaker_1', '18.080 18.560 speaker_1', '19.840 20.720 speaker_1', '23.840 24.240 speaker_1', '27.600 27.920 speaker_1', '35.760 36.080 speaker_1', '36.960 37.280 speaker_1', '39.440 39.840 speaker_1', '41.520 41.920 speaker_1', '48.560 48.800 speaker_1', '52.400 52.720 speaker_1', '70.240 70.640 speaker_1', '75.280 75.680 speaker_1', '77.200 79.440 speaker_1', '81.120 81.920 speaker_1', '86.560 87.120 speaker_1', '94.000 94.400 speaker_1', '102.240 102.640 speaker_1', '107.840 108.640 speaker_1', '111.680 112.000 speaker_1', '117.040 117.840 speaker_1', '118.400 118.560 speaker_1', '119.680 120.480 speaker_1', '131.360 131.920 speaker_1', '139.040 139.120 speaker_1', '139.840 140.640 speaker_1', '143.760 153.760 speaker_1', '161.360 161.840 speaker_1']], [['0.000 5.200 speaker_0', '5.360 16.960 speaker_0', '17.680 19.280 speaker_0', '19.600 22.000 speaker_0', '22.320 26.240 speaker_0', '26.320 39.360 speaker_0', '40.560 49.200 speaker_0', '49.680 56.000 speaker_0', '56.800 77.120 speaker_0', '78.160 86.720 speaker_0', '87.360 88.720 speaker_0', '89.040 97.200 speaker_0', '97.520 100.640 speaker_0', '101.600 104.320 speaker_0', '104.800 112.480 speaker_0', '112.720 117.200 speaker_0', '117.520 122.640 speaker_0', '129.360 130.160 speaker_0', '132.160 135.040 speaker_0', '140.960 153.280 speaker_0', '153.600 166.070 speaker_0', '3.920 4.320 speaker_1', '5.120 5.440 speaker_1', '14.960 15.360 speaker_1', '16.960 17.440 speaker_1', '22.000 22.400 speaker_1', '26.240 26.640 speaker_1', '27.600 28.000 speaker_1', '28.240 28.640 speaker_1', '35.840 36.240 speaker_1', '39.280 41.520 speaker_1', '43.600 43.920 speaker_1', '67.840 69.440 speaker_1', '72.160 72.880 speaker_1', '73.920 74.160 speaker_1', '74.240 74.320 speaker_1', '76.880 77.280 speaker_1', '85.520 86.240 speaker_1', '88.720 89.200 speaker_1', '96.640 97.520 speaker_1', '100.480 101.280 speaker_1', '117.120 117.520 speaker_1', '121.760 129.200 speaker_1', '130.160 131.840 speaker_1', '135.040 140.080 speaker_1', '153.200 153.680 speaker_1', '163.600 164.080 speaker_1']], [['0.000 4.800 speaker_0', '5.680 13.360 speaker_0', '14.240 27.760 speaker_0', '52.160 55.280 speaker_0', '59.760 65.360 speaker_0', '69.120 82.000 speaker_0', '93.200 107.520 speaker_0', '107.840 108.960 speaker_0', '109.120 119.440 speaker_0', '120.240 134.320 speaker_0', '134.960 135.280 speaker_0', '135.760 142.640 speaker_0', '142.880 147.120 speaker_0', '147.280 163.520 speaker_0', '5.440 5.680 speaker_1', '24.000 24.240 speaker_1', '27.840 28.160 speaker_1', '28.960 51.040 speaker_1', '54.560 57.280 speaker_1', '57.600 59.360 speaker_1', '64.960 69.280 speaker_1', '69.360 69.600 speaker_1', '77.280 77.760 speaker_1', '80.480 80.800 speaker_1', '81.920 90.720 speaker_1', '91.440 93.200 speaker_1', '125.600 125.920 speaker_1', '128.720 129.280 speaker_1', '130.000 130.240 speaker_1', '134.240 135.760 speaker_1', '142.800 143.040 speaker_1', '146.480 147.440 speaker_1', '155.200 155.520 speaker_1', '157.760 158.480 speaker_1', '163.920 166.070 speaker_1']], [['0.000 1.280 speaker_0', '1.920 3.680 speaker_0', '4.400 8.160 speaker_0', '11.520 12.000 speaker_0', '12.320 17.760 speaker_0', '18.880 26.960 speaker_0', '4.080 4.480 speaker_1', '8.320 11.520 speaker_1', '12.000 12.400 speaker_1', '17.840 18.720 speaker_1', '29.680 30.960 speaker_2', '32.240 36.560 speaker_2', '37.520 38.960 speaker_2']]]
    shortened_audios_for_diarization = ['GAG02_segment_0.wav', 'GAG02_segment_166000.wav', 'GAG02_segment_332000.wav', 'GAG02_segment_498000.wav', 'GAG02_segment_664000.wav', 'GAG02_segment_830000.wav', 'GAG02_segment_996000.wav']

    speakerDanielAudios = list()
    speakerRichardAudios = list()
    speaker_times = dict()
    # Durchlaufen der verschachtelten Dict
    # speaker_times['segment'] = dict()
    # speaker_times['segment']
    diaCount= 0
    for sublist_level1 in diarizations:
        for sublist_level2 in sublist_level1:
            current_segment = shortened_audios_for_diarization[diaCount]
            diaCount+=1
            speaker_times[current_segment] = dict()
            for entry_string in sublist_level2:
                # Teilen des Strings in Startzeit, Endzeit und Sprecher-ID
                parts = entry_string.split()
                start_time = float(parts[0])
                end_time = float(parts[1])
                speaker_id = str(parts[2])

                #speaker_times[]
                # Prüfe, ob der Hauptschlüssel existiert, ansonsten initialisiere ihn
                if speaker_id not in speaker_times[current_segment]:
                    speaker_times[current_segment][speaker_id] = {} # Initialisiere den Wert als leeres Dictionary

                if 'start_time' not in speaker_times[current_segment][speaker_id]:
                    speaker_times[current_segment][speaker_id]['start_time'] = list()
                
                if 'end_time' not in speaker_times[current_segment][speaker_id]:
                    speaker_times[current_segment][speaker_id]['end_time'] = list()

                #sicher Elemente anhängen
                speaker_times[current_segment][speaker_id]['start_time'].append(start_time)
                speaker_times[current_segment][speaker_id]['end_time'].append(end_time)

    #Audiosegmente mit Sprechanteile rausschneiden, um daraus embeddings zu machen: 

    for key in speaker_times.keys():

        for key2 in speaker_times[key].keys():

            print(key2)
            print(speaker_times[key][key2]['start_time'])
            print(speaker_times[key][key2]['end_time']) 

    print(speaker_times)

    # Define the target sample rate for the NeMo model
    TARGET_SAMPLE_RATE = 16000 # Most ASR/Speaker Recognition models expect 16kHz mono

    richardEmbeddings = list()
    danielEmbeddings = list()
    current_audio_segment = list()
    
    daniel_audio_segments = list()
     
    segmentOne = None
    segment_index = -1
    speaker_for_current_segment = ['speaker_1', 'speaker_1', 'speaker_0', 'speaker_0', 'speaker_0', 'speaker_0', 'speaker_1']
    speaker_for_current_segment_index: int = 0
    for key in speaker_times.keys():

        segment_index += 1
        print(f"Processing main audio file: {key}")
        audio_wav = AudioSegment.from_wav(key)
        for key2 in speaker_times[key].keys():
            if key2 == 'speaker_2':
                    
                continue
            
            print(f"  Processing speaker: {key2}")
            print(f"  Start times: {speaker_times[key][key2]['start_time']}")
            print(f"  End times: {speaker_times[key][key2]['end_time']}") 

            for i in range(0, len(speaker_times[key][key2]['start_time']), 1):
                start_ms = speaker_times[key][key2]['start_time'][i]*1000
                end_ms = speaker_times[key][key2]['end_time'][i]*1000
                if key2 == speaker_for_current_segment[speaker_for_current_segment_index]:

                    print(speaker_for_current_segment[speaker_for_current_segment_index])
                    print(start_ms)
                    print(end_ms)
                    if key == "GAG02_segment_0.wav" and segmentOne == None:
                        segmentOne = audio_wav[start_ms:end_ms]
                    if key == "GAG02_segment_0.wav" and segmentOne != None:
                        segmentOne += audio_wav[start_ms:end_ms]
                    if current_audio_segment:
                        current_audio_segment[0] += audio_wav[start_ms:end_ms]
                        print(len(current_audio_segment[0]))
                    else:
                        current_audio_segment.append(audio_wav[start_ms:end_ms])
                        print(len(current_audio_segment))

                    if daniel_audio_segments and len(daniel_audio_segments)==1 and segment_index==0: 

                        daniel_audio_segments[segment_index] += audio_wav[start_ms:end_ms]
                        print("len(richard_audio_segments[segment_index])")
                        print(len(daniel_audio_segments[segment_index]))
                    elif len(daniel_audio_segments)<segment_index+1 and segment_index!=0:

                        daniel_audio_segments.append(audio_wav[start_ms:end_ms])
                    elif len(daniel_audio_segments)==segment_index+1 and segment_index!=0:

                        daniel_audio_segments[segment_index] += audio_wav[start_ms:end_ms]
                    else:
                        daniel_audio_segments.append(audio_wav[start_ms:end_ms])

        print("speaker_for_current_segment_index")
        print(speaker_for_current_segment_index)
        speaker_for_current_segment_index+=1             
        print("speaker_for_current_segment_index")
        print(speaker_for_current_segment_index)

                    # --- Key changes below ---
                    
                    # 1. Ensure audio is mono and at the target sample rate
                    # NeMo models often expect mono 16kHz audio.
                    # Pydub's .set_channels(1) ensures mono.
                    # Pydub's .set_frame_rate() resamples if needed.
    
    print("länge Dnaiel Segmente: ")
    print(len(current_audio_segment[0]))
    # Define the filename
    output_filename = "saved_audio_segment_daniel.wav"
    # Export the audio segment to a WAV file
    current_audio_segment[0].export(output_filename, format="wav")
    print("segmentOne")
    print(segmentOne)
    """
    if segmentOne.channels != 1:
            segmentOne = segmentOne.set_channels(1)
                        
    if segmentOne.frame_rate != TARGET_SAMPLE_RATE:
        segmentOne = segmentOne.set_frame_rate(TARGET_SAMPLE_RATE)

                        # Create a temporary file to save the audio segment
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        temp_audio_path = tmpfile.name
        segmentOne.export(temp_audio_path, format="wav")
                        
        try:
                            # Get the embedding using the temporary file path
            embedding_segmentOne = speaker_model.get_embedding(temp_audio_path)
                            
                            # The embedding returned by get_embedding is usually already a PyTorch tensor.
                            # You can check its shape if you want:
                            # print(f"  Embedding shape: {embedding.shape}")

            print("embedding_segmentOne:")
            print(embedding_segmentOne)

        finally:
                            # Clean up the temporary file
            os.remove(temp_audio_path)
    """
    for i in range(0, len(current_audio_segment), 1):
        
        if i==1:
            continue
        if current_audio_segment[i].channels != 1:
            current_audio_segment[i] = current_audio_segment[i].set_channels(1)
                        
        if current_audio_segment[i].frame_rate != TARGET_SAMPLE_RATE:
            current_audio_segment[i] = current_audio_segment[i].set_frame_rate(TARGET_SAMPLE_RATE)

                        # Create a temporary file to save the audio segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_audio_path = tmpfile.name
            current_audio_segment[i].export(temp_audio_path, format="wav")
                        
            try:
                            # Get the embedding using the temporary file path
                embedding = speaker_model.get_embedding(temp_audio_path)
                            
                            # The embedding returned by get_embedding is usually already a PyTorch tensor.
                            # You can check its shape if you want:
                            # print(f"  Embedding shape: {embedding.shape}")

                if i == 0:
                    danielEmbeddings.append(embedding)
                else:
                    richardEmbeddings.append(embedding)
            finally:
                            # Clean up the temporary file
                os.remove(temp_audio_path)

    print("Embeddings extraction complete.")
    print(f"Number of Richard embeddings: {len(richardEmbeddings)}")
    print(f"Number of Daniel embeddings: {len(danielEmbeddings)}")
    print(richardEmbeddings)
    print(danielEmbeddings)
    
    embedding_list = danielEmbeddings[0].cpu().squeeze().tolist()

    print(embedding_list)
    # ChromaDB Client initialisieren (Für eine persistente Datenbank)
    # client = chromadb.PersistentClient(path="/path/to/your/chromadb_data") # Lokaler Pfad zum Speichern
    # Ermittle den Pfad zum aktuellen Skript-Verzeichnis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Definiere den Ordnernamen für die ChromaDB-Daten
    db_directory = os.path.join(script_dir, "chroma_db_speakers")
    client = chromadb.PersistentClient(path=db_directory)

    #client = chromadb.Client() # In-Memory Client für schnelles Testen

    # Eine Collection erstellen oder abrufen
    collection_name = "speaker_embeddings"
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except:
        collection = client.create_collection(name=collection_name)
        print(f"Collection '{collection_name}' created.")

    # Metadaten hinzufügen, um das Embedding zu identifizieren
    # Sie sollten hier sinnvolle IDs und Metadaten verwenden
    # z.B. welcher Sprecher, aus welcher Datei, Start-/Endzeit
    embedding_id = "Daniel_audio_embedding" # Eindeutige ID
    metadata = {
        "speaker_id": "Daniel_Speaker_Embedding",
        #"audio_file": "my_meeting.wav",
        #"start_time_ms": 12345,
        #"end_time_ms": 15678
    }

    # Embedding hinzufügen
    collection.add(
        embeddings=embedding_list,  # Muss eine Liste von Embeddings sein, auch wenn es nur eins ist
        metadatas=[metadata],         # Liste von Metadaten, passend zu Embeddings
        ids=[embedding_id]            # Liste von IDs, passend zu Embeddings
    )

    print(f"Embedding '{embedding_id}' erfolgreich in ChromaDB gespeichert.")

    
    result = collection.get(
        ids=[embedding_id]
    )
    print(result)
    
    results = collection.get(
        ids=[embedding_id], 
        include=['embeddings']
    )
    print("saved Embedding:")
    print(results)

    # Beispiel: Abfragen des Embeddings
    results = collection.query(
        query_embeddings=[embedding_list], # Query mit dem gleichen Embedding
        n_results=1,
        include=['embeddings']
    )

    print(results)

    daniel_seperate_segments_embeddings = list()
    #Seperate Richard Embeddings: 
    for i in range(0, len(daniel_audio_segments), 1):

        output_filename_list = f"saved_audio_segment_daniel_{i}.wav"        
        # Export the audio segment to a WAV file
        daniel_audio_segments[i].export(output_filename_list, format="wav")

    for i in range(0, len(daniel_audio_segments), 1):
        
        if daniel_audio_segments[i].channels != 1:
            daniel_audio_segments[i] = daniel_audio_segments[i].set_channels(1)
                        
        if daniel_audio_segments[i].frame_rate != TARGET_SAMPLE_RATE:
            daniel_audio_segments[i] = daniel_audio_segments[i].set_frame_rate(TARGET_SAMPLE_RATE)

                        # Create a temporary file to save the audio segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_audio_path = tmpfile.name
            daniel_audio_segments[i].export(temp_audio_path, format="wav")
                        
            try:
                            # Get the embedding using the temporary file path
                embedding = speaker_model.get_embedding(temp_audio_path)
                            
                            # The embedding returned by get_embedding is usually already a PyTorch tensor.
                            # You can check its shape if you want:
                            # print(f"  Embedding shape: {embedding.shape}")

                daniel_seperate_segments_embeddings.append(embedding.cpu().squeeze().tolist())
            except Exception as e:
                continue
            finally:
                            # Clean up the temporary file
                os.remove(temp_audio_path)
    
    for i in range(0, len(daniel_seperate_segments_embeddings), 1): 

        print(f"daniel_seperate_segments_embeddings{i}")
        print(daniel_seperate_segments_embeddings[i])
        embedding_id = f"daniel_seperate_segments_embeddings{i}" # Eindeutige ID
        metadata = {
            "speaker_id": "Daniel_Speaker_Embedding",
            #"audio_file": "my_meeting.wav",
            #"start_time_ms": 12345,
            #"end_time_ms": 15678
        }

        # Embedding hinzufügen
        collection.add(
            embeddings=daniel_seperate_segments_embeddings[i],  # Muss eine Liste von Embeddings sein, auch wenn es nur eins ist
            metadatas=[metadata],         # Liste von Metadaten, passend zu Embeddings
            ids=[embedding_id]            # Liste von IDs, passend zu Embeddings
        )

        print(f"Embedding '{embedding_id}' erfolgreich in ChromaDB gespeichert.")
    
    for i in range(0, len(daniel_seperate_segments_embeddings), 1):

        embedding_id = f"daniel_seperate_segments_embeddings{i}"
        results = collection.get(
            
            ids=[embedding_id], 
            include=['embeddings']
        )

        print("GET RESULT FROM CHROMADB")
        print(results)

"""
[[['0.240 11.520 speaker_0', '11.760 14.800 speaker_0', '29.600 37.840 speaker_0', '38.320 40.320 speaker_0', '40.400 42.800 speaker_0', '44.160 44.720 speaker_0', '45.280 45.760 speaker_0', '46.080 54.480 speaker_0', '55.200 57.760 speaker_0', '58.000 61.440 speaker_0', '62.640 69.520 speaker_0', '69.760 69.840 speaker_0', '69.920 79.360 speaker_0', '79.520 107.360 speaker_0', '107.440 136.560 speaker_0', '137.120 166.070 speaker_0', '17.920 19.200 speaker_1', '20.400 24.720 speaker_1', '25.760 27.200 speaker_1', '61.360 62.640 speaker_2']], [['0.000 23.840 speaker_0', '29.920 32.000 speaker_0', '33.360 34.720 speaker_0', '36.480 37.360 speaker_0', '41.440 50.880 speaker_0', '51.760 53.120 speaker_0', '53.680 54.640 speaker_0', '55.680 58.000 speaker_0', '58.160 60.240 speaker_0', '60.880 61.120 speaker_0', '64.400 68.400 speaker_0', '73.840 75.600 speaker_0', '76.480 76.560 speaker_0', '77.120 77.520 speaker_0', '78.000 78.160 speaker_0', '78.480 78.640 speaker_0', '78.720 79.920 speaker_0', '80.480 86.400 speaker_0', '86.560 88.400 speaker_0', '89.120 90.720 speaker_0', '96.000 99.120 speaker_0', '100.160 100.880 speaker_0', '101.280 109.040 speaker_0', '109.920 112.240 speaker_0', '117.120 123.600 speaker_0', '124.080 126.640 speaker_0', '127.600 128.400 speaker_0', '128.720 133.120 speaker_0', '133.920 137.200 speaker_0', '143.360 143.440 speaker_0', '144.800 145.040 speaker_0', '149.120 149.360 speaker_0', '150.880 155.680 speaker_0', '156.640 162.000 speaker_0', '164.000 166.070 speaker_0', '24.160 29.600 speaker_1', '31.920 32.160 speaker_1', '32.800 32.880 speaker_1', '32.960 33.040 speaker_1', '60.720 61.120 speaker_2', '78.400 78.800 speaker_2', '115.680 115.760 speaker_2', '137.120 137.200 speaker_2', '37.760 41.200 speaker_3', '60.240 64.160 speaker_3', '68.640 72.640 speaker_3', '91.200 95.440 speaker_3', '112.720 117.200 speaker_3', '137.120 144.800 speaker_3', '145.040 148.720 speaker_3', '149.280 150.800 speaker_3']], [['0.000 2.640 speaker_0', '2.800 4.480 speaker_0', '5.520 15.200 speaker_0', '15.760 26.240 speaker_0', '26.320 28.560 speaker_0', '28.720 31.360 speaker_0', '32.320 34.880 speaker_0', '35.280 42.880 speaker_0', '43.440 48.560 speaker_0', '49.440 55.520 speaker_0', '66.400 73.360 speaker_0', '73.760 76.320 speaker_0', '76.800 81.200 speaker_0', '81.600 84.320 speaker_0', '84.960 88.320 speaker_0', '88.560 95.360 speaker_0', '95.840 99.040 speaker_0', '103.280 108.960 speaker_0', '111.680 113.680 speaker_0', '115.680 117.360 speaker_0', '117.680 120.160 speaker_0', '120.720 124.640 speaker_0', '125.280 127.120 speaker_0', '127.760 146.800 speaker_0', '150.800 158.560 speaker_0', '159.520 166.070 speaker_0', '55.760 65.600 speaker_1', '147.600 150.720 speaker_1', '99.440 100.640 speaker_2', '101.280 102.800 speaker_2', '109.040 111.360 speaker_2']], [['0.000 3.440 speaker_0', '4.320 4.640 speaker_0', '5.680 6.000 speaker_0', '6.320 6.800 speaker_0', '14.880 24.720 speaker_0', '24.960 26.400 speaker_0', '27.040 27.680 speaker_0', '28.480 30.160 speaker_0', '30.720 36.880 speaker_0', '37.520 41.680 speaker_0', '41.840 53.520 speaker_0', '53.680 63.200 speaker_0', '63.920 65.120 speaker_0', '65.440 69.360 speaker_0', '69.840 70.960 speaker_0', '71.040 79.040 speaker_0', '79.200 92.000 speaker_0', '92.960 97.440 speaker_0', '98.160 105.200 speaker_0', '105.440 111.600 speaker_0', '112.240 112.640 speaker_0', '113.120 117.600 speaker_0', '118.160 125.520 speaker_0', '127.680 129.120 speaker_0', '130.000 134.240 speaker_0', '136.000 140.720 speaker_0', '141.200 159.600 speaker_0', '159.840 163.280 speaker_0', '163.440 166.070 speaker_0', '4.400 6.320 speaker_1', '6.640 15.120 speaker_1', '126.080 128.560 speaker_1']], [['0.000 3.200 speaker_0', '3.280 3.360 speaker_0', '3.920 15.200 speaker_0', '15.520 22.880 speaker_0', '23.120 26.720 speaker_0', '26.960 39.760 speaker_0', '40.480 43.200 speaker_0', '43.680 45.920 speaker_0', '46.000 55.760 speaker_0', '55.840 56.320 speaker_0', '56.720 62.880 speaker_0', '66.080 75.360 speaker_0', '75.520 83.200 speaker_0', '83.680 84.800 speaker_0', '85.280 91.760 speaker_0', '98.720 106.240 speaker_0', '106.880 113.520 speaker_0', '113.680 125.920 speaker_0', '126.720 130.240 speaker_0', '130.880 137.360 speaker_0', '137.840 149.520 speaker_0', '150.000 155.680 speaker_0', '155.840 160.320 speaker_0', '161.200 166.070 speaker_0', '63.200 65.520 speaker_1', '92.160 98.400 speaker_1']], [['0.000 7.760 speaker_0', '8.560 12.240 speaker_0', '12.880 17.120 speaker_0', '17.200 23.920 speaker_0', '24.640 24.720 speaker_0', '25.040 26.320 speaker_0', '26.880 32.080 speaker_0', '32.960 40.240 speaker_0', '40.320 46.720 speaker_0', '47.760 48.160 speaker_0', '48.240 48.960 speaker_0', '50.000 60.880 speaker_0', '61.680 73.600 speaker_0', '74.480 77.360 speaker_0', '77.840 80.000 speaker_0', '80.640 84.000 speaker_0', '85.200 86.880 speaker_0', '88.320 88.960 speaker_0', '90.400 90.720 speaker_0', '91.360 92.800 speaker_0', '93.280 94.960 speaker_0', '107.040 107.440 speaker_0', '111.840 112.480 speaker_0', '116.880 118.960 speaker_0', '120.480 121.040 speaker_0', '125.040 125.120 speaker_0', '125.280 125.840 speaker_0', '129.440 130.400 speaker_0', '131.760 132.400 speaker_0', '133.120 134.720 speaker_0', '135.440 137.440 speaker_0', '141.600 142.000 speaker_0', '143.840 148.720 speaker_0', '87.600 88.320 speaker_1', '88.960 90.320 speaker_1', '96.320 100.560 speaker_1', '101.360 104.480 speaker_1', '104.960 106.960 speaker_1', '107.680 116.800 speaker_1', '120.000 120.480 speaker_1', '121.040 122.080 speaker_1', '122.880 126.560 speaker_1', '127.040 128.800 speaker_1', '129.200 129.680 speaker_1', '130.240 131.440 speaker_1', '132.240 134.160 speaker_1', '135.600 135.920 speaker_1', '138.960 139.360 speaker_1', '140.160 141.600 speaker_1', '142.240 143.280 speaker_1', '149.280 151.760 speaker_1', '152.160 153.760 speaker_1', '154.480 155.200 speaker_1', '158.080 159.280 speaker_2', '160.640 164.960 speaker_2', '166.000 166.070 speaker_2']], [['0.000 1.520 speaker_0']]]
"""


def richard_daniel_richard_diarization_durations(shortened_audios_for_diarization, diarizations)->tuple:

        import collections

        # Ihre bereitgestellten Daten
        #speaker_data = [[['0.240 4.320 speaker_0', '11.760 14.800 speaker_0', '36.720 37.680 speaker_0', '41.840 44.800 speaker_0', '45.120 47.600 speaker_0', '48.240 50.560 speaker_0', '51.200 57.440 speaker_0', '58.080 60.000 speaker_0', '60.480 64.480 speaker_0', '65.040 66.960 speaker_0', '70.240 70.320 speaker_0', '71.840 75.360 speaker_0', '76.640 77.440 speaker_0', '77.920 86.080 speaker_0', '86.640 87.440 speaker_0', '87.520 88.320 speaker_0', '88.720 90.000 speaker_0', '90.480 92.160 speaker_0', '92.640 93.520 speaker_0', '94.560 96.160 speaker_0', '96.720 97.680 speaker_0', '97.920 100.960 speaker_0', '101.440 103.120 speaker_0', '104.720 105.200 speaker_0', '106.160 110.080 speaker_0', '110.480 113.440 speaker_0', '113.520 115.920 speaker_0', '119.120 122.240 speaker_0', '122.960 126.080 speaker_0', '126.640 126.800 speaker_0', '126.880 127.600 speaker_0', '128.080 135.200 speaker_0', '135.680 138.800 speaker_0', '140.000 140.880 speaker_0', '141.040 143.520 speaker_0', '144.560 147.280 speaker_0', '155.920 156.320 speaker_0', '157.120 157.920 speaker_0', '163.200 163.680 speaker_0', '164.880 166.070 speaker_0', '4.560 11.520 speaker_1', '29.760 36.560 speaker_1', '38.000 41.440 speaker_1', '67.200 70.000 speaker_1', '74.240 74.960 speaker_1', '75.280 75.920 speaker_1', '86.560 88.720 speaker_1', '93.440 94.640 speaker_1', '96.080 96.320 speaker_1', '105.120 106.240 speaker_1', '116.640 119.040 speaker_1', '119.920 120.320 speaker_1', '147.920 155.760 speaker_1', '158.880 163.120 speaker_1', '163.760 164.880 speaker_1', '17.920 19.200 speaker_2', '20.400 24.800 speaker_2', '25.760 27.200 speaker_2']], [['0.000 2.000 speaker_0', '2.560 18.320 speaker_0', '18.720 30.720 speaker_0', '31.840 32.800 speaker_0', '37.760 40.720 speaker_0', '40.960 41.040 speaker_0', '41.600 42.000 speaker_0', '47.120 47.920 speaker_0', '48.800 50.640 speaker_0', '51.360 54.320 speaker_0', '55.040 88.480 speaker_0', '88.960 90.080 speaker_0', '93.440 94.400 speaker_0', '95.840 104.640 speaker_0', '105.360 118.640 speaker_0', '118.800 121.920 speaker_0', '126.080 126.640 speaker_0', '133.120 138.480 speaker_0', '138.880 155.920 speaker_0', '156.640 159.360 speaker_0', '160.640 160.880 speaker_0', '162.800 166.070 speaker_0', '33.840 37.360 speaker_1', '46.000 48.320 speaker_1', '90.560 94.400 speaker_1', '120.800 126.160 speaker_1', '126.640 133.120 speaker_1', '159.760 162.800 speaker_1']], [['0.000 16.560 speaker_0', '17.520 19.200 speaker_0', '19.600 22.320 speaker_0', '22.560 37.440 speaker_0', '38.080 50.160 speaker_0', '50.880 70.560 speaker_0', '70.960 73.840 speaker_0', '78.880 81.760 speaker_0', '88.080 94.800 speaker_0', '95.840 98.640 speaker_0', '99.280 108.800 speaker_0', '109.280 115.360 speaker_0', '116.000 129.520 speaker_0', '130.560 132.160 speaker_0', '132.800 139.040 speaker_0', '139.520 146.480 speaker_0', '146.960 152.000 speaker_0', '152.560 166.070 speaker_0', '74.240 78.960 speaker_1', '82.160 87.440 speaker_1']], [['0.000 1.760 speaker_0', '2.480 22.000 speaker_0', '26.400 27.440 speaker_0', '28.240 28.720 speaker_0', '29.760 30.960 speaker_0', '40.720 52.160 speaker_0', '52.800 53.840 speaker_0', '54.400 66.800 speaker_0', '67.360 76.480 speaker_0', '77.040 91.840 speaker_0', '92.720 95.200 speaker_0', '95.680 98.240 speaker_0', '98.880 117.680 speaker_0', '121.680 122.160 speaker_0', '142.640 143.120 speaker_0', '143.680 144.320 speaker_0', '145.520 147.120 speaker_0', '148.640 149.040 speaker_0', '149.120 152.400 speaker_0', '155.520 155.840 speaker_0', '160.080 160.640 speaker_0', '164.320 166.070 speaker_0', '22.480 25.680 speaker_1', '32.720 41.680 speaker_1', '118.000 121.840 speaker_1', '122.160 123.920 speaker_1', '125.280 130.720 speaker_1', '131.120 142.240 speaker_1', '144.640 145.520 speaker_1', '147.200 150.000 speaker_1', '152.400 159.520 speaker_1', '160.560 162.960 speaker_1', '163.520 164.080 speaker_1']], [['0.000 4.480 speaker_0', '5.040 6.480 speaker_0', '7.520 9.760 speaker_0', '10.320 11.440 speaker_0', '12.320 12.720 speaker_0', '13.520 14.400 speaker_0', '14.800 16.000 speaker_0', '22.000 25.200 speaker_0', '25.440 28.240 speaker_0', '29.280 30.240 speaker_0', '30.880 31.920 speaker_0', '32.640 36.160 speaker_0', '52.880 55.920 speaker_0', '63.760 70.880 speaker_0', '71.360 75.920 speaker_0', '76.480 76.560 speaker_0', '79.120 80.800 speaker_0', '81.760 89.360 speaker_0', '90.800 93.600 speaker_0', '94.000 94.560 speaker_0', '9.840 10.160 speaker_1', '11.760 12.240 speaker_1', '12.880 13.360 speaker_1', '14.320 14.800 speaker_1', '16.000 18.800 speaker_1', '18.960 23.120 speaker_1', '25.040 25.440 speaker_1', '28.240 29.120 speaker_1', '36.960 39.440 speaker_1', '40.080 41.920 speaker_1', '42.560 45.200 speaker_1', '45.600 48.160 speaker_1', '49.280 50.720 speaker_1', '51.200 52.960 speaker_1', '56.400 58.640 speaker_1', '59.680 61.600 speaker_1', '61.760 63.520 speaker_1', '76.320 78.880 speaker_1', '93.520 94.000 speaker_1', '94.960 96.080 speaker_1', '97.360 98.960 speaker_1', '99.680 100.320 speaker_1', '103.440 104.640 speaker_2', '106.000 110.320 speaker_2', '111.360 112.720 speaker_2']]]

        # Verwenden Sie ein defaultdict, um die Zeiten pro Sprecher zu sammeln
        # Ein defaultdict initialisiert den Wert (hier 0.0) automatisch, wenn ein neuer Schlüssel (Sprecher) hinzugefügt wird.
        speaker_durations = collections.defaultdict(float)

        #Zuerst mit Embedding die Speaker zuordnen und dann ein array [[[]]] erstellen damit
        #Dafür die wav files iterieren und slicen, um sprecherzuordnung durchzuführen
        # Ziel Richard und Daniel zuordnen für jedes Segment, damit 
        #[['3.040 4.480 speaker_0', '5.520 10.000 speaker_0', '10.880 12.400 speaker_0', '15.040 20.640 speaker_1', '23.200 28.080 speaker_1', '28.240 32.080 speaker_1', '41.600 42.000 speaker_1', '57.760 58.240 speaker_1', '21.120 22.960 speaker_2', '32.800 33.680 speaker_2', '34.640 36.640 speaker_2', '36.720 39.120 speaker_2', '40.480 41.040 speaker_2', '42.400 50.800 speaker_2', '51.280 57.840 speaker_2', '58.240 65.920 speaker_2', '66.000 75.680 speaker_2', '75.840 132.880 speaker_2', '133.440 166.070 speaker_2']]
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Definiere den Ordnernamen für die ChromaDB-Daten
        db_directory = os.path.join(script_dir, "chroma_db_speakers")
        client = chromadb.PersistentClient(path=db_directory)
        collection_name = "speaker_embeddings"
        collection = client.get_or_create_collection(name=collection_name)


        #speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        
        #shortened_audios_for_diarization_index = 0

        unique_speakers = set()

        for sublist_1 in diarizations:
            for sublist_2 in sublist_1:
                for item_string in sublist_2:
                    parts = item_string.rsplit(' ', 1)
                    if len(parts) == 2:
                        speaker_name = parts[1]
                        unique_speakers.add(speaker_name) # Füge den Sprecher zum Set hinzu

        # Optional: Die Anzahl der eindeutigen Sprecher ausgeben
        print(f"Es gibt {len(unique_speakers)} verschiedene Sprecher in den Daten.")
        print(f"Die eindeutigen Sprecher sind: {unique_speakers}\n")

        speaker_times = dict()
        # Durchlaufen der verschachtelten Dict
        # speaker_times['segment'] = dict()
        # speaker_times['segment']

        daniel_time: float = 0.0
        richard_time: float = 0.0

        TARGET_SAMPLE_RATE = 16000 # Most ASR/Speaker Recognition models expect 16kHz mono
        tempfile_count: int = 0
        diaCount= 0

        for sublist_level1 in diarizations:
            for sublist_level2 in sublist_level1:
                current_segment = shortened_audios_for_diarization[diaCount]
                audio = AudioSegment.from_wav(current_segment)

                diaCount+=1
                speaker_times[current_segment] = dict()
                for entry_string in sublist_level2:
                    # Teilen des Strings in Startzeit, Endzeit und Sprecher-ID
                    parts = entry_string.split()
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    speaker_id = str(parts[2])

                    #speaker_times[]
                    # Prüfe, ob der Hauptschlüssel existiert, ansonsten initialisiere ihn
                    if speaker_id not in speaker_times[current_segment]:
                        speaker_times[current_segment][speaker_id] = {} # Initialisiere den Wert als leeres Dictionary

                    if 'start_time' not in speaker_times[current_segment][speaker_id]:
                        speaker_times[current_segment][speaker_id]['start_time'] = list()
                    
                    if 'end_time' not in speaker_times[current_segment][speaker_id]:
                        speaker_times[current_segment][speaker_id]['end_time'] = list()

                    #sicher Elemente anhängen
                    speaker_times[current_segment][speaker_id]['start_time'].append(start_time)
                    speaker_times[current_segment][speaker_id]['end_time'].append(end_time)

            #speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

        print(speaker_times)

        no_match_counter = 0

        for current_segment in speaker_times.keys():
            audio = AudioSegment.from_wav(current_segment)
            for speaker_id, time_data_dict in speaker_times[current_segment].items():
                if len(time_data_dict['start_time']) != len(time_data_dict['end_time']):
                    print(f"Warnung: Unstimmige Listenlängen bei {current_segment}, {speaker_id}")
                    continue

                current_audio = None
                overall_time = 0
                for start_time_sec, end_time_sec in zip(time_data_dict['start_time'], time_data_dict['end_time']):
                    start_time_ms = int(start_time_sec * 1000)
                    end_time_ms = int(end_time_sec * 1000)
                    overall_time += abs(end_time_ms - start_time_ms) / 1000
                    if current_audio is None:
                        current_audio = audio[start_time_ms:end_time_ms]
                    else:
                        current_audio += audio[start_time_ms:end_time_ms]

                if current_audio is not None:
                    # Erst alle Transformationen anwenden
                    if current_audio.channels != 1:
                        current_audio = current_audio.set_channels(1)
                    if current_audio.frame_rate != TARGET_SAMPLE_RATE:
                        current_audio = current_audio.set_frame_rate(TARGET_SAMPLE_RATE)

                    # Exportiere (optional) das Audio für den Sprecher
                    output_filename = f"speaker_export_{speaker_id}_{current_segment}.wav"
                    current_audio.export(output_filename, format="wav")
                    print("overall_time", overall_time)

                    # Exportiere in eine temporäre Datei für das Embedding
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        temp_audio_path = tmpfile.name
                        current_audio.export(temp_audio_path, format="wav")

                        try:
                            embedding = speaker_model.get_embedding(temp_audio_path)
                            embedding = embedding.cpu().squeeze().tolist()
                            results = collection.get(include=['embeddings'])
                            all_ids = results['ids']
                            all_chroma_embeddings = np.array(results['embeddings'])

                            print(f"Erfolgreich {len(all_chroma_embeddings)} Embeddings aus der DB abgerufen.")

                            query_embedding_np = np.array(embedding).reshape(1, -1)
                            highest_similarity = -1.0
                            closest_id = None

                            print("\nVergleiche das generierte Embedding mit allen DB-Embeddings...")
                            for i in range(len(all_chroma_embeddings)):
                                db_embedding_id = all_ids[i]
                                db_embedding_vector = all_chroma_embeddings[i].reshape(1, -1)
                                similarity = cosine_similarity(query_embedding_np, db_embedding_vector).item()
                                if similarity > highest_similarity:
                                    highest_similarity = similarity
                                    closest_id = db_embedding_id

                            print("\n--- Ergebnis ---")
                            if closest_id:
                                print(f"Prüfe ID: {closest_id}")
                                regex_pattern_daniel = r"^daniel_seperate_segments_embeddings\d*$"
                                regex_pattern_richard_or_richeard = r"^(richard|richeard)_seperate_segments_embeddings\d*$"
                                if re.match(regex_pattern_daniel, closest_id, re.IGNORECASE):
                                    daniel_time += overall_time
                                    print(f"Die ID '{closest_id}' passt zum Daniel-Muster (case-insensitive).")
                                elif re.match(regex_pattern_richard_or_richeard, closest_id, re.IGNORECASE):
                                    richard_time += overall_time
                                    print(f"Die ID '{closest_id}' passt zum Richard/Richeard-Muster (case-insensitive).")
                                else:
                                    print(f"Die ID '{closest_id}' passt NICHT zu den definierten Mustern (Daniel oder Richard/Richeard).")
                                    no_match_counter += 1
                                print(f"Die ID mit der höchsten Ähnlichkeit ist: '{closest_id}'")
                                print(f"Kosinus-Ähnlichkeit: {highest_similarity:.4f}")

                            print(f"Daniel's Gesamtzeit: {daniel_time}")
                            print(f"Richard's Gesamtzeit: {richard_time}")
                        except Exception as e:
                            print(f"Fehler bei der Embeddings-Verarbeitung oder Datenbankabfrage: {e}")
                        finally:
                            os.remove(temp_audio_path)
                            print("removed")
        # --- Der angefragte try-except-Block für die Endausgabe ---
        try:
            print("\n--- Abschließende Ergebnisse ---")
            print(f"Anzahl der nicht zugeordneten Segmente: {no_match_counter}")
            print(f"Daniel's Gesamtzeit: {daniel_time:.2f}s")
            print(f"Richard's Gesamtzeit: {richard_time:.2f}s")
            return daniel_time, richard_time
        except Exception as e:
            print(f"Ein unerwarteter Fehler ist bei der Ausgabe oder Rückgabe aufgetreten: {e}")
            # Optional: Eine Fehlerwert oder None zurückgeben, um auf den Fehler hinzuweisen
            return 0.0, 0.0 # Oder raise e, je nach gewünschtem Fehlerverhalten

if __name__ == "__main__":

    #split_audio_for_speaker_embeddings()
    #split_audio_for_speaker_embeddings_Daniel()
    os.chdir("..")
    os.chdir("audioData")
    os.chdir("shortened_audios_wav")
    richard_daniel_richard_diarization_durations(["GAG04_segment_0.wav", "GAG04_segment_166000.wav", "GAG04_segment_332000.wav", "GAG04_segment_498000.wav", "GAG04_segment_664000.wav", "GAG04_segment_830000.wav", "GAG04_segment_996000.wav"], [[['0.240 11.520 speaker_0', '11.760 14.800 speaker_0', '29.600 37.840 speaker_0', '38.320 40.320 speaker_0', '40.400 42.800 speaker_0', '44.160 44.720 speaker_0', '45.280 45.760 speaker_0', '46.080 54.480 speaker_0', '55.200 57.760 speaker_0', '58.000 61.440 speaker_0', '62.640 69.520 speaker_0', '69.760 69.840 speaker_0', '69.920 79.360 speaker_0', '79.520 107.360 speaker_0', '107.440 136.560 speaker_0', '137.120 166.070 speaker_0', '17.920 19.200 speaker_1', '20.400 24.720 speaker_1', '25.760 27.200 speaker_1', '61.360 62.640 speaker_2']], [['0.000 23.840 speaker_0', '29.920 32.000 speaker_0', '33.360 34.720 speaker_0', '36.480 37.360 speaker_0', '41.440 50.880 speaker_0', '51.760 53.120 speaker_0', '53.680 54.640 speaker_0', '55.680 58.000 speaker_0', '58.160 60.240 speaker_0', '60.880 61.120 speaker_0', '64.400 68.400 speaker_0', '73.840 75.600 speaker_0', '76.480 76.560 speaker_0', '77.120 77.520 speaker_0', '78.000 78.160 speaker_0', '78.480 78.640 speaker_0', '78.720 79.920 speaker_0', '80.480 86.400 speaker_0', '86.560 88.400 speaker_0', '89.120 90.720 speaker_0', '96.000 99.120 speaker_0', '100.160 100.880 speaker_0', '101.280 109.040 speaker_0', '109.920 112.240 speaker_0', '117.120 123.600 speaker_0', '124.080 126.640 speaker_0', '127.600 128.400 speaker_0', '128.720 133.120 speaker_0', '133.920 137.200 speaker_0', '143.360 143.440 speaker_0', '144.800 145.040 speaker_0', '149.120 149.360 speaker_0', '150.880 155.680 speaker_0', '156.640 162.000 speaker_0', '164.000 166.070 speaker_0', '24.160 29.600 speaker_1', '31.920 32.160 speaker_1', '32.800 32.880 speaker_1', '32.960 33.040 speaker_1', '60.720 61.120 speaker_2', '78.400 78.800 speaker_2', '115.680 115.760 speaker_2', '137.120 137.200 speaker_2', '37.760 41.200 speaker_3', '60.240 64.160 speaker_3', '68.640 72.640 speaker_3', '91.200 95.440 speaker_3', '112.720 117.200 speaker_3', '137.120 144.800 speaker_3', '145.040 148.720 speaker_3', '149.280 150.800 speaker_3']], [['0.000 2.640 speaker_0', '2.800 4.480 speaker_0', '5.520 15.200 speaker_0', '15.760 26.240 speaker_0', '26.320 28.560 speaker_0', '28.720 31.360 speaker_0', '32.320 34.880 speaker_0', '35.280 42.880 speaker_0', '43.440 48.560 speaker_0', '49.440 55.520 speaker_0', '66.400 73.360 speaker_0', '73.760 76.320 speaker_0', '76.800 81.200 speaker_0', '81.600 84.320 speaker_0', '84.960 88.320 speaker_0', '88.560 95.360 speaker_0', '95.840 99.040 speaker_0', '103.280 108.960 speaker_0', '111.680 113.680 speaker_0', '115.680 117.360 speaker_0', '117.680 120.160 speaker_0', '120.720 124.640 speaker_0', '125.280 127.120 speaker_0', '127.760 146.800 speaker_0', '150.800 158.560 speaker_0', '159.520 166.070 speaker_0', '55.760 65.600 speaker_1', '147.600 150.720 speaker_1', '99.440 100.640 speaker_2', '101.280 102.800 speaker_2', '109.040 111.360 speaker_2']], [['0.000 3.440 speaker_0', '4.320 4.640 speaker_0', '5.680 6.000 speaker_0', '6.320 6.800 speaker_0', '14.880 24.720 speaker_0', '24.960 26.400 speaker_0', '27.040 27.680 speaker_0', '28.480 30.160 speaker_0', '30.720 36.880 speaker_0', '37.520 41.680 speaker_0', '41.840 53.520 speaker_0', '53.680 63.200 speaker_0', '63.920 65.120 speaker_0', '65.440 69.360 speaker_0', '69.840 70.960 speaker_0', '71.040 79.040 speaker_0', '79.200 92.000 speaker_0', '92.960 97.440 speaker_0', '98.160 105.200 speaker_0', '105.440 111.600 speaker_0', '112.240 112.640 speaker_0', '113.120 117.600 speaker_0', '118.160 125.520 speaker_0', '127.680 129.120 speaker_0', '130.000 134.240 speaker_0', '136.000 140.720 speaker_0', '141.200 159.600 speaker_0', '159.840 163.280 speaker_0', '163.440 166.070 speaker_0', '4.400 6.320 speaker_1', '6.640 15.120 speaker_1', '126.080 128.560 speaker_1']], [['0.000 3.200 speaker_0', '3.280 3.360 speaker_0', '3.920 15.200 speaker_0', '15.520 22.880 speaker_0', '23.120 26.720 speaker_0', '26.960 39.760 speaker_0', '40.480 43.200 speaker_0', '43.680 45.920 speaker_0', '46.000 55.760 speaker_0', '55.840 56.320 speaker_0', '56.720 62.880 speaker_0', '66.080 75.360 speaker_0', '75.520 83.200 speaker_0', '83.680 84.800 speaker_0', '85.280 91.760 speaker_0', '98.720 106.240 speaker_0', '106.880 113.520 speaker_0', '113.680 125.920 speaker_0', '126.720 130.240 speaker_0', '130.880 137.360 speaker_0', '137.840 149.520 speaker_0', '150.000 155.680 speaker_0', '155.840 160.320 speaker_0', '161.200 166.070 speaker_0', '63.200 65.520 speaker_1', '92.160 98.400 speaker_1']], [['0.000 7.760 speaker_0', '8.560 12.240 speaker_0', '12.880 17.120 speaker_0', '17.200 23.920 speaker_0', '24.640 24.720 speaker_0', '25.040 26.320 speaker_0', '26.880 32.080 speaker_0', '32.960 40.240 speaker_0', '40.320 46.720 speaker_0', '47.760 48.160 speaker_0', '48.240 48.960 speaker_0', '50.000 60.880 speaker_0', '61.680 73.600 speaker_0', '74.480 77.360 speaker_0', '77.840 80.000 speaker_0', '80.640 84.000 speaker_0', '85.200 86.880 speaker_0', '88.320 88.960 speaker_0', '90.400 90.720 speaker_0', '91.360 92.800 speaker_0', '93.280 94.960 speaker_0', '107.040 107.440 speaker_0', '111.840 112.480 speaker_0', '116.880 118.960 speaker_0', '120.480 121.040 speaker_0', '125.040 125.120 speaker_0', '125.280 125.840 speaker_0', '129.440 130.400 speaker_0', '131.760 132.400 speaker_0', '133.120 134.720 speaker_0', '135.440 137.440 speaker_0', '141.600 142.000 speaker_0', '143.840 148.720 speaker_0', '87.600 88.320 speaker_1', '88.960 90.320 speaker_1', '96.320 100.560 speaker_1', '101.360 104.480 speaker_1', '104.960 106.960 speaker_1', '107.680 116.800 speaker_1', '120.000 120.480 speaker_1', '121.040 122.080 speaker_1', '122.880 126.560 speaker_1', '127.040 128.800 speaker_1', '129.200 129.680 speaker_1', '130.240 131.440 speaker_1', '132.240 134.160 speaker_1', '135.600 135.920 speaker_1', '138.960 139.360 speaker_1', '140.160 141.600 speaker_1', '142.240 143.280 speaker_1', '149.280 151.760 speaker_1', '152.160 153.760 speaker_1', '154.480 155.200 speaker_1', '158.080 159.280 speaker_2', '160.640 164.960 speaker_2', '166.000 166.070 speaker_2']], [['0.000 1.520 speaker_0']]])