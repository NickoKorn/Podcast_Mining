import torch
from audioPreprocess import * 
from whispersModel import *
from dataScraping import * 
import os
from speakerDiarizer import *
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import tempfile
from gui import get_whisper_instance
from speakerEmbeddings import richard_daniel_richard_diarization_durations
from mongodb_connector import *
import re
import gc
from textEmbeddings import *

"""
Facade Class for User input that controlls the dataflows, what is saved/deelted in the database or processed like with WhisperModel
"""

def extract_timestamps_and_texts(data, segment_lengths:list)->list:
    """
    Extrahiert Timestamps und zugehörige Texte aus der gegebenen Datenstruktur.

    Args:
        data (list or dict): Die Datenstruktur, die die Texte und Chunks enthält.

    Returns:
        list: Eine Liste von Dictionaries, wobei jedes Dictionary 'start_time',
              'end_time' und 'text' enthält.
    """
    extracted_data = []

    segment_index: int = -1

    current_start_timestamp = 0
    current_end_timestamp = 0
    # Prüfen, ob es eine Liste von Dictionaries ist (wie im Beispiel)
    if isinstance(data, list):
        for entry in data:
            
            segment_index+=1

            if 'chunks' in entry and isinstance(entry['chunks'], list):
                for chunk in entry['chunks']:
                    if 'timestamp' in chunk and 'text' in chunk:
                        start_time, end_time = chunk['timestamp']
                        if start_time != None:
                           current_start_timestamp = start_time
                        if end_time != None:
                           current_end_timestamp = end_time 
                        if end_time==None and start_time!=None:
                            extracted_data.append({
                                'start_time': start_time,
                                'end_time': abs(segment_lengths[segment_index]-start_time),
                                'text': chunk['text']
                            })
                        elif end_time!=None and start_time==None:
                            extracted_data.append({
                                'start_time': current_start_timestamp,
                                'end_time': end_time,
                                'text': chunk['text']
                            })
                        elif end_time!=None and start_time!=None:
                            extracted_data.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'text': chunk['text']
                            })
                        elif end_time==None and start_time==None:
                            extracted_data.append({
                                'start_time': current_start_timestamp,
                                'end_time': current_end_timestamp,
                                'text': chunk['text']
                            })
                        """
                        if end_time==None:
                            extracted_data.append({
                                'start_time': start_time,
                                'end_time': abs(segment_lengths[segment_index]-start_time),
                                'text': chunk['text']
                            })
                        else:
                            extracted_data.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'text': chunk['text']
                            })
                        """
    elif isinstance(data, dict) and 'chunks' in data and isinstance(data['chunks'], list):
        # Falls die Top-Ebene ein Dictionary ist und direkt die 'chunks' enthält
        for chunk in data['chunks']:

            segment_index+=1

            if 'timestamp' in chunk and 'text' in chunk:
                start_time, end_time = chunk['timestamp']
                if end_time==None:

                    extracted_data.append({
                        'start_time': start_time,
                        'end_time': abs(segment_lengths[segment_index]-start_time),
                        'text': chunk['text']
                    })
                else:
                    extracted_data.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': chunk['text']
                    })
    return extracted_data

def get_last_time_stamp_with_lenght_of_segment(shortened_audios_for_transcriptions:list)->list:

    lenghts_per_episode = list()
    for i in range(0, len(shortened_audios_for_transcriptions), 1):

        try:
            audio = AudioSegment.from_file(shortened_audios_for_transcriptions[i], format="mp3")
            duration_in_seconds = len(audio) / 1000.0  # pydub gibt Dauer in Millisekunden zurück
            lenghts_per_episode.append(duration_in_seconds)
        except Exception as e:
            print(f"Fehler beim Ermitteln der Dauer für {shortened_audios_for_transcriptions[i]}: {e}")
            return None
    print("lenghts_per_episode:")
    print(lenghts_per_episode)
    return lenghts_per_episode

def run_dynamic_pipeline(episodes_with_description: dict, episode_numbers: list, current_titels_and_summaries: dict, pipeline_steps: set):
        
        """
        Based on checked boxes: 
        Every Task need these parts in the pipeline: 
        pipeline_steps.add("Request")
                    pipeline_steps.add("Scrape")
                    pipeline_steps.add("Preprocess")

        So it will be executed, optinal with DB savings, regardless of the chosen task

        After that some if statemrents will check what further models are needed, like whisper and WhisperX
        Segmentation Embedding classification is special with ChromaDB
        """
        
        mongoDB = MongoDBManager()

        print("vor Loop")
        episode_documents = list()
        for i in range(0, len(episode_numbers), 1):


            shortened_audios_for_transcriptions = None
            daniel_time = None
            richard_time = None
            full_concatenated_text = None
            timeline_of_segments = None
            texts = None
            
            check_database_Data = mongoDB.check_episode_data_completeness(episode_numbers[i], episodes_with_description) 
            print(check_database_Data)

            episode_title = episodes_with_description.get(str(episode_numbers[i]-1), {}).get("title")
            print(episode_title)
            if not episode_title:
                print(f"Fehler: Titel für Episode {episode_numbers[i]} nicht gefunden.")
                return

            match = re.search(r'\d+', episode_title)
            zahl_als_int = None
            print("bis hier")

            if match:
                zahl_als_string = match.group(0)
                zahl_als_int = int(zahl_als_string)
                    #print(f"Die extrahierte Zahl (als String): {zahl_als_string}")
                    #print(f"Die extrahierte Zahl (als Integer): {zahl_als_int}")
                    #print(f"Typ der extrahierten Zahl: {type(zahl_als_int)}")
            else:
                print("Keine Zahl im String gefunden. Episode kann nicht gespeichert werden.")

            downloadChosenEpisode([episode_numbers[i]])
            #print("titels_and_summaries")
            #print(titels_and_summaries)
            
            shortened_audios_for_transcriptions, shortened_audios_for_diarization = audioPreprocessPipeline({str(episode_numbers[i] - 1): current_titels_and_summaries[str(episode_numbers[i] - 1)]})
            os.chdir("..")
            os.chdir("audioData")
            os.chdir("shortened_audios")

            # if os.path.exists("../../models/primeline"):

                #WhisperModell = 
            #else:
                
            segment_lengths: list = get_last_time_stamp_with_lenght_of_segment(shortened_audios_for_transcriptions)
            print(segment_lengths)

            if "Whisper_Pipe" in pipeline_steps and check_database_Data["full_text_transcript"] == True:
            
                full_concatenated_text = mongoDB.get_full_text_transcript(zahl_als_int)
                texts = mongoDB.get_whisper_texts(zahl_als_int)
                print(full_concatenated_text)

            elif "Whisper_Pipe" in pipeline_steps and check_database_Data["full_text_transcript"] == False:
                whisperModel = get_whisper_instance()
                texts  = list()

                print(os.getcwd())

                for j in range(0, len(shortened_audios_for_transcriptions), 1):

                    texts.append(whisperModel.transcripteText(shortened_audios_for_transcriptions[j]))

                print(texts)
                
                full_concatenated_text = ""

                # Durch jedes Haupt-Dictionary in der Liste 'data' iterieren
                for item_dict in texts:
                    # Zuerst den Text aus dem Hauptschlüssel 'text' hinzufügen
                    if 'text' in item_dict and isinstance(item_dict['text'], str):
                        full_concatenated_text += item_dict['text'] + " "

                # Abschließend alle überflüssigen Leerzeichen am Anfang und Ende entfernen
                full_concatenated_text = full_concatenated_text.strip()

            #print(full_concatenated_text)

            torch.cuda.empty_cache()
            #del whisperModel

            
            if "segment_classification" in pipeline_steps and check_database_Data["segment_timeline"] == True:

                timeline_of_segments = mongoDB.get_segment_timeline(zahl_als_int)
                print(timeline_of_segments)

            elif "segment_classification" in pipeline_steps and check_database_Data["segment_timeline"] == False:
                
                extract_timestamps_and_texts_list = extract_timestamps_and_texts(texts, segment_lengths)
                #print(extract_timestamps_and_texts_list)
                
                classified_similarites = classify_text_similarity_with_chromadb(extract_timestamps_and_texts_list)
                
                classified_similarites_list = list()
                #print(classified_similarites)
                for segment in classified_similarites:
                    new_entry = {
                        'Klasse': segment['classified_category'],
                        'Start-Zeit': segment['start_time'],
                        'Gesamt-Zeit': segment['Overall time'],
                        #'Original-Text': segment['original Segment']
                    }
                    classified_similarites_list.append(new_entry)

                # --- Start of the corrected timeline generation logic ---

                # Initialize classified_labels_index to 0 to start search from the beginning of the list
                classified_labels_index = 0
                # Define the classes you are interested in finding the first occurrence of
                segment_labels_to_find = ["Intro", "Werbung", "Feedback", "Inhalt", "Diskussion", "Feedbackhinweisblock", "Danksagung", "Outro", "Schlussgag"]

                timeline_of_segments = []
                found_classes = set() # Use a set for efficient lookup of already found classes

                print("Searching for the first occurrence of each specified class...")

                for segment_data in classified_similarites_list:
                    current_class = segment_data['Klasse']

                    if current_class in segment_labels_to_find and current_class not in found_classes:
                        current_segment_dict = {
                            'Start-Zeit': float(segment_data['Start-Zeit']),
                            'Gesamt-Zeit': float(segment_data['Gesamt-Zeit']),
                            'Klasse': segment_data['Klasse'],
                            # 'Original-Text': segment_data['Original-Text'] # Uncomment if needed
                        }
                        timeline_of_segments.append(current_segment_dict)
                        found_classes.add(current_class) # Mark this class as found
                        print(f"Found earliest: {current_segment_dict}")

                        # Optional: If you want to stop early once all desired classes are found
                        if len(found_classes) == len(segment_labels_to_find):
                            print("All desired classes found.")
                            break

                print("\n---")
                print("Timeline of First Occurrences (Chronological by Appearance):")
                # Sort the final list by 'Start-Zeit' to ensure absolute chronological order,
                # as 'classified_similarites_list' might not be perfectly sorted if 'Overall time' was used as key.
                # Assuming 'Start-Zeit' is a float for proper sorting.
                timeline_of_segments.sort(key=lambda x: x['Start-Zeit'])
                print(timeline_of_segments)
            #Ab hier Speaker Diarization
            os.chdir("..")
            os.chdir("shortened_audios_wav")

            if "SpeakerDiariazation" in pipeline_steps and check_database_Data["speaker_times"] == True:

                speaker_diarization = mongoDB.get_speaker_times(zahl_als_int)
                daniel_time = speaker_diarization["Daniel"]
                richard_time = speaker_diarization["Richard"]

                print(speaker_diarization)

            elif "segment_classification" in pipeline_steps and check_database_Data["segment_timeline"] == False:
            
                diarizations = list()
                for j in range(0, len(shortened_audios_for_diarization), 1):

                    embedding = nemo(shortened_audios_for_diarization[j])
                    #print("Segment" + str(i))
                    #print(embedding)
                    diarizations.append(embedding)
                    #print("Segment" + str(i) + " " + embedding)
                print(diarizations)

                #gc.collect()

                #shortened_audios_for_transcriptions.clear()
                
                daniel_time, richard_time = richard_daniel_richard_diarization_durations(shortened_audios_for_diarization, diarizations)
                print("funktioniert bis hier")
                shortened_audios_for_diarization.clear()

            os.chdir("..")
            os.chdir("..")
            os.chdir("src")
            print(timeline_of_segments)
            #print(richard_anteile)
            #print(daniel_anteile)

            #print("../audioData/"+"GAG"+str(episode_numbers[0])+".mp3")
            #print("../audioData/shortened_audios/"+str(shortened_audios_for_transcriptions[0]))

            print(episode_numbers)
            print(i)
            print(episode_numbers[i])

            #speaker_diarization = mongoDB.get_speaker_info(zahl_als_int)

            print("bis zur DB")
            episode_document = store_episode_data_in_mongodb(
                episodes_with_description=episodes_with_description,
                episode_numbers=[episode_numbers[i]],
                shortened_audios_for_transcriptions=shortened_audios_for_transcriptions[0],
                daniel_time=daniel_time,
                richard_time=richard_time,
                full_concatenated_text=full_concatenated_text,
                texts = texts,
                timeline_of_segments=timeline_of_segments,
                pipeline_steps = pipeline_steps
            )
            episode_documents.append(episode_document)
        return episode_documents
        
        """
        print(episode_numbers[0])
        #print(episodes_with_description)
        match = re.search(r'\d+', episodes_with_description[str(episode_numbers[0])]["title"])

        if match:
            # Wenn ein Match gefunden wird, extrahiert group(0) den gefundenen String
            zahl_als_string = match.group(0)
            zahl_als_int = int(zahl_als_string) # Optional: Umwandlung in Integer

            print(f"Die extrahierte Zahl (als String): {zahl_als_string}")
            print(f"Die extrahierte Zahl (als Integer): {zahl_als_int}")
            print(f"Typ der extrahierten Zahl: {type(zahl_als_int)}")
        else:
            print("Keine Zahl im String gefunden.")

                # 1. AudioPaths initialisieren
        audio_paths_instance = AudioPaths(
            full="../audioData/"+"GAG"+zahl_als_string+".mp3",
            split=["../audioData/shortened_audios/"+str(shortened_audios_for_transcriptions[0])]
        )

        # Um dies in MongoDB zu speichern, wandeln wir es in ein Dictionary um
        audio_paths_dict = {
            "full": audio_paths_instance.full,
            "split": audio_paths_instance.split
        }

        speaker_times = {
            "Daniel": daniel_time,
            "Richard": richard_time
        }

        hauptsprecher = max(daniel_time, richard_time)
        print("funktioniert bis hier AHuptsprecher ")

        # --- Dokument vorbereiten, das in MongoDB gespeichert wird ---

        episode_document = {
            "_id": zahl_als_int,  # Episodennummer als MongoDB _id verwenden
            "audio_paths": audio_paths_dict,
            "full_text_transcript": full_concatenated_text,
            "speaker_times": speaker_times,
            "segment_timeline": timeline_of_segments,
            "Geschichtenerzähler": hauptsprecher
        }

        print("funktioniert bis hier")

        # Eine Instanz deines MongoDBManagers erstellen
        db_manager = MongoDBManager()

        # Die Collection auswählen, in der die Episodendaten gespeichert werden sollen
        # Nennen wir sie 'episodes'
        episodes_collection = db_manager.get_collection("episodes")

        print("funktioniert bis hier episodes_collection")
        if episodes_collection:
            try:
                # Das Dokument in die Collection einfügen
                # update_one mit upsert=True ist hier gut, um ein Dokument einzufügen oder zu aktualisieren,
                # falls es bereits eine Episode mit dieser _id gibt.
                result = episodes_collection.update_one(
                    {"_id": episode_document["_id"]},
                    {"$set": episode_document},
                    upsert=True
                )
                if result.upserted_id:
                    print(f"Episode {episode_document['_id']} erfolgreich eingefügt.")
                elif result.modified_count > 0:
                    print(f"Episode {episode_document['_id']} erfolgreich aktualisiert.")
                else:
                    print(f"Episode {episode_document['_id']} existierte bereits, aber keine Änderungen vorgenommen.")

            except Exception as e:
                print(f"Fehler beim Speichern der Episode in MongoDB: {e}")
        else:
            print("Konnte keine Verbindung zur 'episodes' Collection herstellen.")

        if db_manager._db: # Prüfen, ob das DB-Objekt existiert
            try:
                # Liste alle Collection-Namen in der Datenbank auf
                collection_names = db_manager._db.list_collection_names()
                print(f"\nCollections in Datenbank '{db_manager.db_name}': {collection_names}")

                # Überprüfe die 'episodes' Collection
                if "episodes" in collection_names:
                    episode_count = episodes_collection.count_documents({})
                    print(f"Anzahl der Dokumente in 'episodes' Collection: {episode_count}")
                    # Optional: Die ersten paar Dokumente ausgeben
                    # for doc in episodes_collection.find().limit(3):
                    #     print(doc)
                else:
                    print("Die 'episodes' Collection existiert noch nicht (oder ist leer).")

            except Exception as e:
                print(f"Fehler beim Abfragen der Collections oder Dokumente: {e}")
        else:
            print("Keine Datenbankverbindung verfügbar, um Collections abzufragen.")
        # --- Ende des NEUEN Blocks ---

        # Verbindung schließen, wenn fertig
        db_manager.close_connection()
        """