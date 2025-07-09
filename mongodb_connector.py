import os
import re
from pymongo import MongoClient
from bson.objectid import ObjectId
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- Bereits vorhandene Klassen ---
class AudioPaths:
    def __init__(self, full: str, split: List[str]):
        self.full = full
        self.split = split

class MongoDBManager:
    """
    Verwaltet die MongoDB-Verbindung und Datenbankoperationen.
    """
    def __init__(self, host: str = "localhost", port: int = 27017,
                 db_name: str = "GAG_DB", user: str = "GAG", password: str = "GAG321"):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.user = user
        self.password = password
        self._client: Optional[MongoClient] = None
        self._db = None
        self._connect()

    def _connect(self):
        """Stellt die Verbindung zur MongoDB-Instanz her."""
        try:
            self._client = MongoClient(f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}/?authSource=admin")
            self._client.admin.command('ping')
            self._db = self._client[self.db_name]
            print(f"Erfolgreich mit MongoDB verbunden: {self.host}:{self.port}, Datenbank: {self.db_name}")
        except Exception as e:
            print(f"Fehler beim Verbinden mit MongoDB: {e}")
            print("Stellen Sie sicher, dass der MongoDB Docker-Container läuft und die Anmeldedaten korrekt sind.")
            self._client = None
            self._db = None

    def get_collection(self, collection_name: str):
        """Gibt eine Datenbank-Collection zurück."""
        if self._db is not None:
            return self._db[collection_name]
        return None

    def close_connection(self):
        """Schließt die MongoDB-Verbindung."""
        if self._client:
            self._client.close()
            print("MongoDB-Verbindung geschlossen.")

    def check_episode_data_completeness(self, episode_id: int, episodes_with_description) -> Dict[str, bool]:
        """
        Überprüft, welche der Hauptdatenpunkte für eine gegebene Episode-ID
        in der Datenbank vorhanden und nicht leer/null sind.

        Args:
            episode_id (int): Die _id der Episode, die überprüft werden soll.

        Returns:
            Dict[str, bool]: Ein Dictionary, das für jeden Datenpunkt True zurückgibt,
                             wenn er vorhanden und nicht leer/null/ungültig ist, sonst False.
                             Gibt Standard-False-Werte zurück, wenn die Verbindung fehlschlägt
                             oder die Episode nicht existiert.
        """

        #print("bis hier")

        episode_title = episodes_with_description.get(str(episode_id-1), {}).get("title")
        print(episode_title)
        if not episode_title:
            print(f"Fehler: Titel für Episode {episode_id} nicht gefunden.")

        match = re.search(r'\d+', episode_title)
        zahl_als_int = None
        print("bis hier")

        if match:
            zahl_als_string = match.group(0)
            zahl_als_int = int(zahl_als_string)
            print(f"Die extrahierte Zahl (als String): {zahl_als_string}")
            print(f"Die extrahierte Zahl (als Integer): {zahl_als_int}")
            print(f"Typ der extrahierten Zahl: {type(zahl_als_int)}")
        else:
            print("Keine Zahl im String gefunden. Episode kann nicht gespeichert werden.")
            
        print("bis hier")
        results = {
            "audio_paths": False,
            "full_text_transcript": False,
            "speaker_times": False,
            "segment_timeline": False,
            "Geschichtenerzähler": False # Hauptsprecher
        }
        
        if self._db is None:
            print("Fehler: Keine Datenbankverbindung verfügbar, um Daten-Vollständigkeit zu prüfen.")
            return results

        episodes_collection = self._db["episodes"]
        
        try:
            # Projiziere alle relevanten Felder in einer einzigen Abfrage
            existing_doc = episodes_collection.find_one(
                {"_id": zahl_als_int},
                {
                    "audio_paths": 1,
                    "full_text_transcript": 1,
                    "speaker_times": 1,
                    "segment_timeline": 1,
                    "Geschichtenerzähler": 1,
                    "_id": 0 # _id ausschließen, da wir es nur zur Suche verwenden
                }
            )

            if existing_doc:
                # 1. Prüfe "audio_paths"
                # Erwarte ein Dictionary mit "full" (str) und "split" (list mit mindestens einem Element)
                if existing_doc.get("audio_paths") and \
                   isinstance(existing_doc["audio_paths"], dict) and \
                   existing_doc["audio_paths"].get("full") and \
                   isinstance(existing_doc["audio_paths"].get("split"), list) and \
                   len(existing_doc["audio_paths"]["split"]) > 0:
                    results["audio_paths"] = True
                
                # 2. Prüfe "full_text_transcript"
                # Erwarte einen nicht-leeren String
                if existing_doc.get("full_text_transcript") and \
                   isinstance(existing_doc["full_text_transcript"], str) and \
                   existing_doc["full_text_transcript"] != "":
                    results["full_text_transcript"] = True
                
                # 3. Prüfe "speaker_times"
                # Erwarte ein Dictionary mit Daniel und Richard, die Zahlenwerte haben
                if existing_doc.get("speaker_times") and \
                   isinstance(existing_doc["speaker_times"], dict) and \
                   existing_doc["speaker_times"].get("Daniel") is not None and \
                   isinstance(existing_doc["speaker_times"].get("Daniel"), (int, float)) and \
                   existing_doc["speaker_times"].get("Richard") is not None and \
                   isinstance(existing_doc["speaker_times"].get("Richard"), (int, float)):
                    results["speaker_times"] = True

                # 4. Prüfe "segment_timeline"
                # Erwarte eine nicht-leere Liste
                if existing_doc.get("segment_timeline") and \
                   isinstance(existing_doc["segment_timeline"], list) and \
                   len(existing_doc["segment_timeline"]) > 0:
                    results["segment_timeline"] = True

                # 5. Prüfe "Geschichtenerzähler" (hauptsprecher)
                # Erwarte eine Zahl
                if existing_doc.get("Geschichtenerzähler") is not None and \
                   isinstance(existing_doc["Geschichtenerzähler"], (int, float)):
                    results["Geschichtenerzähler"] = True
            
        except Exception as e:
            print(f"Fehler beim Prüfen der Daten-Vollständigkeit für Episode {episode_id}: {e}")
            # Bei einem Fehler ist es sicherer anzunehmen, dass die Daten nicht zuverlässig vorhanden sind
            return {key: False for key in results} # Setze alle auf False im Fehlerfall
        
        return results

    def get_full_text_transcript(self, episode_id: int) -> Optional[str]:
        """
        Ruft den vollständigen transkribierten Text für eine gegebene Episode-ID ab.

        Args:
            episode_id (int): Die _id der Episode, deren Transkript abgerufen werden soll.

        Returns:
            Optional[str]: Der vollständige transkribierte Text als String,
                           oder None, wenn die Episode nicht gefunden wird,
                           das Feld nicht existiert, leer ist oder ein Fehler auftritt.
        """
        if self._db is None:
            print("Fehler: Keine Datenbankverbindung verfügbar, um Transkript abzurufen.")
            return None

        episodes_collection = self._db["episodes"]
        
        try:
            # Suche das Dokument und projiziere nur das Feld 'full_text_transcript'
            document = episodes_collection.find_one(
                {"_id": episode_id},
                {"full_text_transcript": 1, "_id": 0} # Nur den Text und keine _id zurückgeben
            )

            if document:
                transcript = document.get("full_text_transcript")
                if isinstance(transcript, str) and transcript: # Prüfen, ob es ein nicht-leerer String ist
                    return transcript
                else:
                    print(f"Warnung: 'full_text_transcript' für Episode {episode_id} gefunden, aber leer oder kein String.")
                    return None
            else:
                print(f"Info: Dokument für Episode {episode_id} nicht in der Collection 'episodes' gefunden.")
                return None
        except Exception as e:
            print(f"Fehler beim Abrufen des Transkripts für Episode {episode_id}: {e}")
            return None

    def get_segment_timeline(self, episode_id: int) -> Optional[List[Dict[str, Any]]]:
        """
        Ruft die Segment-Timeline für eine gegebene Episode-ID ab.

        Args:
            episode_id (int): Die _id der Episode, deren Segment-Timeline abgerufen werden soll.

        Returns:
            Optional[List[Dict[str, Any]]]: Die Segment-Timeline als Liste von Dictionaries,
                                             oder None, wenn die Episode nicht gefunden wird,
                                             das Feld nicht existiert, leer ist oder ein Fehler auftritt.
        """
        if self._db is None:
            print("Fehler: Keine Datenbankverbindung verfügbar, um Segment-Timeline abzurufen.")
            return None

        episodes_collection = self._db["episodes"]
        
        try:
            # Suche das Dokument und projiziere nur das Feld 'segment_timeline'
            document = episodes_collection.find_one(
                {"_id": episode_id},
                {"segment_timeline": 1, "_id": 0} # Nur die Timeline und keine _id zurückgeben
            )

            if document:
                timeline = document.get("segment_timeline")
                if isinstance(timeline, list) and timeline: # Prüfen, ob es eine nicht-leere Liste ist
                    return timeline
                else:
                    print(f"Warnung: 'segment_timeline' für Episode {episode_id} gefunden, aber leer oder keine Liste.")
                    return None
            else:
                print(f"Info: Dokument für Episode {episode_id} nicht in der Collection 'episodes' gefunden.")
                return None
        except Exception as e:
            print(f"Fehler beim Abrufen der Segment-Timeline für Episode {episode_id}: {e}")
            return None

    def get_speaker_times(self, episode_id: int) -> Optional[Dict[str, float]]:
        """
        Ruft die Sprecherzeiten ('speaker_times' für Daniel und Richard) für eine gegebene Episode-ID ab.

        Args:
            episode_id (int): Die _id der Episode, deren Sprecherzeiten abgerufen werden sollen.

        Returns:
            Optional[Dict[str, float]]: Ein Dictionary mit Daniel und Richard's Sprechzeiten (z.B. {'Daniel': 123.4, 'Richard': 56.7}),
                                     oder None, wenn die Episode nicht gefunden wird,
                                     das Feld nicht existiert, ungültig ist oder ein Fehler auftritt.
        """
        if self._db is None:
            print("Fehler: Keine Datenbankverbindung verfügbar, um Sprecherzeiten abzurufen.")
            return None

        episodes_collection = self._db["episodes"]
        
        try:
            # Suche das Dokument und projiziere nur das Feld 'speaker_times'
            document = episodes_collection.find_one(
                {"_id": episode_id},
                {"speaker_times": 1, "_id": 0} # Nur speaker_times projizieren
            )

            if document:
                speaker_times = document.get("speaker_times")

                # Prüfen, ob speaker_times gültig ist und die erwarteten Keys enthält
                is_speaker_times_valid = (
                    isinstance(speaker_times, dict) and
                    speaker_times.get("Daniel") is not None and isinstance(speaker_times.get("Daniel"), (int, float)) and
                    speaker_times.get("Richard") is not None and isinstance(speaker_times.get("Richard"), (int, float))
                )

                if is_speaker_times_valid:
                    # Sicherstellen, dass die Werte Floats sind
                    return {
                        "Daniel": float(speaker_times["Daniel"]),
                        "Richard": float(speaker_times["Richard"])
                    }
                else:
                    print(f"Warnung: 'speaker_times' für Episode {episode_id} gefunden, aber unvollständig oder ungültig.")
                    return None
            else:
                print(f"Info: Dokument für Episode {episode_id} nicht in der Collection 'episodes' gefunden.")
                return None
        except Exception as e:
            print(f"Fehler beim Abrufen der Sprecherzeiten für Episode {episode_id}: {e}")
            return None
    
    def get_whisper_texts(self, episode_id: int) -> Optional[List[Dict[str, Any]]]: # <-- NEUE FUNKTION HIER
        """
        Ruft die detaillierten Whisper-Texte (mit Chunks und Timestamps) für eine Episode ab.
        """
        episodes_collection = self.get_collection("episodes")
        if episodes_collection:
            episode_doc = episodes_collection.find_one({"_id": episode_id}, {"Whisper_texts": 1})
            if episode_doc:
                return episode_doc.get("Whisper_texts")
        return None

## MongoDB-Funktion zur Speicherung von Episodendaten

def store_episode_data_in_mongodb(
    episodes_with_description: Dict[str, Any],
    episode_numbers: List[int],
    shortened_audios_for_transcriptions: List[str],
    daniel_time: float,
    richard_time: float,
    full_concatenated_text: str,
    texts,
    timeline_of_segments: List[Dict[str, Any]],
    pipeline_steps: set
):
    """
    Extrahiert Episodendaten, bereitet sie auf und speichert sie in einer MongoDB-Collection.

    Args:
        episodes_with_description (Dict[str, Any]): Dictionary mit Episodenbeschreibungen.
        episode_numbers (List[int]): Liste der Episodennummern. Es wird die erste Nummer verwendet.
        shortened_audios_for_transcriptions (List[str]): Liste der Pfade zu gekürzten Audios.
                                                          Es wird der erste Pfad verwendet.
        daniel_time (float): Sprechzeit von Daniel.
        richard_time (float): Sprechzeit von Richard.
        full_concatenated_text (str): Der vollständige transkribierte Text.
        timeline_of_segments (List[Dict[str, Any]]): Zeitachse der Segmente.
    """
    if not episode_numbers:
        print("Fehler: Keine Episodennummern bereitgestellt.")
        return

    # 1. Episodennummer aus dem Titel extrahieren
    episode_title = episodes_with_description.get(str(episode_numbers[0]-1), {}).get("title")
    print(episode_title)
    if not episode_title:
        print(f"Fehler: Titel für Episode {episode_numbers[0]} nicht gefunden.")
        return

    match = re.search(r'\d+', episode_title)
    zahl_als_int = None

    if match:
        zahl_als_string = match.group(0)
        zahl_als_int = int(zahl_als_string)
        print(f"Die extrahierte Zahl (als String): {zahl_als_string}")
        print(f"Die extrahierte Zahl (als Integer): {zahl_als_int}")
        print(f"Typ der extrahierten Zahl: {type(zahl_als_int)}")
    else:
        print("Keine Zahl im String gefunden. Episode kann nicht gespeichert werden.")
        return

    # 2. AudioPaths initialisieren
    # Sicherstellen, dass shortened_audios_for_transcriptions nicht leer ist
    if not shortened_audios_for_transcriptions:
        print("Fehler: Keine gekürzten Audiodaten für Transkriptionen bereitgestellt.")
        return

    audio_paths_instance = AudioPaths(
        full="../audioData/" + "GAG" + zahl_als_string + ".mp3",
        split=["../audioData/shortened_audios/" + str(shortened_audios_for_transcriptions[0])]
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

    if daniel_time is None or richard_time is None:
        hauptsprecher = None
        print("Warnung: Eine der Sprecherzeiten (Daniel oder Richard) ist None. 'hauptsprecher' wird auf None gesetzt.")
    else:
        hauptsprecher = max(daniel_time, richard_time)

    #print("funktioniert bis hier AHuptsprecher ")

    # --- Dokument vorbereiten, das in MongoDB gespeichert wird ---
    episode_document = {
        "_id": zahl_als_int,  # Episodennummer als MongoDB _id verwenden
        "audio_paths": audio_paths_dict,
        "full_text_transcript": full_concatenated_text,
        "Whisper_texts": texts,
        "speaker_times": speaker_times,
        "segment_timeline": timeline_of_segments,
        "Geschichtenerzähler": hauptsprecher
    }

    episode_document_for_gui = {
        "_id": zahl_als_int,  # Episodennummer als MongoDB _id verwenden
        "audio_paths": audio_paths_dict,
        "full_text_transcript": full_concatenated_text,
        "speaker_times": speaker_times,
        "segment_timeline": timeline_of_segments,
        "Geschichtenerzähler": hauptsprecher
    }

    if "db_build" in pipeline_steps:

        #print("funktioniert bis hier")

        # Eine Instanz deines MongoDBManagers erstellen
        db_manager = MongoDBManager()

        # Die Collection auswählen, in der die Episodendaten gespeichert werden sollen
        episodes_collection = db_manager.get_collection("episodes")

        print("funktioniert bis hier episodes_collection")
        if episodes_collection:
            try:
                # Das Dokument in die Collection einfügen oder aktualisieren
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

        if db_manager._db:
            try:
                collection_names = db_manager._db.list_collection_names()
                print(f"\nCollections in Datenbank '{db_manager.db_name}': {collection_names}")

                if "episodes" in collection_names:
                    episode_count = episodes_collection.count_documents({})
                    print(f"Anzahl der Dokumente in 'episodes' Collection: {episode_count}")
                else:
                    print("Die 'episodes' Collection existiert noch nicht (oder ist leer).")

            except Exception as e:
                print(f"Fehler beim Abfragen der Collections oder Dokumente: {e}")
        else:
            print("Keine Datenbankverbindung verfügbar, um Collections abzufragen.")

        # Verbindung schließen, wenn fertig
        db_manager.close_connection()
        return episode_document_for_gui
    else:
        #db_manager.close_connection()
        return episode_document_for_gui
