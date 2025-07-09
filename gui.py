import streamlit as st
import pandas as pd
from streamlit_timeline import *
from dataScraping import *
from pipeline import *
import re

@st.cache_data
def gui_load_json(file_path_json)->dict:

    #file_path_json = "titles_summaries.json"
    # Daten laden (deserialisieren)
    loaded_data_json = {}
    try:
        with open(file_path_json, 'r', encoding='utf-8') as f:
            loaded_data_json = json.load(f)
        print(f"Daten erfolgreich aus '{file_path_json}' geladen:")
        #print(loaded_data_json)
        return loaded_data_json
    except FileNotFoundError:
        print(f"Datei '{file_path_json}' nicht gefunden.")
        return loaded_data_json

episodes_with_description = gui_load_json("all_titles_summaries.json")

if get_episode_count() > len(episodes_with_description.keys()):

    make_titles_summaries_json()
    episodes_with_description = gui_load_json("all_titles_summaries.json")

#latest_episode_count_with_timestamp = None

@st.cache_resource
def get_whisper_instance():
    """
    Diese Funktion lädt die WhisperModell-Instanz nur einmal
    und cached sie für nachfolgende Ausführungen der App.
    """
    return WhisperModell()

def validate_number_input(input_string: str) -> bool:
    """
    Validiert eine Eingabe, die einzelne positive Ganzzahlen,
    kommagetrennte Listen positiver Ganzzahlen oder
    Bereiche positiver Ganzzahlen (z.B. 10-20) enthalten kann.
    Keine negativen Zahlen, Floats oder Buchstaben.
    """
    # Das Regex-Muster
    # ^                                  # Start des Strings
    # (?:                                # Beginnt eine nicht-fangende Gruppe für Alternativen
    #    \d+                             #    Fall 1: Eine einzelne positive ganze Zahl (z.B. "10")
    #    |                               #    ODER
    #    \d+(?:,\d+)* #    Fall 2: Eine kommagetrennte Liste (z.B. "1,2,3")
    #                                    #       \d+        : Erste Zahl
    #                                    #       (?:,\d+)* : Null oder mehr Wiederholungen von ",Zahl"
    #    |                               #    ODER
    #    \d+-\d+                         #    Fall 3: Ein Bereich (z.B. "10-20")
    # )                                  # Endet die nicht-fangende Gruppe
    # $                                  # Ende des Strings
    pattern = r"^(?:\d+|\d+(?:,\d+)*|\d+-\d+)$"

    # re.fullmatch ist hier ideal, da der gesamte String dem Muster entsprechen muss.
    if re.fullmatch(pattern, input_string):
        return True
    else:
        return False

def parse_episode_input(input_str)->list:
    
    nums = []
    for part in input_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            nums.extend(range(start, end + 1))
        else:
            nums.append(int(part))
    return sorted(list(set(nums)))

#Funktion für Backend
def check_and_load_json_file(file_path: str) -> dict | None:
    
    #Überprüft, ob eine JSON-Datei existiert, lesbar ist und gültiges JSON enthält.
    #Gibt den Inhalt als Dictionary zurück oder None bei Fehlern.
    
    print(f"Versuche, Datei '{file_path}' zu laden...")
    try:
        # 1. Prüfen, ob die Datei existiert (optional, aber nützlich für klare Fehlermeldungen)
        if not os.path.exists(file_path):
            print(f"Fehler: Datei '{file_path}' existiert nicht.")
            return None

        # 2. Versuch, die Datei zu öffnen und zu lesen
        with open(file_path, 'r', encoding='utf-8') as f:
            # 3. Versuch, den Inhalt als JSON zu parsen
            data = json.load(f)
            print(f"Datei '{file_path}' erfolgreich geladen und geparst.")
            return data
            
    except FileNotFoundError:
        # Dieser Fehler wird gefangen, wenn os.path.exists() nicht verwendet wird
        # oder wenn der Pfad aus anderen Gründen nicht gefunden wird (z.B. falsche Berechtigungen).
        print(f"Fehler: Datei '{file_path}' konnte nicht gefunden werden (oder Zugriff verweigert).")
        return None
    except json.JSONDecodeError:
        # Dieser Fehler tritt auf, wenn der Dateiinhalt kein gültiges JSON ist
        print(f"Fehler: Datei '{file_path}' enthält kein gültiges JSON oder ist leer.")
        return None
    except PermissionError:
        # Wenn die App keine Leserechte für die Datei hat
        print(f"Fehler: Keine Berechtigung zum Lesen der Datei '{file_path}'.")
        return None
    except Exception as e:
        # Fängt alle anderen unerwarteten Fehler ab
        print(f"Ein unerwarteter Fehler ist aufgetreten beim Laden von '{file_path}': {e}")
        return None

# --- Streamlit GUI Start ---

def start_gui():

    analysis_results = None

    #global episodes_with_description
    #global latest_episode_count_with_timestamp

    st.title("Podcast Analyse Tool")
    st.markdown("Ein Werkzeug zum Bauen von Pipelines für den Podcast 'Geschichten aus der Geschichte'!")

    with st.expander("⚙️ Verfügbare Episoden zeigen mit Titel und Beschreibung", expanded=True):
        
        st.subheader("1. Episoden auswählen für Auflistung von Episoden mit Titel und Beschreibung in absteigender Reihenfolge, dabei steht 1 für die zuletzt veröffentlichte Folge")

        episode_input = st.text_input(
            "Geben Sie die gewünschten Episodennummern ein, um interessante Themen zu finden (z.B. '1,2,5-7'):",
            value="1" # Default value for testing
        )

        #Erst checken, ob ein json dafür gespeichert ist
        #Wenn es gespeichert ist, dann rausladen und ausgeben
        #Ansonsten erstelle ein dict dafür und speichere es und gebe es aus im Streamlit 
        st.info("Momentan verfügbare Episoden: " + str(get_episode_count()))
        if episode_input != '' and validate_number_input(episode_input):
            
            parsed_episode_numbers = parse_episode_input(episode_input)
            integer_liste = [int(x) for x in parsed_episode_numbers]

            if max(integer_liste) >= len(episodes_with_description.keys()):

                st.info("Zu hohe Zahl.")

            else:
                
                st.info(f"Ausgewählte Episoden: {', '.join(map(str, parsed_episode_numbers)) if parsed_episode_numbers else 'Keine'}")
                
                current_titels_and_summaries = dict()
                #current_titels_and_summaries = getTitlesAndDescriptions(parsed_episode_numbers)
                for i in range(0, len(parsed_episode_numbers), 1):

                    current_titels_and_summaries[str(parsed_episode_numbers[i]-1)] = {'title': episodes_with_description[str(parsed_episode_numbers[i]-1)]['title'], 'Beschreibung':episodes_with_description[str(parsed_episode_numbers[i]-1)]['Beschreibung']}
                st.json(current_titels_and_summaries)

        else:

            st.info('Keine Episode momentan ausgewählt')

        #Get data about title and summary of chosen episodes
    #-------------------------------------------------------------------------------------------------------------

    with st.expander("⚙️ Konfigurieren und Starten der Analyse", expanded=True):
        st.subheader("2. Episoden auswählen für Analyse-Pipeline")
        #episode_input = st.text_input(
        #    "Geben Sie die gewünschten Episodennummern ein, um diese zu analysieren mit der Pipeline (z.B. '1,2,5-7'):",
        #    value="1" # Default value for testing
        #)
        # Helper function to parse "1,2,5-7"

        #if episode_input != '' and validate_number_input(episode_input) and int(episode_input) <=get_episode_count():
            
        #    parsed_episode_numbers_for_analysis = parse_episode_input(episode_input)
        #    st.info(f"Ausgewählte Episoden: {', '.join(map(str, parsed_episode_numbers)) if parsed_episode_numbers else 'Keine'}")
            
        #    #current_titels_and_summaries2 = getTitlesAndDescriptions(parsed_episode_numbers_for_analysis)
        #st.info(f"Ausgewählte Episoden: {', '.join(map(str, parsed_episode_numbers_for_analysis)) if parsed_episode_numbers_for_analysis else 'Keine'}")

        st.subheader("2. Aufgaben für die Pipeline auswählen")
        col1, col2 = st.columns(2)
        with col1:
            #do_download = st.checkbox("📥 Audiodownload", value=False)
            do_db_build = st.checkbox("💾 Daten speichern in Datenbank", value=False)
            do_transcribe = st.checkbox("🎤 Transkription", value=False)
            do_segment_classification = st.checkbox("🧩 Segmentklassifizierung", value = False)
        with col2:
            do_SpeakerDiariazation = st.checkbox("📊Sprecherzuordnung", value=False)
            do_speaker_analysis = st.checkbox("🗣️ Hauptsprecher bestimmen", value=False)
            #do_
            #do_content_analysis = st.checkbox("🔍 Referenzen zu anderen Folgen bestimmen", value=False)

        #whisper_model = st.selectbox(
        #    "Whisper Modell für Transkription:",
        #    options=["tiny", "base", "small", "medium", "large"],
        #    index=1 # 'base' is a good default
        #)

        if st.button("🚀 Analyse starten"):
            if not parsed_episode_numbers:
                st.warning("Bitte geben Sie mindestens eine Episodennummer ein.")
            else:
                # DYNAMISCHE ERSTELLUNG DER PIPELINE-SCHRITTE BASIEREND AUF USER-INPUT
                #PING-PONG-KASKADE: Gui zu Pipeline zu den jeweiligen Funktionen, die als Paramter benutzt werden
                #und das Result zurück an die Gui wieder

                pipeline_steps = set()
                #Pipelines with strings in set, filled according to the Checking of the boxes from the user
            
                # Jeder Eintrag ist (Funktion, (optionale_positions_args), {optionale_keyword_args})
                # Die Funktion muss prev_result und episode_data_context als erste Argumente erwarten!

                #if do_download: 

                #    pipeline_steps.add("download")

                if do_transcribe:

                    pipeline_steps.add("Whisper_Pipe")
                
                if do_SpeakerDiariazation:

                    pipeline_steps.add("SpeakerDiariazation")

                if do_speaker_analysis:

                    pipeline_steps.add("speaker_analysis")

                if do_segment_classification:

                    pipeline_steps.add("segment_classification")

                if do_db_build:

                    pipeline_steps.add("db_build")

                #current_pipeline = AnalysisPipeline(pipeline_steps)
                
                #if not len(pipeline_steps) == 0:
                #    st.warning("Bitte wählen Sie mindestens eine Aufgabe aus.")
                    #continue

                
                #whisperModel = get_whisper_instance()
                try: 
                    with st.spinner(f"Starte Analyse für {len(parsed_episode_numbers)} Episoden..."):
                        analysis_results = run_dynamic_pipeline(
                            episodes_with_description,
                            parsed_episode_numbers,
                            current_titels_and_summaries,
                            pipeline_steps = pipeline_steps
                        )
                except Exception as e:

                    st.error("Fehler bei der Eingabe der Episodennummern. Bitte geben Sie gültige Zahlen oder Bereiche ein (z.B. 1, 3-5).")


    with st.expander("⚙️ Daten und Analyse-Ergebnisse zur Anzeige auswählen", expanded=True):

        if analysis_results is not None: # Ändere !=None zu is not None für Pythonic Style
            for episode_doc in analysis_results:

                episode_id = episode_doc.get("_id")
                st.markdown(f"---")
                st.markdown(f"### Ergebnisse für Episode {episode_id}")

                # Audioplayer
                st.subheader("📥 Audioplay")
                audio_path = episode_doc.get("audio_paths", {}).get("full")
                if audio_path and os.path.exists(audio_path):
                    st.audio(audio_path, format='audio/mp3', start_time=0)
                else:
                    st.warning(f"Audio-Datei nicht gefunden oder Pfad ungültig für Episode {episode_id}: {audio_path}")

                # Transkription
                st.subheader("🎤 Transkription")
                full_text = episode_doc.get("full_text_transcript")
                if full_text:
                    st.text_area("Vollständiger Transkriptionstext von Episode"+str(episode_id), full_text, height=200, key=f"transcript_ep_{str(episode_id)}")
                else:
                    st.info("Keine vollständige Transkription für diese Episode verfügbar.")

                # Sprecherzuordnung (Segment-Timeline)
                st.subheader("📊 Länge der Segmente (Segment-Timeline)")
                timeline = episode_doc.get("segment_timeline")
                if timeline:
                    for i, segment in enumerate(timeline):
                        current_start_time = segment.get("Start-Zeit")
                        overall_segment_duration = segment.get("Gesamt-Zeit") # Dauer des aktuellen Segments
                        label = segment.get("Klasse")

                        # --- Eingabevalidierung ---
                        if any(val is None for val in [current_start_time, overall_segment_duration, label]):
                            st.warning(f"Ein Segment-Dictionary ist unvollständig oder fehlerhaft: {segment}")
                            continue

                        # Initialisierung für die Anzeigenstrings
                        duration_in_main_line = "" # Für "Dauer: [X.Ys]" in der Segmentzeile
                        header_line_text = ""      # Für die Zeile darüber

                        time_elapsed_since_previous_start = None
                        if i == 0:
                            # --- Logik für das erste Segment ---
                            header_line_text = f"**Gesamt-Dauer des ersten Segments: {overall_segment_duration:.1f}s**"
                            # Hier war ein Kommentar, der die Zeile auskommentiert hat, ich lasse ihn so, wie Sie es gesendet haben.
                            # duration_in_main_line = f"Dauer: [{overall_segment_duration:.1f}s]"
                        elif i == len(timeline)-1:
                            # Diese Logik für das vorletzte und letzte Segment scheint auf einer spezifischen Annahme zu basieren.
                            # next_segment_start_time sollte der Start des NÄCHSTEN Segments sein.
                            # Für das letzte Segment gibt es kein "nächstes Segment".
                            # Die vorherige Implementierung, bei der der Start des nächsten Segments (timeline[i+1]) verwendet wurde,
                            # ist konsistenter mit der Berechnung der "End-Zeit".
                            # Ich lasse es so, wie Sie es angegeben haben, obwohl es möglicherweise überarbeitet werden sollte,
                            # um die Endzeit des Letzten segments besser zu definieren (z.B. current_start_time + overall_segment_duration).
                            next_segment_start_time = timeline[i].get("Gesamt-Zeit")
                            time_elapsed_since_previous_start = overall_segment_duration
                        else:
                            # --- Logik für alle nachfolgenden Segmente (inkl. vorletztem und letztem) ---
                            next_segment_start_time = timeline[i+1].get("Start-Zeit")

                            if next_segment_start_time is not None:
                                # Die Zeit seit dem Start des VORHERIGEN Segments bis zum Start des AKTUELLEN Segments
                                time_elapsed_since_previous_start = abs(current_start_time - next_segment_start_time)

                                # Hier waren Kommentare, die die Zeilen auskommentiert haben.
                                # header_line_text = f"**Dauer: {time_elapsed_since_previous_start:.1f}s**"
                                # duration_in_main_line = f"Dauer: [{time_elapsed_since_previous_start:.1f}s vom letzten Start]"
                            else:
                                # Fallback, falls die Startzeit des vorherigen Segments fehlt
                                header_line_text = "**Fehler: Konnte vorherige Startzeit nicht abrufen.**"
                                # duration_in_main_line = f"Dauer: [{overall_segment_duration:.1f}s (Segmentlänge)]"

                        # --- Ausgabe der Kopfzeile ---
                        # Hier war ein Kommentar, der die Zeile auskommentiert hat.
                        # st.markdown(header_line_text)

                        if i > 0: # Trennlinie nur zwischen Segmenten (nicht vor dem ersten)
                            st.markdown("---")

                        # --- Anzeige der Details des aktuellen Segments ---
                        if i == 0:
                            st.markdown(
                                f"**Start-Zeit[{current_start_time:.1f}s] – End-Zeit: {timeline[i+1].get('Start-Zeit'):.1f}s – {duration_in_main_line} – Klasse: {label}**"
                            )
                        else:
                            st.markdown(
                                f"**Start-Zeit[{current_start_time:.1f}s] – End-Zeit: {next_segment_start_time:.1f}s – {time_elapsed_since_previous_start} – Klasse: {label}**"
                            )
                        st.markdown(" ") # Zusätzlicher Leerzeilen-Platzhalter für die Optik

                else:
                    st.info("Keine Segmentklassifizierung für diese Episode verfügbar.")

                # Hauptsprecher
                st.subheader("🗣️ Hauptsprecher")
                speaker_times = episode_doc.get("speaker_times")
                hauptsprecher_time = episode_doc.get("Geschichtenerzähler")
                if speaker_times:
                    st.write("Sprechzeiten:")
                    for speaker, time_val in speaker_times.items():
                        # Hinzufügen der None-Überprüfung hier:
                        if time_val is not None:
                            st.write(f"- {speaker}: {time_val:.2f} Sekunden")
                        else:
                            st.write(f"- {speaker}: Daten nicht verfügbar")
                    if hauptsprecher_time is not None:
                        hauptsprecher_name = "Unbekannt"
                        # Finde den Namen des Sprechers, der die 'Geschichtenerzähler'-Zeit hat
                        for speaker, time_val in speaker_times.items():
                            if time_val == hauptsprecher_time:
                                hauptsprecher_name = speaker
                                break
                        st.markdown(f"**Der Hauptsprecher ist: {hauptsprecher_name}**")
                else:
                    st.info("Keine Sprecheranalyse für diese Episode verfügbar.")

                # Referenzen zu anderen Folgen
                st.subheader("🔍 Referenzen zu anderen Folgen")
                related_episodes = episode_doc.get("related_episodes")
                if related_episodes:
                    # Der äußere if-Block ist überflüssig, wenn related_episodes bereits geprüft wurde
                    # (if related_episodes: if related_episodes:)
                    # Die Logik wurde in früheren Antworten bereits vereinfacht.
                    for ref in related_episodes:
                        st.write(f"- Episode {ref.get('id')} (Ähnlichkeit: {ref.get('similarity'):.2f})")
        else:
            st.info("Keine Referenzen zu anderen Folgen gefunden.")

    with st.expander("⚙️ Evaluation der Analyse-Pipeline", expanded=True):
        
        st.subheader("4. Episoden auswählen und Aufgaben und diese mit eigenen Daten evaluieren")    
# --- Startpunkt der Anwendung ---
if __name__ == "__main__":
    # Initialisierung des Session State, falls noch nicht vorhanden
    
    start_gui()