import torch
import os
import chromadb
from chromadb.utils import embedding_functions

# --- Modellname für die Embedding-Funktion ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_chroma_db_with_embeddings():
    """
    Erstellt eine ChromaDB-Datenbank und speichert die bereitgestellten Embeddings persistent.
    Verwendet das angegebene SentenceTransformer-Modell.
    """

    data_to_embed = {
    "Feedback": [
        "Wir freuen uns über eure Kommentare und Nachrichten. Schreibt uns gerne eure Meinung!",
        "Ihr könnt uns bewerten auf eurer Lieblings-Podcast-Plattform. Jede Rezension hilft uns sehr.",
        "Für Anregungen oder Kritik nutzt bitte unsere E-Mail-Adresse oder sozialen Kanäle.",
        "Euer Feedback ist uns wichtig. Teilt uns mit, was ihr denkt!",
    ],
    "Intro": [
        "Lernt ein bisschen Geschichte, dann werdet ihr sehen, wie sich der Reporter damals entwickelt hat. Hallo und herzlich willkommen bei Geschichten aus der Geschichte. Mein Name ist Richard. Und mein Name ist Daniel",
        "Hallo liebe Hörerinnen und Hörer, wir melden uns kurz aus der Zukunft. Seit Folge 270 heisst dieser Podcast Geschichten aus der Geschichte. Mehr dazu findet ihr unter Geschichte.fm. Und jetzt geht's zurück in die Vergangenheit. Viel Spass. Lernt ein bisschen Geschichte, dann werdet ihr sehen, wie der Reporter sich damals entwickelt hat. Wie der sich damals entwickelt hat. Hallo und herzlich willkommen bei Zeitsprung-Geschichten aus der Geschichte. Ich bin Daniel. Und ich bin Richard. Ja, Richard, wir sehen uns heute zur Episode 32. Ja. Dick. Für alle, die neu dazugekommen sind, wir sind zwei Historiker, erzählen Geschichten aus der Geschichte und da wir das mal abwechselnd tun und ich letzte Woche die Geschichte erzählt habe. Bist du diese Woche dran, Richard? Um was geht es diesmal?",
        "Einen wunderschönen guten Tag und herzlich willkommen zu einer neuen Folge eures Lieblingspodcasts.",
        "Guten Abend allerseits und schön, dass ihr wieder eingeschaltet habt bei Geschichten aus der Geschichte.",
        "Starten wir in die heutige Episode! Richard und Daniel begrüßen euch.",
        "Liebe Zuhörer, schön, dass ihr wieder dabei seid. Wir beginnen jetzt mit der Geschichte.",
        "Und ich bin Daniel",
        "Ich bin Richard",
        "Wie der Report sich damals entwickelt hat.",
    ],
    "Feedbackhinweisblock": [
        "Wer uns Feedback geben will zu dieser Folge oder auch zu anderen Folgen, die wir gemacht haben, kann das per E-Mail machen. Feedback-at-Geschichte.fm kann es auf den diversen Social-Media-Plattformen tun. Dort heissen wir gemeinhin Geschichte FM, ausser bei Mastodon, da gibt man einfach Geschichte.social in einen Browser ein und landet dann direkt auf unserem Profil",
        "Und für alle, die uns Feedback geben wollen, Feedback gerne wie immer über unterschiedliche Kanäle und Plattformen, Entweder Twitter an den Daniel-at-Messner oder mich at Stormgrass oder auch auf Facebook unter facebook.com.com. oder auch direkt auf unserem Blog unter zeitsprung.fm oder auch eine Bewertung auf iTunes, entweder mit Sternen oder in Textform als Rezension oder auch per E-Mailail, feedback-at-zeit.de. Und danke für alle Bewertungen, für alle Kommentare, die bis jetzt schon gekommen sind. Ja, Nachrichten, alles. Und auch danke für die Flatter-Clicks, die wir bekommen haben. Oh, Flatter. Flatter ist ja leider so ein bisschen gerade am Untergehen, aber so ein bisschen funktioniert es noch. Das heisst, danke, die uns da fördern",
        "Ihr erreicht uns auf Twitter unter @Messner und @Stormgrass, auf Facebook oder per Mail.",
        "Vergesst nicht, eine Bewertung auf Apple Podcasts zu hinterlassen – Sterne und Rezensionen sind sehr willkommen!",
        "Alle Links zum Feedback und unseren Social Media Kanälen findet ihr in den Shownotes.",
        "Eure Rückmeldungen sind wertvoll. Besucht uns auf unserer Webseite zeitsprung.fm oder schreibt uns an Feedback-at-Zeit.de.",
        "Gehen wir zum Feedback",
        "Wer uns Feedback geben möchte",
        "Wir freuen uns über Feedback",
    ],
    "Werbung": [
        "Hier kommt unser Werbpartner ins Spiel",
        "Bevor es hier jetzt aber weitergeht, kommt noch eine kleine Werbeeinschaltung.",
        "Kurze Unterbrechung für unsere Partner. Danach geht's direkt weiter mit der Geschichte.",
        "Dieser Podcast wird präsentiert von Werbepartnern. Mehr Infos dazu im Anhang.",
        "Ein kurzer Hinweis zu unserem heutigen Sponsor.",
        "Wir machen eine kurze Pause für eine Nachricht von unserem Werbepartner.",
        "Unser Merch",
        "Tickets für unsere Tour",
        "Unsere Tour",
        "Werbung",
        "Und jetzt eine kurze Nachricht von unseren Sponsoren.",
        "Eine kleine Botschaft unserer Unterstützer.",
        "Lasst uns kurz über unseren Werbepartner sprechen.",
        "Ein Wort zu unserem Werbepartnern.",
        "Für mehr Informationen, schaut in die Shownotes.",
        "Den Link dazu findet ihr in der Beschreibung.",
        "Besucht unsere Webseite für Tickets und Merchandise.",
        "Unterstützt uns, indem ihr unser Merch kauft.",
        "Unsere Live-Termine findet ihr auf unserer Homepage.",
        "Hier ist eine Empfehlung für euch.",
        "Wir möchten euch etwas vorstellen.",
        "Dieser Inhalt wurde ermöglicht durch Werbepartnern",
        "Es gibt Tickets zu kaufen",
        "Bevor es weiter geht, gibt es eine Werbeeinschaltung"
    ],
    "Outro": [
        "ich glaube, uns bleibt gar nichts mehr zu sagen.",
        "Das war's für heute. Vielen Dank fürs Zuhören, bis zum nächsten Mal!",
        "Damit verabschieden wir uns. Macht's gut und bis zur nächsten Episode!",
        "Wir hoffen, es hat euch gefallen. Wir hören uns in der nächsten Woche wieder.",
        "Schaltet auch das nächste Mal wieder ein, wenn wir neue Geschichten erzählen.",
        "Vergesst nicht zu abonnieren, damit ihr keine Folge verpasst. Auf Wiederhören!",
        "Lernen Sie ein bisschen Geschichte",
        "Lernen Sie ein bisschen Geschichte, dann werden Sie sehen, der Reporter sich damals entwickelt hat."
    ],
    "Inhalt": [
        "Richard, soll ich eine Geschichte erzählen?",
        "Daniel, soll ich eine Geschichte erzählen?",
        "Dann würde ich sagen, Richard, springen wir zur ersten Geschichte, die du vorlesen wirst",
        "Dann würde ich sagen, Daniel, springen wir zur ersten Geschichte, die du vorlesen wirst",
        "Na gut, dann walte deines Amtes und erzähl eine Geschichte, denn letztes Mal habe ich erzählt und jetzt solltest du auch was erzählen",
        "Du bist dran. Was hast du für eine Geschichte mitgebracht?",
        "ich habe auch eine Geschichte dabei",
        "Beginnen wir mit der ersten Story des Tages.",
        "Kommen wir nun zu der Geschichte",
        "Richard wird uns jetzt in die Tiefen der Geschichte entführen.",
        "Daniel, deine Erzählung beginnt jetzt.",
        "Und damit zum heutigen Thema.",
        "Also, worum geht's heute, Daniel?",
        "Lasst uns direkt in die Materie eintauchen.",
        "Was hast du uns heute mitgebracht, Richard?",
        "Kommen wir zum Hauptteil der heutigen Episode.",
        "Ich bin gespannt auf deine Ausführungen.",
        "Deine Geschichte wartet schon auf uns.",
        "Bereit für das heutige Thema?",
        "Fangen wir mit dem Kern der Sache an.",
        "Was ist die Story, die du uns heute erzählen möchtest?",
        "Ich höre gespannt zu, Daniel.",
        "Die Geschichte, die ich mitgebracht habe",
        "Die Geschichte handelt um",
        "Die Geschichte startet",
        "Ich erzähle über die Geschichte",
        "Ich beginne mit der Geschichte",
        "Heute sprechen wir über...",
        "Springen wir direkt in die Thematik.",
        "Mein Thema für heute ist...",
        "Lasst uns die heutige Story aufrollen.",
        "Ich hab da was vorbereitet.",
        "Es geht los mit der Erzählung von...",
        "Widmen wir uns der heutigen Hauptstory.",
        "Ich starte dann mal mit meiner Geschichte.",
        "Hier kommt die Story des Tages.",
        "Also, ab in die Geschichte!",
        "Was verbirgt sich hinter dem heutigen Thema?",
        "Die spannende Geschichte beginnt jetzt.",
        "erzähl bitte noch die Geschichte",
        "War das noch nicht die Geschichte?",
        "Mir fällt eine Geschichte ein",
        "Ich habe eine Geschichte mitgebracht",
        "Mir fällt eine Geschichte ein",
        "Ich habe eine Geschichte mitgebracht",
        "Ich wollte schon immer mal die Geschichte erzählen von",
        "Da fällt mir eine weitere Anekdote ein",
        "Da fällt mir eine weitere Begebenheit ein",
        "Da fällt mir eine weitere Story ein",
        "Eine ganz besondere Geschichte die ich heute erzählen will handelt von",
        "Ich habe da etwas für euch vorbereitet eine kleine Geschichte",
        "Was ich dir noch nicht erzählt habe ist die Geschichte von",
        "Lass mich dir eine Geschichte erzählen",
        "Heute tauchen wir ein in die Geschichte von",
        "Ich habe eine wirklich spannende Geschichte dabei",
        "Ich habe eine wirklich interessante Geschichte dabei",
        "Mir ist da letztens eine Geschichte über den Weg gelaufen die ich unbedingt teilen muss",
        "Kennst du eigentlich die Geschichte über",
        "Bin ich heute mir der Geschichte dran?",
        "Erzählst du heute eine Geschihte?",
        "Worüber handelt deine Geschichte?",
        "Mir fällt eine Geschichte ein",
        "Mir fällt tatsächliche eine Geschichte ein",
        "Exkurs in die Geschichte",
        "Ich habe eine Geschichte mitgebracht und eine Einstiegsfrage dazu",
        "Leg los mit deiner Geschichte"
    ],
    "Diskussion": [
        "Und das war eigentlich die Geschichte.",
        "Vielen Dank für die Geschichte",
        "Wie bist du darauf gestossen?",
        "Was denkst du darüber?",
        "Das ist ja spannend, da muss ich direkt nachhaken.",
        "Gibt es dazu noch weitere Informationen?",
        "Wie schätzt du diese Entwicklung ein?",
        "Das führt zu einer interessanten Frage:",
        "Was ist deine Meinung zu diesem Punkt?",
        "Sehr spannende Geschichte",
        "Ich habe nicht damit gerechnet, dass die Geschichte so verläuft",
        "ich hatte keine Ahnung bisher davon",
        "Spannend, was du am Ende zu der Geschichte gesagt hast",
        "Ich fande die Geschichte auch sehr faszinierend",
        "Ich fand die Quellen faszinierend",
        "Spannende Geschichte, Richard",
        "Spannende Geschichte, Daniel",
        "Was meinst du dazu?",
        "Gibt es noch etwas, das wir besprechen sollten?",
        "Welche Schlussfolgerungen ziehst du daraus?",
        "Das wirft die Frage auf...",
        "Wie bewertest du diese Aspekte?",
        "Eine wirklich interessante Wendung.",
        "Das hab ich so noch nie gehört.",
        "Da muss ich kurz einhaken.",
        "Was ist deine Einschätzung?",
        "Lass uns das mal genauer beleuchten.",
        "Das regt zum Nachdenken an.",
        "Haben wir noch offene Punkte zu diesem Thema?",
        "Gibt es da weitere Erkenntnisse?",
        "Das ist ein guter Punkt.",
        "Absolut faszinierender Einblick in deine Geschichte.",
        "Mich hat besonders beeindruckt, dass...",
        "Was ist deine Perspektive darauf?",
        "Das gibt mir zu denken."
        "Sehr spannend, was ich mich als erstes dazu frage",
        "Ja, Richard, hast du diese Geschichte noch was hinzuzufügen oder sollen wir… Nein, ich glaube, wir können weitergehen zum nächsten Teil dieser Folge, und zwar zum Feedback-Hinweis-Blog",
        "Ja, Daniel, hast du diese Geschichte noch was hinzuzufügen oder sollen wir… Nein, ich glaube, wir können weitergehen zum nächsten Teil dieser Folge, und zwar zum Feedback-Hinweis-Blog"
    ],
    "Danksagung": [
        "Das heisst, danke, die uns da fördern. Ja, in dem Fall, danke fürs Zuhören. Ja, danke.",
        "nur deshalb noch, nicht nur weil Richard ihn mit mir macht, sondern auch weil wir so viel Feedback bekommen und weil er getragen wird von ganz vielen, die ihn anhören. Ohne Leute, die unseren Podcast hören, wird es nichts bringen. Ganz genau. Würde man auch nicht machen. Also vielen, vielen Dank. Wir sind unheimlich dankbar und hoffen, dass wir uns spätestens auf der Tour sehen",
        "Vielen Dank fürs Einschalten und eure Unterstützung!",
        "Ein riesiges Dankeschön an alle unsere Hörerinnen und Hörer!",
        "Wir bedanken uns ganz herzlich für eure Treue.",
        "Danke an alle, die uns bis hierher gefolgt sind!",
        "Wir bedanken uns an folgende Personen: ",
        "Herzlichen Dank an euch alle!",
        "Wir wissen eure Unterstützung sehr zu schätzen.",
        "Euer Support bedeutet uns viel.",
        "Ein großes Dankeschön geht an...",
        "Wir sind sehr dankbar für eure Treue.",
        "Vielen, vielen Dank für alles!",
        "Ohne euch wäre das nicht möglich.",
        "Danke, dass ihr dabei wart!",
        "Wir möchten uns herzlich bedanken.",
        "Großen Dank an unsere Zuhörer.",
        "Danke für eure Aufmerksamkeit.",
        "Danke für die Unterstützung."
    ],
    "Schlussgag": [
        "dann bleibt uns nichts anderes mehr als einem das letzte Wort zu geben. Bruno Kreisky.",
        "Eröffnet durch den, der auch immer alles abschliesst, nämlich the one and only Bruno Kreisky.",
        "dann bleibt uns nichts anderes mehr als einem das letzte Wort zu geben. Bruno Kreisky. Wie der Report sich damals entwickelt hat.",
        "Eröffnet durch den, der auch immer alles abschliesst, nämlich the one and only Bruno Kreisky. Wie der Report sich damals entwickelt hat."
        "Und wie immer zum Abschluss: ",
        "Damit ist das letzte Wort gesprochen, bis zum nächsten Mal!",
        "Das letzte Wort hat  Bruno Kreisky",
        "Und zum Schluss noch ein Gedanke von Bruno Kreisky",
        "Wir verabschieden uns mit den Worten von  Bruno Kreisky",
        "Und damit übergeben wir an  Bruno Kreisky",
        "Der letzte Satz gehört heute  Bruno Kreisky",
        "Mit diesen Worten beenden wir die Folge.",
        "Zum Abschluss noch ein Zitat von Bruno Kreisky",
        "Das war's für heute, mit einem Augenzwinkern von Bruno Kreisky",
        "Bevor wir ganz Schluss machen, hören wir noch Bruno Kreisky",
        "Bruno Kreisky",
        "Dem einen das Wort Geben, der es immer hat, Bruno Kreisky"
    ]
    }

    # Ermittle den Pfad zum aktuellen Skript-Verzeichnis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Definiere den Ordnernamen für die ChromaDB-Daten
    db_directory = os.path.join(script_dir, "chroma_db_data")

    # Initialisiere den ChromaDB Client für persistente Speicherung
    client = chromadb.PersistentClient(path=db_directory)

    # Definiere die Embedding-Funktion
    # Dies lädt das Modell herunter, wenn es noch nicht lokal vorhanden ist.
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # Erstelle oder hole eine Collection UND GIB DIE EMBEDDING-FUNKTION AN
    collection_name = "podcast_segments"
    # WICHTIG: Wenn die Collection bereits mit einer ANDEREN Embedding-Funktion erstellt wurde,
    # MUSS sie gelöscht und neu erstellt werden, damit die neue Funktion angewendet wird.
    # Oder eine neue Collection mit einem anderen Namen.
    try:
        # Versuche, die Collection zu holen
        collection = client.get_collection(name=collection_name)
        
        print(f"Collection '{collection_name}' existiert bereits. Stelle sicher, dass sie mit '{EMBEDDING_MODEL_NAME}' erstellt wurde.")
    except:
        # Wenn die Collection nicht existiert, erstelle sie neu
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )


    # Bereite die Daten für ChromaDB vor
    documents = []
    metadatas = []
    ids = []
    id_counter = 0

    for category, texts in data_to_embed.items():
        for text in texts:
            documents.append(text)
            metadatas.append({"category": category})
            ids.append(f"doc_{id_counter}")
            id_counter += 1

    # Füge die Dokumente zur Collection hinzu
    # ChromaDB wird die Embeddings hier erstellen, da eine embedding_function in der Collection definiert ist.
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"ChromaDB-Collection '{collection_name}' wurde erfolgreich erstellt und mit {len(documents)} Dokumenten befüllt.")
    print(f"Anzahl der Dokumente in der Collection: {collection.count()}")

    # Beispielabfrage: Finde ähnliche Segmente zu einem bestimmten Text
    print("\n--- Beispielabfrage ---")
    query_text = "Vielen Dank für's Zuhören!"
    results = collection.query(
        query_texts=[query_text],
        n_results=2
    )

    print(f"Ähnlichste Segmente zu '{query_text}':")
    for i in range(len(results['documents'][0])):
        print(f"- Dokument: '{results['documents'][0][i]}'")
        print(f"  Kategorie: {results['metadatas'][0][i]['category']}")
        print(f"  ID: {results['ids'][0][i]}")
        print("-" * 20)


def classify_text_similarity_with_chromadb(new_segments_to_classify, similarity_threshold=0.8):
    """
    Classifies new text segments based on similarity to categories in an existing ChromaDB database.
    Segments with a similarity distance exceeding the threshold are marked as 'Unclassified'.
    Uses the specified SentenceTransformer model for embeddings.

    Args:
        new_segments_to_classify (list): A list of dictionaries with 'start_time',
                                          'end_time', and 'text' for the segments to classify.
        similarity_threshold (float): The maximum distance value for a segment to be
                                      considered classified. Smaller values mean higher similarity.
                                      A good starting point for cosine distances with sentence embeddings
                                      is often between 0.6 and 0.8, but requires fine-tuning.
    Returns:
        list: A list of dictionaries containing the classified segments with timestamps,
              text, and the assigned category.
    """

    print("NewSEGMENTS")
    print(new_segments_to_classify)

    # Determine the path to the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the folder name for ChromaDB data (must be the same as when created)
    db_directory = os.path.join(script_dir, "chroma_db_data")

    classified_results_summary = []

    try:
        # Initialize the ChromaDB client and connect to the persistent database
        client = chromadb.PersistentClient(path=db_directory)
        collection_name = "podcast_segments"

        # Definiere die Embedding-Funktion, die auch für die Abfrage verwendet wird
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

        # Hole die Collection mit der definierten Embedding-Funktion
        # Die Collection MUSS mit der GLEICHEN Embedding-Funktion erstellt worden sein.
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )


        if collection.count() == 0:
            print("Error: The ChromaDB collection is empty. Please ensure the database was populated with 'create_chroma_db_with_embeddings()'.")
            return []

        print(f"Successfully connected to ChromaDB collection '{collection_name}'. {collection.count()} documents found.")
        print(f"Using similarity threshold (distance): {similarity_threshold:.4f}")
        print(f"Using Embedding Model: {EMBEDDING_MODEL_NAME}")

        print("\n--- Classifying new text segments ---")

        overall_time : float = 0.0
        first_segment : bool = True

        for segment in new_segments_to_classify:

            segment_text = segment['text']
            start_time = segment['start_time']
            end_time = segment['end_time'] 
            
            # IMPORTANT: Handle NoneType for timestamps before calculating length
            if start_time is None or end_time is None:
                classified_results_summary.append({
                    'original Segment': segment_text, # Behalte original Segment Text
                    'start_time': overall_time,
                    'end_time': abs(end_time- overall_time),
                    'segment_text': segment_text,
                    'segment_length': "N/A (Missing Timestamp)",
                    'classified_category': "Unklassifiziert (Fehlende Zeitstempel)",
                    'similarity_distance': "N/A",
                    'most_similar_known_text_snippet': "N/A"
                })
                continue # Skip to the next segment

            segment_length = abs(end_time - start_time)
            overall_time += segment_length
            # Query the database for similar segments
            # Die query_texts werden hier mit der in der Collection definierten embedding_function eingebettet.
            results = collection.query(
                query_texts=[segment_text],
                n_results=1,
                include=['metadatas', 'documents', 'distances']
            )

            current_classification = "Unklassifiziert"
            most_similar_known_text = None
            similarity_distance = None

            if results['documents'] and results['documents'][0]:
                most_similar_doc = results['documents'][0][0]
                most_similar_category = results['metadatas'][0][0]['category']
                similarity_distance = results['distances'][0][0]

                # Logic with the threshold
                if similarity_distance <= similarity_threshold:
                    current_classification = most_similar_category
                    most_similar_known_text = most_similar_doc
                else:
                    current_classification = "Unklassifiziert (Distance too high)"
                    most_similar_known_text = most_similar_doc
                    continue
                    # Wenn nicht klassifiziert, dann braucht es auch kein 'continue',
                    # da es trotzdem in die results_summary aufgenommen werden soll,
                    # aber eben mit der "Unklassifiziert"-Markierung.

            if first_segment:
                classified_results_summary.append({
                    'original Segment': segment_text, # Statt segment['text'] direkt segment_text nutzen
                    'start_time': start_time,
                    'end_time': overall_time, 
                    'segment_text': segment_text,
                    'segment_length': f"{segment_length:.2f}s",
                    'Overall time' : f"{overall_time:.2f}",
                    'classified_category': current_classification,
                    'similarity_distance': f"{similarity_distance:.4f}" if similarity_distance is not None else "N/A",
                    'most_similar_known_text_snippet': f'"{most_similar_known_text[:50]}..."' if most_similar_known_text else "N/A"
                })
                first_segment = False
            else: 
                classified_results_summary.append({
                    'original Segment': segment_text, # Statt segment['text'] direkt segment_text nutzen
                    'start_time': overall_time - segment_length,
                    'end_time': overall_time,
                    'segment_text': segment_text,
                    'segment_length': f"{segment_length:.2f}s",
                    'Overall time' : f"{overall_time:.2f}",
                    'classified_category': current_classification,
                    'similarity_distance': f"{similarity_distance:.4f}" if similarity_distance is not None else "N/A",
                    'most_similar_known_text_snippet': f'"{most_similar_known_text[:50]}..."' if most_similar_known_text else "N/A"
                })

        return classified_results_summary

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the ChromaDB database exists and is populated in the 'chroma_db_data' folder.")
        # Wenn ein Fehler auftritt, leere Liste zurückgeben, um Abstürze zu vermeiden
        return []
"""
if __name__ == "__main__":
    # Die neuen Textsegmente, die klassifiziert werden sollen
    new_segments = [
        {'start_time': 0.0, 'end_time': 25.48, 'text': " Hallo liebe Hörerinnen und Hörer, wir melden uns kurz aus der Zukunft. Seit Folge 270 heisst dieser Podcast Geschichten aus der Geschichte. Mehr dazu findet ihr unter Geschichte.fm. Und jetzt geht's zurück in die Vergangenheit. Viel Spass. Lernt ein bisschen Geschichte, dann werdet ihr sehen, wie der Reporter sich damals entwickelt hat."},
        {'start_time': 27.12, 'end_time': 29.24, 'text': ' Wie der sich damals entwickelt hat.'},
        {'start_time': 35.62, 'end_time': 36.04, 'text': ' Hallo und herzlich willkommen bei Zeitsprung Geschichten aus der Geschichte.'},
        {'start_time': 37.38, 'end_time': 37.8, 'text': ' Mein Name ist Richard.'},
        {'start_time': 38.64, 'end_time': 40.26, 'text': ' Und ich bin Daniel.'},
        {'start_time': 86.32, 'end_time': 86.8, 'text': ' Ja Daniel, du hast uns jetzt so kurz vor Weihnachten eine Geschichte mitgebracht. Hat sie was mit Weihnachten zu tun? Na, die Geschichte, die ich mitgebracht habe, war vielleicht nicht unbedingt, aber ich möchte gerne mit einer Frage starten. Oh, sehr gut. Sag mal, bist du sehr beschäftigt mit dem Beantworten von Weihnachtsgrüssen, von Neujahrsgrüssen, bist du sehr beschäftigt im Schreiben von Weihnachtsgrüssen und Neujahrsgrüssen? Nein. Ist auch vielleicht gut so, wenn du nämlich Beamter wärst, wenn du österreichischer Beamter wärst, dann hättest du vielleicht sogar ein Problem, wenn du zu viele Weihnachtsgrüsse verschicken würdest. Erklär das bitte genauer, Daniel.'},
        {'start_time': 93.82, 'end_time': 94.34, 'text': ' Ich werde es nachher genauer erklären. Zunächst mal werden wir uns den Julius Raab anhören,'},
        {'start_time': 1000.0, 'end_time': 1005.0, 'text': 'Vielen Dank für das Zuhören und bis zum nächsten Mal!'}
    ]

    # Ruf die Klassifizierungsfunktion auf. Du kannst den Schwellenwert anpassen.
    # Ein höherer Wert lässt mehr Klassifikationen zu, ein niedrigerer ist strenger.
    # Der Standardwert 0.8 ist ein guter Startpunkt für Cosinus-Distanzen.
    final_classified_segments = classify_text_similarity_with_chromadb(new_segments, similarity_threshold=0.9)

    # Hier kannst du weiter mit final_classified_segments arbeiten,
    # z.B. nur die relevanten Informationen speichern oder anzeigen.
    print("\n--- Zusammenfassung der finalen Klassifizierung (nur relevante Felder) ---")
    for segment in final_classified_segments:
        print(f"Timestamp: ({segment['start_time']}s, {segment['end_time']}s), Länge: {segment['segment_length']}, Kategorie: {segment['classified_category']}")
"""
if __name__ == "__main__":
    create_chroma_db_with_embeddings()

#if __name__ == "__main__":

#    save_embeddings()
