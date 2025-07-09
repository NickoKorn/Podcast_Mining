import requests
import shutil
import feedparser
import re
import os
import json
import time

# Download the MP3 files

#titels_and_summaries = dict()
current_episode_count_with_timestamp = dict()

def create_current_episode_count_json():

    global current_episode_count_with_timestamp
    current_episodes_count = get_episode_count()

    # 1. Aktuellen Unix-Timestamp abrufen (in Sekunden seit der Epoche)
    #current_unix_timestamp = time.time()
    #print(f"Aktueller Unix-Timestamp (Float): {current_unix_timestamp}")
    # Beispiel-Output: Aktueller Unix-Timestamp (Float): 1718012825.123456

    current_unix_timestamp_int = int(time.time())
    print(f"Aktueller Unix-Timestamp (Integer): {current_unix_timestamp_int}")
    # Beispiel-Output: Aktueller Unix-Timestamp (Integer): 1718012825
    current_episode_count_with_timestamp['latest_episode_with_timestamp'] = {'count': current_episodes_count, 'time': current_unix_timestamp_int}
    #save_json("current_episode_count_with_timestamp.json", current_episode_count_with_timestamp)
    return current_episode_count_with_timestamp

def save_json(file_path_json: str, current_dict: dict):

    #file_path_json = "titles_summaries.json"

    # Daten speichern (serialisieren)
    with open(file_path_json, 'w', encoding='utf-8') as f:
        json.dump(current_dict, f, ensure_ascii=False, indent=4) # indent für Lesbarkeit

    print(f"Daten erfolgreich in '{file_path_json}' gespeichert.")
# Dekorieren Sie Ihre Ladefunktion mit @st.cache_data

def load_json(file_path_json):

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

def downloadChosenEpisode(episodes: list):

    #logic for scraping all the files or adding left ones, so maybe donwload loop until some condition is reached
    #global titels_and_summaries
    url = "https://www.geschichte.fm/feed/mp3/"
    response = requests.head(url) # Nur den Header anfragen
    #print(response.headers.get('Content-Type'))
    feed = feedparser.parse(url)
    print("episodes :")
    print(episodes)

    #print(feed.entries[10]['links'][1]['href'])
    #feed.entries[len(feed.entries)-1]
    for i in range(0, len(episodes), 1):
        
        #print((len(feed.entries)-1))
        #print((len(feed.entries)-1)-needed_episodes_list[i])
        #print(feed.entries[0]['title'])
        link = feed.entries[episodes[i]-1]['links'][1]['href']
        # Hier können Sie dann auf feed.feed.title, feed.entries etc. zugreifen
        #print(f"Podcast-Titel: {feed.feed.title}")
        #for i in range(0, len(episodes), 1):

            #link = str(feed.entries[i]['title']).split(":", 1)[0]

        os.chdir("..")
        os.chdir("audioData")
        r = requests.get(link, verify=False, stream=True)
        r.raw.decode_content = True
        episodes
        if os.path.exists(str(feed.entries[int(episodes[i]-1)]['title']).split(":", 1)[0] + ".mp3"):
        
            print(f"Datei '{str(feed.entries[int(episodes[i]-1)]['title']).split(':', 1)[0] + '.mp3'}' existiert bereits. Überspringe Download.")
        
        else:
            with open(str(feed.entries[int(episodes[i]-1)]['title']).split(":", 1)[0] + ".mp3", 'wb') as f:
                
                shutil.copyfileobj(r.raw, f)

def downloadEpisodesMP3Files():
        
    #url="https://adswizz.podigee-cdn.net/media/podcast_47771_geschichten_aus_der_geschichte_episode_1879019_gag500_kleine_geschichte_eines_jubilaums.mp3?awCollectionId=svo_3857a7&awEpisodeId=1879019&aw_0_1st.age=28-44&aw_0_1st.blocklist=%5B%22hard_alcohol%22%2C%22gambling%22%2C%22erotic%22%2C%22ph_supplements%22%2C%22f_crypto%22%2C%22diet%22%2C%22energy_companies%22%2C%22fast_food%22%2C%22n_water%22%2C%22m_health%22%2C%22vpn_provider%22%2C%22insurance%22%5D&aw_0_1st.gender=maennlich&source=webplayer-download&v=1745363289"    #Note: It's https
    #r = requests.get(url, verify=False, stream=True)
    #r.raw.decode_content = True
    #with open("episode.mp3", 'wb') as f:
    #        shutil.copyfileobj(r.raw, f)

    # Loop with 
    url = "https://adswizz.podigee-cdn.net/version/1743699890/media/podcast_47771_geschichten_aus_der_geschichte_episode_543036_gag07_geteilte_habsburger.mp3"
    r = requests.get(url, verify=False, stream=True)
    r.raw.decode_content = True
    with open("episode7.mp3", 'wb') as f:
            shutil.copyfileobj(r.raw, f)

def getTitle_and_Summary(episodes : list):

    title_and_summary = dict()

    database = False

    if database == False:

        url = "https://www.geschichte.fm/feed/mp3/"
        response = requests.head(url) # Nur den Header anfragen
        #print(response.headers.get('Content-Type'))
        feed = feedparser.parse(url)    
        #print(feed.entries[538].keys())
        #print(str(feed.entries[10]['title']))
        with open("episode" + str(feed.entries[0]['title']).split(":", 1)[0] + ".mp3", 'wb') as f:
            
            shutil.copyfileobj(r.raw, f)

def getTitlesAndDescriptions(episode: list)->dict():

    titels_and_summaries = dict()
    url = "https://www.geschichte.fm/feed/mp3/"
    response = requests.head(url) # Nur den Header anfragen
    #print(response.headers.get('Content-Type'))
    feed = feedparser.parse(url)    
    #print(feed.entries[538].keys())

    for i in range(0, len(episode), 1):

        #title = str(feed.entries[i]['title']).split(":", 1)[0]
        original_title = str(feed.entries[episode[i]-1]['title']) 

        current_summary = feed.entries[episode[i]-1]['summary']

        target_List = ['Aus unserer Werbung', 'Erwähnte Episoden', 'Literatur', 'Erwähnte Folgen', 'Tour', 'AUS UNSERER WERBUNG', '//Erwähnte Folgen', 'Shownotes', 'Themenblöcke']

        #pattern_capture_group1 = fr"(.*?){re.escape(target_name_Episoden)}"
        #pattern_capture_group2 = fr"(.*?){re.escape(target_name_Literatur)}"
        #pattern_capture_group3 = fr"(.*?){re.escape(target_name_Folgen)}"
        #pattern_capture_group4 = fr"(.*?){re.escape(target_name_Tour)}"

        match_List = []

        for j in range(len(target_List)-1, -1, -1):

            match_List.append(re.search(fr"(.*?){re.escape(target_List[j])}", current_summary))

        check_For_Exisitng_Match = False

        for j in range(0, len(match_List), 1):

            if match_List[j]:

                check_For_Exisitng_Match = True
        
        if check_For_Exisitng_Match:

            #print(type(match_obj_1))
            indices_list = [match_item.start() for match_item in match_List if match_item]
            
            min_index = min(indices_list)
            
            #print(min_index)

            #print(current_summary[0:min_index])
            #titels_and_summaries[title]['summary'] = current_summary[0:min_index]
            titels_and_summaries[original_title] = {'Beschreibung': current_summary[0:min_index]}

        else:

            #print(current_summary)
            titels_and_summaries[original_title] = {'Beschreibung': current_summary}
        
    #print(titels_and_summaries)
    return titels_and_summaries
        
def getPossibleTitlesOfPodcast():

    url = "https://www.geschichte.fm/feed/mp3/"
    response = requests.head(url) # Nur den Header anfragen
    #print(response.headers.get('Content-Type'))
    feed = feedparser.parse(url)    
    #print(feed.entries[538].keys())
    for i in range(len(feed.entries)-1, 0, -1):

        #print(str(feed.entries[i]['title']))
        #print(str(feed.entries[i]['title']).split(":", 1)[0])
        title = str(feed.entries[i]['title']).split(":", 1)[0]
        original_title = str(feed.entries[i]['title'])
        episode_number = title[3:len(title)]
        #print(episode_number)
        if not re.match(r"^[0-9]", str(episode_number[0])):

            print(episode_number)
            print(episode_number[0])
            continue
        """
        Dict with follwing structure: '500': 'title', 'summary'
        """
        #titels_and_summaries[title] = str(episode_number)

        #print('Beschreibung: ')
        current_summary = feed.entries[i]['summary']
        #print('OG Beschreibung: ')
        #print(current_summary)

        #print(current_summary)
        #print(current_summary)

        target_List = ['Aus unserer Werbung', 'Erwähnte Episoden', 'Literatur', 'Erwähnte Folgen', 'Tour', 'AUS UNSERER WERBUNG', '//Erwähnte Folgen', 'Shownotes']

        #pattern_capture_group1 = fr"(.*?){re.escape(target_name_Episoden)}"
        #pattern_capture_group2 = fr"(.*?){re.escape(target_name_Literatur)}"
        #pattern_capture_group3 = fr"(.*?){re.escape(target_name_Folgen)}"
        #pattern_capture_group4 = fr"(.*?){re.escape(target_name_Tour)}"

        match_List = []

        for j in range(len(target_List)-1, -1, -1):

            match_List.append(re.search(fr"(.*?){re.escape(target_List[j])}", current_summary))

        check_For_Exisitng_Match = False

        if len(match_List)>0:

            #print('true')
            check_For_Exisitng_Match = True

        if check_For_Exisitng_Match:

            #print(type(match_obj_1))
            indices_list = []
            
            for j in range(0, len(match_List), 1):

                if match_List[j]:

                    indices_list.append(match_List[j].start())
            
            min_index = min(indices_list)

            #print(min_index)

            #print(current_summary[0:min_index])
            #titels_and_summaries[title]['summary'] = current_summary[0:min_index]
            titels_and_summaries[episode_number] = {'title': original_title, 'Beschreibung': current_summary[0:min_index], 'epiosde_index': i}

        else:

            #print(current_summary)
            titels_and_summaries[episode_number] = {'title': original_title, 'Beschreibung': current_summary, 'epiosde_index': i}
        
    #print(titels_and_summaries['09']['summary'])
    save_json("titles_summaries.json", titels_and_summaries)

def make_titles_summaries_json():

    titels_and_summaries = dict()
    url = "https://www.geschichte.fm/feed/mp3/"
    response = requests.head(url) # Nur den Header anfragen
    #print(response.headers.get('Content-Type'))
    feed = feedparser.parse(url)    
    #print(feed.entries[538].keys())

    for i in range(0, get_episode_count(), 1):

        #title = str(feed.entries[i]['title']).split(":", 1)[0]
        original_title = str(feed.entries[i]['title']) 

        current_summary = feed.entries[i]['summary']

        target_List = ['Links zu erwähnten Themen:', 'Aus unserer Werbung', 'Erwähnte Episoden', 'Literatur', 'Erwähnte Folgen', 'Tour', 'AUS UNSERER WERBUNG', '//Erwähnte Folgen', 'Shownotes', 'Themenblöcke']

        #pattern_capture_group1 = fr"(.*?){re.escape(target_name_Episoden)}"
        #pattern_capture_group2 = fr"(.*?){re.escape(target_name_Literatur)}"
        #pattern_capture_group3 = fr"(.*?){re.escape(target_name_Folgen)}"
        #pattern_capture_group4 = fr"(.*?){re.escape(target_name_Tour)}"

        match_List = []

        for j in range(len(target_List)-1, -1, -1):

            match_List.append(re.search(fr"(.*?){re.escape(target_List[j])}", current_summary))

        check_For_Exisitng_Match = False

        for j in range(0, len(match_List), 1):

            if match_List[j]:

                check_For_Exisitng_Match = True
        
        if check_For_Exisitng_Match:

            #print(type(match_obj_1))
            indices_list = [match_item.start() for match_item in match_List if match_item]
            
            min_index = min(indices_list)
            
            #print(min_index)

            #print(current_summary[0:min_index])
            #titels_and_summaries[title]['summary'] = current_summary[0:min_index]
            titels_and_summaries[str(i)] = {'title': original_title, 'Beschreibung': current_summary[0:min_index]}

        else:

            #print(current_summary)
            titels_and_summaries[str(i)] = {'title': original_title, 'Beschreibung': current_summary}
        
    #print(titels_and_summaries)
    save_json("all_titles_summaries.json", titels_and_summaries)

def getPossibleTitlesOfPodcast():

    url = "https://www.geschichte.fm/feed/mp3/"
    response = requests.head(url) # Nur den Header anfragen
    #print(response.headers.get('Content-Type'))
    feed = feedparser.parse(url)    
    #print(feed.entries[538].keys())
    for i in range(len(feed.entries)-1, 0, -1):

        #print(str(feed.entries[i]['title']))
        #print(str(feed.entries[i]['title']).split(":", 1)[0])
        title = str(feed.entries[i]['title']).split(":", 1)[0]
        original_title = str(feed.entries[i]['title'])
        episode_number = title[3:len(title)]
        #print(episode_number)
        if not re.match(r"^[0-9]", str(episode_number[0])):

            print(episode_number)
            print(episode_number[0])
            continue
        """
        Dict with follwing structure: '500': 'title', 'summary'
        """
        #titels_and_summaries[title] = str(episode_number)

        #print('Beschreibung: ')
        current_summary = feed.entries[i]['summary']
        #print('OG Beschreibung: ')
        #print(current_summary)

        #print(current_summary)
        #print(current_summary)

        target_List = ['Aus unserer Werbung', 'Erwähnte Episoden', 'Literatur', 'Erwähnte Folgen', 'Tour', 'AUS UNSERER WERBUNG', '//Erwähnte Folgen', 'Shownotes']

        #pattern_capture_group1 = fr"(.*?){re.escape(target_name_Episoden)}"
        #pattern_capture_group2 = fr"(.*?){re.escape(target_name_Literatur)}"
        #pattern_capture_group3 = fr"(.*?){re.escape(target_name_Folgen)}"
        #pattern_capture_group4 = fr"(.*?){re.escape(target_name_Tour)}"

        match_List = []

        for j in range(len(target_List)-1, -1, -1):

            match_List.append(re.search(fr"(.*?){re.escape(target_List[j])}", current_summary))

        check_For_Exisitng_Match = False

        if len(match_List)>0:

            #print('true')
            check_For_Exisitng_Match = True

        if check_For_Exisitng_Match:

            #print(type(match_obj_1))
            indices_list = []
            
            for j in range(0, len(match_List), 1):

                if match_List[j]:

                    indices_list.append(match_List[j].start())
            
            min_index = min(indices_list)

            #print(min_index)

            #print(current_summary[0:min_index])
            #titels_and_summaries[title]['summary'] = current_summary[0:min_index]
            titels_and_summaries[episode_number] = {'title': original_title, 'Beschreibung': current_summary[0:min_index], 'epiosde_index': i}

        else:

            #print(current_summary)
            titels_and_summaries[episode_number] = {'title': original_title, 'Beschreibung': current_summary, 'epiosde_index': i}
        
    #print(titels_and_summaries['09']['summary'])
    save_json("titles_summaries.json", titels_and_summaries)

def getReferencesFromScraping():

    url = "https://www.geschichte.fm/feed/mp3/"
    response = requests.head(url) # Nur den Header anfragen
    #print(response.headers.get('Content-Type'))
    feed = feedparser.parse(url)    
    print(feed.entries[538].keys())
    for i in range(len(feed.entries)-1, 0, -1):

        print(feed.entries[i]['summary'])

def get_episode_count()->int: 

    url = "https://www.geschichte.fm/feed/mp3/"
    response = requests.head(url) # Nur den Header anfragen
    #print(response.headers.get('Content-Type'))
    feed = feedparser.parse(url)    
    return len(feed.entries)

def getNewestMP3Episodes():

    #logic for scraping all the files or adding left ones, so maybe donwload loop until some condition is reached
    url = "https://www.geschichte.fm/feed/mp3/"
    response = requests.head(url) # Nur den Header anfragen
    #print(response.headers.get('Content-Type'))
    feed = feedparser.parse(url)
    print(str(feed.entries[0]['title']))
    # Hier können Sie dann auf feed.feed.title, feed.entries etc. zugreifen
    #print(f"Podcast-Titel: {feed.feed.title}")
    print(f"Anzahl der Episoden im Feed: {len(feed.entries)}")
    print(feed.entries[0].keys())
    link = feed.entries[0]["links"][1]['href']
    print(link)
    r = requests.get(link, verify=False, stream=True)
    r.raw.decode_content = True
    os.chdir("..")
    os.chdir("audioData")
    print(str(feed.entries[0]['title']).split(":", 1)[0])
    with open("episode" + str(feed.entries[0]['title']).split(":", 1)[0] + ".mp3", 'wb') as f:
            shutil.copyfileobj(r.raw, f)

if __name__ == "__main__":

    make_titles_summaries_json()
    #downloadEpisodesMP3Files()
    #getPossibleTitlesOfPodcast()
    #getNewestMP3Episodes()
    #getTitle_and_Summary([2,3])