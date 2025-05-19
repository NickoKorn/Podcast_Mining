import requests
import shutil

# Download the MP3 files

def downloadEpisodesMP3Files():
        
    url="https://adswizz.podigee-cdn.net/media/podcast_47771_geschichten_aus_der_geschichte_episode_1879019_gag500_kleine_geschichte_eines_jubilaums.mp3?awCollectionId=svo_3857a7&awEpisodeId=1879019&aw_0_1st.age=28-44&aw_0_1st.blocklist=%5B%22hard_alcohol%22%2C%22gambling%22%2C%22erotic%22%2C%22ph_supplements%22%2C%22f_crypto%22%2C%22diet%22%2C%22energy_companies%22%2C%22fast_food%22%2C%22n_water%22%2C%22m_health%22%2C%22vpn_provider%22%2C%22insurance%22%5D&aw_0_1st.gender=maennlich&source=webplayer-download&v=1745363289"    #Note: It's https
    r = requests.get(url, auth=('usrname', 'password'), verify=False,stream=True)
    r.raw.decode_content = True
    with open("episode.mp3", 'wb') as f:
            shutil.copyfileobj(r.raw, f)

def getNewestMP3Episodes():

    #logic for scraping all the files or adding left ones, so maybe donwload loop until some condition is reached
    pass

if __name__ == "__main__":

    downloadEpisodesMP3Files()