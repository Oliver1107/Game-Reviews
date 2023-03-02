"""Collect the data of game reviews off of MetaCritic."""


from requests import get
from bs4 import BeautifulSoup
import pandas as pd


# retrieve all pages on metacritics game rankings
soups = []
for i in range(198):
    url = 'https://www.metacritic.com/browse/games/score/metascore/all/all/filtered?page=' + f'{i}'
    page = get(
        url,
        headers={
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'User-Agent': "Magic Browser"
            }
        )
    soup = BeautifulSoup(page.content, 'html.parser')
    soups.append(soup)

# retrieve the title of each game
games = []
for soup in soups:
    elements = soup.find_all('a')
    for element in elements:
        if element.get('class') == ['title']:
            games.append(element)

# retrieve the details page of each game
review_urls = []
for game in games:
    url = 'https://www.metacritic.com' + game.get('href') + '/details'
    review_urls.append(url)

# collect the details of each game
game_info = []
for url in review_urls:
    page = get(
        url,
        headers={
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'User-Agent': "Magic Browser"
            }
        )
    soup = BeautifulSoup(page.content, 'html.parser')

    try:
        date = soup.find_all()[([
            soup.find_all().index(element) for element in soup.find_all() if element.text == 'Release Date:'
        ][0] + 1)].text
    except IndexError:
        date = 'None'
    try:
        summary = soup.find_all()[([
            soup.find_all().index(element) for element in soup.find_all() if element.text == 'Summary:'
        ][0] + 1)].text
    except IndexError:
        summary = 'None'
    try:
        developer = soup.find_all()[([
            soup.find_all().index(element) for element in soup.find_all() if element.text == 'Developer:'
        ][0] + 1)].text
    except IndexError:
        developer = 'None'
    try:
        platform = [data.text for data in soup.find_all('span') if data.get('class') == ['platform']][0]
    except IndexError:
        platform = 'None'
    try:
        genres = soup.find_all()[([
            soup.find_all().index(element) for element in soup.find_all() if element.text == 'Genre(s):'
        ][0] + 1)].text
    except IndexError:
        genres = 'None'
    try:
        rating = soup.find_all()[([
            soup.find_all().index(element) for element in soup.find_all() if element.text == 'Rating:'
        ][0] + 1)].text
    except IndexError:
        rating = 'None'
    try:
        players = soup.find_all()[([
            soup.find_all().index(element) for element in soup.find_all() if element.text == 'Number of Players:'
        ][0] + 1)].text
    except IndexError:
        players = '1 Player'
    try:
        online = soup.find_all()[([
            soup.find_all().index(element) for element in soup.find_all() if element.text == 'Number of Online Players:'
        ][0] + 1)].text
    except IndexError:
        online = 'None'
    try:
        esrb = soup.find_all()[([
            soup.find_all().index(element) for element in soup.find_all() if element.text == 'ESRB Descriptors:'
        ][0] + 1)].text
    except IndexError:
        esrb = 'None'
    try:
        metascore = [element.text for element in soup.find_all('span') if element.get('itemprop') == 'ratingValue'][0]
    except IndexError:
        metascore = 'None'
    try:
        userscore = [element.text for element in soup.find_all() if (type(element.get('class')) == list and element.get('class').__contains__('user'))][0]
    except IndexError:
        userscore = 'None'

    info = {
        'Release Date': date,
        'Summary': summary,
        'Rating': rating,
        'Developer': developer,
        'Platform': platform,
        'Genre(s)': genres,
        'Number of Players': players,
        'Number of Online Players': online,
        'ESRB Descriptors': esrb,
        'Metascore': metascore,
        'User Score': userscore
    }
    game_info.append(info)

# create the dataframe of scraped data and save as csv
columns = {
    'Title': [game.text for game in games],
    'Release Date': [game['Release Date'] for game in game_info],
    'Summary': [game['Summary'] for game in game_info],
    'Rating': [game['Rating'] for game in game_info],
    'Developer': [game['Developer'] for game in game_info],
    'Platform': [game['Platform'] for game in game_info],
    'Genre(s)': [game['Genre(s)'] for game in game_info],
    'Number of Players': [game['Number of Players'] for game in game_info],
    'Number of Online Players': [game['Number of Online Players'] for game in game_info],
    'ESRB Descriptors': [game['ESRB Descriptors'] for game in game_info],
    'Metascore': [game['Metascore'] for game in game_info],
    'User Score': [game['User Score'] for game in game_info]
}
df = pd.DataFrame(data=columns)
df.to_csv('game_data/original_reviews.csv')
