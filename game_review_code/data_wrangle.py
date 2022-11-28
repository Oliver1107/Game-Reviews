"""Wrangle review data."""


import pandas as pd
import numpy as np


df = pd.read_csv('game_data/original_reviews.csv', )
df.drop('Unnamed: 0', axis=1, inplace=True)


for col in df:
    df[col] = df[col].apply(lambda x: np.NaN if x == 'None' else x)

df['Release Date'][12719] = 'Aug 19, 2015'
df['Release Date'][17601] = 'June 23, 2008'
df['Release Date'] = df['Release Date'].apply(pd.to_datetime)


df.dropna(subset=['Summary', 'Rating'], inplace=True)

drop = df[(df['Rating'] == 'K-A') |
          (df['Rating'] == 'AO') |
          (df['Rating'] == 'RP')].index

df.drop(drop, inplace=True)


df['Developer'][655] = 'Out Of The Park Developments'
df['Developer'][7036] = 'Electronic Arts'
df['Developer'][8540] = 'FireFly Studios'
df['Developer'][9269] = 'Red Barrels'
df['Developer'][10758] = 'Activision'
df['Developer'][12041] = 'Konami'
df['Developer'][12901] = 'SCEA'
df['Developer'][13038] = 'Konami'
df['Developer'][15532] = 'Midway'
df['Developer'][15993] = 'Crave'
df['Developer'][16435] = 'LucasArts'
df['Developer'][17307] = 'Black Bean Games'
df['Developer'][17728] = 'Square Enix'
df['Developer'][17927] = 'Tetric Online, Inc'
df['Developer'][18151] = 'Microsoft Game Studios'
df['Developer'][18648] = 'Game Factory Interactive'
df['Developer'][19135] = 'Electronic Arts'
df['Developer'][19287] = 'No Reply Games'
df['Developer'][19658] = 'WWO Partners Ltd'


df['Platform'] = df['Platform'].apply(
    lambda x: x.replace('\n', '').replace('\t', '').strip()
)


df['Genre(s)'] = df['Genre(s)'].apply(
    lambda x: x.replace('\r\n', '').strip()
)

df['Genre(s)'] = df['Genre(s)'].str.replace(
    '                                            ', ''
)


drop = df['Number of Players'][df['Number of Players'].apply(
    lambda x: x.__contains__(' 1'))].index

df.drop(drop, inplace=True)
drop = df['Number of Players'][
    df['Number of Players'] == 'Number of Players:'].index

df.drop(drop, inplace=True)
local = ['1 Player', '1-2 Players', '1-3 Players',
         '1-4 Players', '1-5 Players', '1-6 Players']
df['Number of Players'] = df['Number of Players'].apply(
    lambda x: x if x in local else '+'
)


df['Number of Online Players'] = df['Number of Online Players'].fillna('No')
df['Number of Online Players'] = df['Number of Online Players'].apply(
    lambda x: 0 if x.__contains__('No') else 1
)

df.rename(columns={'Number of Online Players': 'Online Multiplayer'},
          inplace=True)
for i in range(len(df)):
    if df['Number of Players'].iloc[i] == '+':
        df['Online Multiplayer'].iloc[i] = 1


df.drop(columns=['ESRB Descriptors'], inplace=True)


drop = df[df['User Score'] == 'tbd'].index
df.drop(drop, inplace=True)
df['User Score'] = df['User Score'].apply(float)

df['Metascore'][17290] = '56'
df['Metascore'] = df['Metascore'].apply(int)


for game in df['Title'].value_counts()[df['Title'].value_counts() > 1].index:
    game_df = df[df['Title'] == game]
    title = game
    release = game_df['Release Date'].min()
    summary = game_df['Summary'].value_counts().index[0]
    rating = game_df['Rating'].value_counts().index[0]
    developer = game_df['Developer'].value_counts().index[0]
    platform = ', '.join(game_df['Platform'].unique())
    genres = []
    for genre in game_df['Genre(s)']:
        gs = genre.split(',')
        for g in gs:
            genres.append(g)
    genres = ','.join(genres)
    num_players = game_df['Number of Players'].value_counts().index[0]
    online = game_df['Online Multiplayer'].max()
    metascore = round(game_df['Metascore'].mean())
    userscore = round(game_df['User Score'].mean(), 1)
    new_row = pd.DataFrame(index=[game_df.index[0]],
                           data={'Title': title,
                                 'Release Date': release,
                                 'Summary': summary,
                                 'Rating': rating,
                                 'Developer': developer,
                                 'Platform': platform,
                                 'Genre(s)': genres,
                                 'Number of Players': num_players,
                                 'Online Multiplayer': online,
                                 'Metascore': metascore,
                                 'User Score': userscore})
    drop = game_df.index
    df.drop(drop, inplace=True)
    df = pd.concat([df, new_row])


df['Release Month'] = df['Release Date'].dt.month
df['Release Day of Month'] = df['Release Date'].dt.day
df['Release Year'] = df['Release Date'].dt.year
df.drop(columns=['Release Date'], inplace=True)


df['Genre(s)'] = df['Genre(s)'].apply(
    lambda x: x.replace(',,', ',').rstrip(',')
)

transform = {'Skateboarding': 'Sports',
             'Skate / Skateboard': 'Sports',
             'Skating': 'Sports',
             'Snowboarding': 'Sports',
             'Skiing': 'Sports',
             'Ski / Snowboard': 'Sports',
             'Surfing': 'Sports',
             'Wakeboarding': 'Sports',
             'Surf / Wakeboard': 'Sports',
             'Soccer': 'Sports',
             'Ice Hockey': 'Sports',
             'Football': 'Sports',
             'Baseball': 'Sports',
             'Tennis': 'Sports',
             'Basketball': 'Sports',
             'Rugby': 'Sports',
             'Golf': 'Sports',
             'Cricket': 'Sports',
             'Biking': 'Sports',
             'Boxing': 'Sports',
             'Boxing / Martial Arts': 'Sports',
             'Volleyball': 'Sports',
             'Bowling': 'Sports',
             'Olympic Sports': 'Sports',
             'Athletics': 'Sports',
             'Wrestling': 'Fighting',
             "Beat-'Em-Up": 'Fighting',
             'Car Combat': 'Driving / Racing,Combat',
             'Artillery': 'Shooter',
             "Shoot-'Em-Up": 'Shooter',
             'Light Gun': 'Shooter',
             'Military': 'Shooter',
             'WWII': 'Shooter',
             'Wargame': 'Shooter',
             'WWI': 'Shooter',
             'War': 'Shooter',
             'Motocross': 'Driving / Racing',
             'Demolition Derby': 'Driving / Racing,Combat',
             'Formula One': 'Driving / Racing',
             'Drag': 'Driving / Racing',
             'Street': 'Driving / Racing',
             'Stock Car': 'Driving / Racing',
             'GT / Street': 'Driving / Racing',
             'Driving': 'Driving / Racing',
             'Racing': 'Driving / Racing',
             'Motorcycle': 'Driving / Racing',
             'Driving': 'Driving / Racing',
             'Kart': 'Driving / Racing',
             'Rally / Offroad': 'Driving / Racing',
             'Racing': 'Driving / Racing',
             'Logic': 'Puzzle',
             'Sandbox': 'Open-World',
             'Sim': 'Simulation',
             'Virtual Life': 'Simulation',
             'Virtual': 'Simulation',
             'Action RPG': 'RPG',
             'Console-style RPG': 'RPG',
             'Role-Playing': 'RPG',
             'PC-style RPG': 'RPG',
             'Business / Tycoon': 'Tycoon',
             'Music Maker': 'Music',
             'Rhythm': 'Music',
             'Dancing': 'Music',
             'Massively Multiplayer': 'Massively Multiplayer Online',
             'Helicopter': 'Flight',
             'Old Jet': 'Flight',
             'Modern Jet': 'Flight',
             'Civilian Plane': 'Flight',
             'Futuristic Jet': 'Flight',
             'Party': 'Party / Minigame',
             'Billiards': 'Board / Card Game',
             'Card Battle': 'Board / Card Game',
             'Board Games': 'Board / Card Game',
             'Tactical': 'Tactical',
             'Tactics': 'Tactical',
             'Pinball': 'Arcade',
             'Action Adventure': 'Action,Adventure'}

new_genres = ['Sports', 'Fighting', 'Combat', 'Shooter', 'Driving / Racing',
              'Puzzle', 'Open-World', 'Simulation', 'RPG', 'Tycoon',
              'Music', 'Massively Multiplayer Online', 'Flight', 'Action',
              'Party / Minigame', 'Board / Card Game', 'Tactical', 'Arcade',
              'Platformer', 'Sci-Fi', 'Horror', 'Adventure', 'Survival',
              'Exercise / Fitness', 'Strategy', 'Trivia / Game Show', 'Other',
              'Third-Person', 'Gambling', 'Fantasy', 'First-Person']

all_platforms = set()
for plat in df['Platform'].unique():
    s = plat.split(', ')
    for plat in s:
        all_platforms.add(plat)


genre_counts = {}
for genres in df['Genre(s)']:
    genre_list = genres.split(',')
    for genre in genre_list:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1
new_genres = []
other = []
for genre in genre_counts:
    if genre_counts[genre] > 800:
        new_genres.append(genre)
    else:
        other.append(genre)
new_genres.append('Other')


def convert_genre(s):
    """Convert less frequent genres to 'Other'."""
    old = s.split(',')
    new = []
    for genre in old:
        if genre in other:
            new.append('Other')
        else:
            new.append(genre)
    return ','.join(new)


def one_hot_encode(col, names):
    """One hot encode a column with multiple variables per row."""
    one_hot = pd.DataFrame(index=df.index, columns=names)
    for name in names:
        one_hot[name] = df[col].apply(
            lambda x: int(name in x.split(','))
        )
    new_df = pd.concat([df, one_hot], axis=1).drop(columns=[col])
    return new_df


df['Genre(s)'] = df['Genre(s)'].apply(convert_genre)
df = one_hot_encode('Genre(s)', new_genres)

# df = one_hot_encode('Platform', list(all_platforms))


df['Developer'] = df['Developer'].apply(
    lambda x: 'EA' if (x.__contains__('Electronic Arts')) else x)
df['Developer'] = df['Developer'].apply(
    lambda x: x.split(', ')[0])
same = {}
ignore = ['Happy', 'Third', 'Secret', 'White', 'Pixel', 'Atomic', 'No', 'Mad',
          'Steel', 'High', 'Infinite', 'Blue', 'Magic', 'Double', 'The',
          'Deep', 'Flying', 'Red', 'New', 'Massive', 'Big', 'Studio', 'Black',
          'Digital', 'Team', 'Game', 'Creative', 'Ghost', 'Night', 'Lucky',
          'Games', 'Hidden', 'Spiral', 'We', 'Super', 'Silver', 'Matt', 'Cold',
          'Awesome', 'La', 'Tom', 'System', 'Art', 'Three', 'Liquid', 'Player',
          'Image', 'Visual', 'Retro', 'Dan', 'Stage', 'Lab', 'Data', 'Deck',
          'Plastic', 'Frozen', 'Mega', 'Sand', 'Running', 'Santa', 'Upper',
          'Wide', 'Heavy', 'Chasing', 'Dark', 'Foam', 'Lunar', 'Micro',
          'Purple', 'Fun', 'Phantom', 'Virtual', 'Wish', 'Draw', '3D',
          'Modern', 'Infinity', 'Sports', 'Sonic', 'Left', 'Next', 'Monster',
          'Raw', 'Terrible', 'Neon', 'Artifact', 'Full', 'Vivid', 'Idol',
          'Just', 'Little', 'Acid', 'Lazy', 'Humble', 'House', 'Boss', 'Aqua',
          'One', 'Nine', 'Pacific', 'Shadow', 'Artificial', 'Will', 'Onion',
          'Glass', 'Vertex', 'Daniel', 'Heart', 'Panic', 'Radical', 'Rogue',
          'United', 'Light', 'Human', 'Realtime', 'Parallel', 'Fallen',
          'Right', 'Paul', 'Over', 'Two', 'Out', 'David', 'Free', 'Strange',
          'Pocket', 'Stainless', 'Piranha', 'Cardboard', 'Void', 'Project',
          'Revolution', 'Mass', 'Iron', 'Polygon', 'Good', 'Code', 'Giant',
          'Noise', 'Ninja']
for dev in df['Developer'].unique():
    f = dev.split(' ')[0]
    if f not in same:
        s = []
        for d in df['Developer'].unique():
            if d.split(' ')[0] == f:
                s.append(d)
        if (len(s) > 2) and (f not in ignore):
            same[f] = s
df['Developer'] = df['Developer'].apply(
    lambda x: x.split(' ')[0] if x.split(' ')[0] in same else x)
outlier = df['Developer'].value_counts()[
    df['Developer'].value_counts() < 5].index
drop = df['Developer'][df['Developer'].apply(lambda x: x in outlier)].index
df.drop(drop, inplace=True)
other = df['Developer'].value_counts()[
    df['Developer'].value_counts() < 6].index
df['Developer'] = df['Developer'].apply(lambda x: 'Other' if x in other else x)


df.to_csv('game_data/wrangled_reviews.csv')
