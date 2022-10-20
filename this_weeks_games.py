'''
Objective: 
This file contains the function 'get_this_weeks_games', 
which saves a given week's college football schedule to a file. 
This function is used to create the schedule the ML model uses to make predictions for this week. 

'''

# Import Libraries
# Data Manipulation
import pandas as pd

# Web Scrapping
import requests
from bs4 import BeautifulSoup

'''
get_this_weeks_games: gets the entire college football schedule for this month, day, and year.

Writes output of home and away teams for each game to the filepath games_<month>_<day>_<year>.csv.
'''

def get_this_weeks_games(month: int, day: int, year: int) -> pd.DataFrame:
	url = f'https://www.sports-reference.com/cfb/boxscores/index.cgi?month={month}&day={day}&year={year}'

	page = requests.get(url)
	soup = BeautifulSoup(page.text, 'html.parser')

	games = soup.find_all('div', attrs={'class': 'game_summary nohover'})

	home_teams = []
	away_teams = []

	for game in games:
		try:
			table = game.find('table')
			refs = table.find_all('a')
			team_names = [x.get('href').split('/')[3] for x in refs]
			home_team = team_names[2]
			away_team = team_names[0]

			home_teams.append(home_team)
			away_teams.append(away_team)


		except AttributeError:
			continue

	games_this_week = pd.DataFrame({
		'home_teams': home_teams,
		'away_teams': away_teams
	})

	games_this_week.to_csv(f'schedules/games_{month}_{day}_{year}.csv', index=False)

	return games_this_week
