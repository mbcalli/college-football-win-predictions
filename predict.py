# Date
import datetime

# Data Manipulation
import pandas as pd

# Schedule Retrieval
from this_weeks_games import get_this_weeks_games

# Model Loading
import joblib

def get_game_data(home_team: str, away_team: str, offense_df: pd.DataFrame, defense_df: pd.DataFrame) -> pd.DataFrame:
	home_offense = offense_df.query('team == @home_team')[[x for x in offense_df.columns if 'remove' not in x and x != 'team']].astype('float').reset_index(drop=True)
	home_defense = defense_df.query('team == @home_team')[[x for x in offense_df.columns if 'remove' not in x and x != 'team']].astype('float').reset_index(drop=True)
	away_offense = offense_df.query('team == @away_team')[[x for x in offense_df.columns if 'remove' not in x and x != 'team']].astype('float').reset_index(drop=True)
	away_defense = defense_df.query('team == @away_team')[[x for x in offense_df.columns if 'remove' not in x and x != 'team']].astype('float').reset_index(drop=True)

	game_df = pd.merge(home_offense, home_defense, left_index=True, right_index=True, suffixes=('_home_off', '_home_def'))
	game_df = pd.merge(game_df, away_offense, left_index=True, right_index=True, suffixes=('', '_away_off'))
	game_df = pd.merge(game_df, away_defense, left_index=True, right_index=True, suffixes=('', '_away_off'))

	return game_df


def predict(override=False):
	today = datetime.datetime.today()

	if today.weekday() != 5 and override == False:
		return

	offense_df = pd.read_csv('data/offense.csv')
	defense_df = pd.read_csv('data/defense.csv')

	games_this_week = get_this_weeks_games(month=today.month, day=today.day, year=today.year)

	home_teams = []
	away_teams = []

	X_weekend = None

	for home_team, away_team in zip(games_this_week['home_teams'], games_this_week['away_teams']):
		X_game = get_game_data(home_team, away_team, offense_df, defense_df)

		if X_game.shape[0] != 1:
			continue

		home_teams.append(home_team)
		away_teams.append(away_team)

		if X_weekend is None:
			X_weekend = X_game
		else:
			X_weekend = pd.concat((X_weekend, X_game))

	clf = joblib.load('models/cfb_lr_model.joblib')

	weekend_df = pd.DataFrame(
		{
			'home_teams': home_teams,
			'away_teams': away_teams,
			'win_prob': clf.predict_proba(X_weekend)[:, 1]
		}
	)

	weekend_df['winner'] = weekend_df.apply(lambda row: row['home_teams'] if row['win_prob'] >= 0.5 else row['away_teams'], axis=1)

	weekend_df.to_csv(f'predictions/predictions_{today.month}_{today.day}_{today.year}.csv', index=False)
	

if __name__ == '__main__':
	predict(override=True)
