import pymysql
import sys
import os
import json

from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()  # use .env as environmental variables from OS


def get_connection() -> pymysql.Connection:
	try:
		conn = pymysql.connect(
			user=os.environ.get("DB_USER"),
			password=os.environ.get("DB_PASSWORD"),
			host=os.environ.get("DB_HOST"),
			port=int(os.environ.get("DB_PORT")),
			database=os.environ.get("DB_DATABASE"),
		)

		return conn
	except pymysql.Error as e:
		print(f"Error connecting to DB Server: {e}")
		sys.exit(1)


def insert_tracking_data(conn: pymysql.Connection, data: {}, hive_id, video_id):
	cur = conn.cursor()

	insert_tuples: [()] = []

	for key in data.keys():
		insert_tuples.append(
			(hive_id, video_id, datetime.fromtimestamp(key), data[key][0], data[key][1], data[key][2], data[key][3]))

	cur.executemany(
		'INSERT INTO video_data (hive_id, video_id, timestamp, approaches_count, departures_count, no_movement_count, number_of_detected_bees) VALUES (%s, %s, %s, %s, %s, %s, %s)',
		insert_tuples
	)

	conn.commit()

	cur.close()
