import requests
from datetime import datetime
from datetime import datetime, timedelta

# -------------------------------------------------------

uri = 'https://api.open-meteo.com/v1/forecast?latitude=53.1235&longitude=18.0076&current=temperature_2m&daily=temperature_2m_max,temperature_2m_min&timezone=Europe%2FBerlin&forecast_days=2'

req = requests.get(uri).json()

acctual_temp = req.get('current').get('temperature_2m')


temperature_tomorrow_max = req.get('daily').get('temperature_2m_max')[1]
temperature_tomorrow_min = req.get('daily').get('temperature_2m_min')[1]


print('\r\n------------ Bydgoszcz ------------')
print('Aktualna temperatura: '+str(acctual_temp)+'°C')
print('')
print('Jutro:')
print('Minimalna temp: '+str(temperature_tomorrow_min)+'°C')
print('Maksymalna temp: '+str(temperature_tomorrow_max)+'°C')