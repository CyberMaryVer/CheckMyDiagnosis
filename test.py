import requests
import json

url = 'http://localhost:5000/test/'

data = 'https://bc-clinic.ru/photogallery/dermatology/vospalilas_rodinka.jpg'

j_data = {'name':data}
r = requests.get(url = url, params = j_data)
print(r.text)