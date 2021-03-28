import requests

url = 'http://localhost:5000/predict_api'

r = requests.post(url,json={'Experience':10, 'Income':120, 'Education'1:2, 
			    'Family':2, 'CreditCard':1, 'CCAvg':3, 'Online':1, 
			    'Mortgage':0, 'Securities Account':0, 'CD Account':1})

print(r.json())