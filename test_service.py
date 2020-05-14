import requests
from jsondiff import diff
bkg_no = 'BKKVL8636600'

r1 = requests.get(url='http://si-automation.dounets.com:8803/opus/booking-info/' + bkg_no)
r2 = requests.get(url='http://si-automation.dounets.com:8803/result/booking-info/' + bkg_no)

json1 = r1.json()
json2 = r2.json()

# print(type(json1))

print(diff(json1, json2))