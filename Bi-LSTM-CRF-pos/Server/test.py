-*- coding: UTF-8 -*-
import requests   
respones = requests.post('http://10.200.1.239:8080/train',data={"sequenceid":'["d96853f4-ab57-45bf-8ad9-1403ba1cb504"]'})
print (respones)



