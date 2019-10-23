import pGet_txt as pGet
TList =pGet.temperatureList
dicList = TList.ToList()
for name,value in dicList.items():
    if float(value) >=28.0:
        print("這個月\""+name+"\"太熱了! 氣溫:"+value)
    elif float(value) <=20.0:
        print("這個月\"" + name + "\"太冷了! 氣溫:" + value)
    else:
        print("這個月\"" + name + "\" 氣溫:" + value)

#print(dicList)