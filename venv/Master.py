import pGet_txt as pGet #引用檔案PGet_txt
TList =pGet.temperatureList #引用檔案裡的class
dicList = TList.ToList() #撈取資料
for name,value in dicList.items():
    if float(value) >=28.0: #判斷超過28度顯示太熱
        print("這個月\""+name+"\" 氣溫:"+value+" 太熱了!")
    elif float(value) <=20.0: #判斷小於20度顯示太冷
        print("這個月\"" + name+"\" 氣溫:" + value+ " 太冷了!")
    else:
        print("這個月\"" + name + "\" 氣溫:" + value)

#print(dicList)