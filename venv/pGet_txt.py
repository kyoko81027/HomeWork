class temperatureList():

    def OpenExcel():
        f =open('氣候月平均.txt')  # txt檔案和當前指令碼在同一目錄下，所以不用寫具體路徑
        p = f.readlines()  #把每一行讀出來
        #emperatureList.ToList(p)
        return p

        #print(p)
    def ToList():
        data = temperatureList.OpenExcel() #讀txt
        dicList= {} #宣告Dictionaries
        for pArry in data: #跑回圈把資料塞到Dictionaries裡
            Restr = pArry.replace('\n','') #把換行符號拿掉
            str = Restr.split('\t') #利用空格符號切割資料
            dicList[str[0]] = str[1]

        #print(dicList)
        return dicList
#temperatureList.ToList()




