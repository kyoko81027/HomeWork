class temperatureList():

    def OpenExcel():
        f =open('氣候月平均.txt')  # txt檔案和當前指令碼在同一目錄下，所以不用寫具體路徑
        p = f.readlines()
        #emperatureList.ToList(p)
        return p

        #print(p)
    def ToList():
        data = temperatureList.OpenExcel()
        dicList= {}
        for pArry in data:
            Restr = pArry.replace('\n','')
            str = Restr.split('\t')
            dicList[str[0]] = str[1]

        #print(dicList)
        return dicList
#temperatureList.ToList()




