import numpy as np
class loadCsv():

   def loadCSVfile2():
      tmp = np.loadtxt("dataset.csv", dtype=np.str, delimiter=",")
      tital = tmp[0, 0:].astype(np.chararray)  # 加载数据部分
      data = tmp[1:,1:].astype(np.float)#加载数据部分
      label = tmp[1:,0].astype(np.float)#加载类别标签部分
      return data #返回array类型的数据

#print(loadCsv.loadCSVfile2())