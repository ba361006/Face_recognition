import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-3,3,50)
y1 = 2*x +1
y2 = 2**x
y3 = 1
y4 = 10

fig = plt.figure(num = 3, figsize = (8,5))#創造一個空白的圖片

fig.add_subplot(221)#與plt.subplot功能幾乎一樣但不知道哪裡有差，(1,2,1)為1列、2行、第一張圖的位置
plt.scatter(x,y1)#與plt.plot功能相同，對圖做事情，scatter為畫出來的圖用散點表示；因為接在fig.add_subplot(121)底下，所以是對第一張圖做畫

plt.subplot(222)#與fig.add_subplot功能幾乎一樣(fig是plt.figure的名字，可改)，
plt.plot(x,y2,'r-',lw=5)#與plt.scatter一樣是畫圖，r-為紅色實線、lw為線的寬；因為接在plt.subplot(122)下面，所以是對第二張圖做畫

plt.subplot(223)
plt.plot(x,y3,'b')

plt.subplot(224)
plt.plot(x,y4,'g')


# plt.ion() #若要能秀出圖片的同時一直對圖片上的畫做事就要開啟這個功能(interaction on)；關閉則是plt.ioff()
plt.show()#秀出圖片，並暫停對圖片上的畫做事

