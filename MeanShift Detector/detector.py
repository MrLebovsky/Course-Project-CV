#Идея алгоритма: разобьем изображение на кластеры
#Найдем дескрипторы, которые принадлежат i-му кластеру
#Замэтчим дескрипторы текущего кластера с эталоном. Хорошо мэтчится? Там объект!

import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 100

img1 = cv2.imread('images/my/etalon.jpg', 0)  # эталонное изображение
img2 = cv2.imread('images/my/test.jpg', 0) # изображения для поиска эталона

#найдем интересные точки и построим дескрипторы
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None) 
kp2, des2 = sift.detectAndCompute(img2, None)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

#запомним координаты всех точек интереса для изображения, на котором будем осуществлять поиск объекта
x = np.array([kp2[0].pt])
for i in range(len(kp2)):
    x = np.append(x, [kp2[i].pt], axis=0)

x = x[1:len(x)]

#Квантииль — значение, которое заданная случайная величина не превышает с фиксированной вероятностью
#Используем 500 образцов для оценки пропускной способности MeanShift
bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

#Вычислим средний сдвиг
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
ms.fit(x) #непосредственно кластеризация

labels = ms.labels_ #labels of each point
cluster_centers = ms.cluster_centers_  #координаты центров каждого кластера

labels_unique = np.unique(labels) #найдем все уникальные метки
n_clusters_ = len(labels_unique)
print("Кол-во предполагаемых кластеров : %d" % n_clusters_)

#массив длинной n_clusters_, инициализировали null-значением
s = [None] * n_clusters_

#Определим точки интереса, принадлежащие каждому кластеру
#Идем по уникальным кластерам
for i in range(n_clusters_):
    l = ms.labels_
    d, = np.where(l == i) #получим массив индексов, где метка == i
    print(d.__len__())
    s[i] = list(kp2[xx] for xx in d) #запомним нужные нам точки интереса

des2_ = des2

#Найдем эталонный объект!
for i in range(n_clusters_):

    kp2 = s[i] #точки интереса текущего кластера
    l = ms.labels_
    d, = np.where(l == i)
    des2 = des2_[d, ] #дескрипторы текущего кластера

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    des1 = np.float32(des1)
    des2 = np.float32(des2)
	
    matches = flann.knnMatch(des1, des2, k=2) #замэтчили дескрипторы эталона с дескрипторами текущего кластера

    # Запомним все хорошие мэтчи, дальше по сути как обычно все
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            print ("Не вычислили гомографию!")
        else:
            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            img3 = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 255), 3)	#нарисуем местоположение объекта			

    else:
        print ("Не достаточно мэтчей - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
		
plt.imshow(img3, 'gray'), plt.show()