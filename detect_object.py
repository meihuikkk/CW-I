#
# import numpy as np
# import matplotlib.pyplot as plt
# # from sklearn.metrics import roc_curve, auc
#
# # 计算
# # fpr, tpr, thread = roc_curve(y_test, y_score)
# # roc_auc[i] = auc(fpr, tpr)
#
# fpr=[1, 0.232, 0.278, 0.09,0.02]
# tpr=[1, 1, 1, 1, 1]
# # 绘图
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % 1)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve')
# plt.legend(loc="lower right")
# plt.savefig('roc.png',)
# plt.show()

import os
import cv2



entry_detector = cv2.CascadeClassifier("cascade.xml")

# img = cv2.imread('No_entry_new/NoEntry5.bmp')
imgs = os.listdir('No_entry_new')
imgNum = len(imgs)
print(imgNum)
for i in range(imgNum):
    print(imgs[i])
    img = cv2.imread('No_entry_new/'+imgs[i])
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    no_entry = entry_detector.detectMultiScale(img, 1.05, 5, cv2.CASCADE_SCALE_IMAGE)
    for x,y,w,h in no_entry:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img',img)
    cv2.imwrite('No_entry/result_scale_1.05_minNei_5/NoEntry_'+str(i)+'result.jpg',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
