import pandas as pd  
df=pd.DataFrame({'height':[165,165,157,170,175,165,155,170],
                        'weight':[48,57,50,54,64,61,43,59]})
from sklearn import  linear_model  #导入线性回归模型的机器学习包
X=pd.DataFrame(df['height'])       #定义x变量
y=df['weight']                   #定义y变量

clf = linear_model.LinearRegression() #建立线性回归模型，并将变量代入模型进行训练
clf.fit(X, y)                    #拟合回归系数
print('回归系数: \n', clf.coef_)   # 输出回归参数w
print('截距: \n', clf.intercept_)   # 输出回归参数b
y_pred =clf.predict(X)          # 输出预测值
print(y_pred)

import matplotlib.pyplot as plt               #可视化/绘图
plt.scatter(X, y,  color='red')                 #真实值散点图
plt.plot(X,y_pred, color='blue', linewidth=1.5)   #线性回归预测趋势线
plt.show()