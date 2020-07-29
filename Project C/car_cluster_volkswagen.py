# 使用KMeans进行聚类
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

# 数据加载
data = pd.read_csv('CarPrice_Assignment.csv')
car_x = data[["symboling", "fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","wheelbase","carlength","carwidth",	"carheight","curbweight","enginetype","cylindernumber",	"enginesize","fuelsystem","boreratio","stroke",	"compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
car_x['fueltype'] = le.fit_transform(car_x['fueltype'])
car_x['aspiration'] = le.fit_transform(car_x['aspiration'])
car_x['doornumber'] = le.fit_transform(car_x['doornumber'])
car_x['carbody'] = le.fit_transform(car_x['carbody'])
car_x['drivewheel'] = le.fit_transform(car_x['drivewheel'])
car_x['enginelocation'] = le.fit_transform(car_x['enginelocation'])
car_x['enginetype'] = le.fit_transform(car_x['enginetype'])
car_x['cylindernumber'] = le.fit_transform(car_x['cylindernumber'])
car_x['fuelsystem'] = le.fit_transform(car_x['fuelsystem'])

# 规范化到 [0,1] 空间
min_max_scaler=preprocessing.MinMaxScaler()
car_x=min_max_scaler.fit_transform(car_x)
pd.DataFrame(car_x).to_csv('temp.csv', index=False)

#使用KMeans聚类
kmeans = KMeans(n_clusters=10)
kmeans.fit(car_x)
predict_y = kmeans.predict(car_x)

# 合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'cluster_number'},axis=1,inplace=True)
# 将结果导出到CSV文件中
result.to_csv("car_cluster_result.csv",index=False)