#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import datetime
import calendar
import holidays
from math import radians, sin, cos, acos
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as mcolors
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms

def get_distance(lat1, lng1, lat2, lng2):
    """
    based on the longitude and latitude to calculate the distance / metric
    """
    if abs(lat1) > 90 or abs(lat2) > 90 or abs(lng1) > 180 or abs(lng2) > 180 or abs(lat1-lat2) == 0 and abs(lng1-lng2) == 0:
        return None  # exam longitude and latitude
    rad_lat1, rad_lng1, rad_lat2, rad_lng2 = map(
        radians, [lat1, lng1, lat2, lng2])
    distance = 6371 * acos(
        sin(rad_lat1) * sin(rad_lat2) + cos(rad_lat1) * cos(rad_lat2) * cos(rad_lng1 - rad_lng2))
    return distance * 1000
    
def kmeans_62_data(data_cluster):
    '''
    using kmeans to make all stations to 62 cluser
    the same step as processing the 2018 data 
    aim to reduce the coding 
    '''
    scaler = MinMaxScaler()
    data_cluster_normal = scaler.fit_transform(data_cluster)
    kmeans = KMeans(n_clusters=62,max_iter=10000, init="k-means++", tol=1e-6)
    result = kmeans.fit(data_cluster_normal)
    label = kmeans.labels_
    return label

def exact_dict_eage(df_grap_test_1):
    '''
    each of riding has start_station_id','end_station_id, 
    each of different start_station_id','end_station_id has the different color
    we build a dictionary for each note and each color, which includes all of the renting-lending 
    because every day's renting-lending doesn't include all notes
    we can use this function to exact the sample about stations per day from the whole dictionary for graph 

    '''
            
    # pick up the edge color dic for drawing 
    df_grap_test_t= df_grap_test_1.drop_duplicates(subset=['start_station_id','end_station_id'])
    value = []
    for _, row in df_grap_test_t.iterrows():
        value.append((row['start_station_id'], row['end_station_id']))
    edge_color = {}
    edge_count={}
    for key in value:
        if key in edge_color_all:
            value = edge_color_all[key]
            edge_color[key] = value
        if key in edge_count_all:
            value = edge_count_all[key]
            edge_count[key] = value
    return edge_color,edge_count


def exam_channels(image_paths):
    image = Image.open(image_paths)
    mode = image.mode
    
    if mode == 'RGB':
        num_channels = 3
    elif mode == 'RGBA':
        num_channels = 4
    else:
        num_channels = 1  

    return print(f'The image has {num_channels} channel(s).')


use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")
    

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()

def randnorm(size):
    return np.random.normal(0, 1, size).astype('float32')





class processingData:
    def __init__(self, df):
        self.df = df


    def processingData_after_cluster(self):
        '''
        processing data includes      adding hour
                                      seeting label:m_d_h(month_day_hour)
                                      adding is weekend 
                                      adding is_hoilday
                                      adding count in per time period
                                      adding sesaon 

        '''


        ##adding hour
        self.df['starttime_hour'] = self.df['starttime'].dt.hour
        self.df['stoptime_hour'] = self.df['stoptime'].dt.hour


        ## setting label 
        self.df['month']= self.df['month'].astype(str)
        self.df['start_day']=self.df['start_day'].astype(str)
        self.df['starttime_hour']=self.df['starttime_hour'].astype(str)
        self.df['end_day']=self.df['end_day'].astype(str)
        self.df['stoptime_hour']=self.df['stoptime_hour'].astype(str)

        self.df['start_m_d_h'] = self.df['month']+'_'+self.df['start_day']+'_'+self.df['starttime_hour']
        self.df['end_m_d_h'] = self.df['month']+'_'+self.df['end_day']+'_'+self.df['stoptime_hour']

        self.df['month']=self.df['month'].astype(int)
        self.df['start_day']=self.df['start_day'].astype(int)
        self.df['starttime_hour']=self.df['starttime_hour'].astype(int)
        self.df['end_day']= self.df['end_day'].astype(int)
        self.df['stoptime_hour']= self.df['stoptime_hour'].astype(int)

        ## adding is weekend 
        self.df['is_weekend'] =self.df['starttime'].dt.dayofweek // 5 == 1
        self.df['is_weekend'] = self.df['is_weekend'].apply(lambda x: 1 if x==True else 0) # one-hot

        ## adding is_holiday
        
        us_holidays = holidays.US()

        is_holiday = []
        for date in self.df.starttime:
            if date in us_holidays:
                a = 1
            else:
                a = 0
            is_holiday.append(a)
        self.df['is_holiday'] = is_holiday



        ### adding count for period(0-1/1-2)
        BorrowCounts= self.df.groupby(['cluster_label', 'start_m_d_h']).size().reset_index()
        self.df = pd.merge(self.df, BorrowCounts, on=['cluster_label', 'start_m_d_h'], how='left')

        BorrowCounts= self.df.groupby(['cluster_label', 'end_m_d_h']).size().reset_index()
        self.df = pd.merge(self.df, BorrowCounts, on=['cluster_label', 'end_m_d_h'], how='left')

        self.df  = self.df.rename(columns={'0_x': 'count_start','0_y': 'count_end'})

        ## conver object to datetime
        self.df['startime_y_m_d']= pd.to_datetime(self.df['startime_y_m_d'], format='%Y/%m/%d')

        ## adding season
        season =[]
        for month in self.df['month']:
            if month in [3, 4, 5]:
                a = 1
            elif month in [6, 7, 8]:
                a = 2
            elif month in [9, 10, 11]:
                a = 3
            else:
                a = 4
            season.append(a)
        self.df['season'] = season

        return self.df  





    def processingData_before_cluster(self):
        '''
        processing data includes transform object to datetime 
                                 cope with NAN
                                 adding the duration mintues
                                 adding the time interval
                                 addding month/start_day/end_day
                                 adding weekday
                                 adding starttime y_m_d
                                 one-hot usertype


        '''
        # translate the format 'datetime'
        self.df['starttime']=self.df['starttime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
        self.df['stoptime']=self.df['stoptime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))


        # cope with the NAN
        print('The precentage of NAN of the whole data before processing NAN:',self.df.isna().sum()[3]/len(self.df))
        self.df.dropna(inplace=True)
        print('The precentage of NAN of the whole data after processing NAN :',self.df.isna().sum()[3]/len(self.df))


        # adding the duration mintues
        duration = self.df.stoptime - self.df.starttime
        self.df['duration_mintues'] = duration.apply(lambda x:  round(x.total_seconds()/60)) 

        # adding the time interval
        bins = [0, 30, 60, 90, 120, 150, 180,max(self.df['duration_mintues'])]
        self.df['duration_bin']= pd.cut(self.df['duration_mintues'], bins=bins)


        # addding month/start_day/end_day
        self.df['month'] = self.df['starttime'].dt.month
        self.df['start_day'] = self.df['starttime'].dt.day
        self.df['end_day'] = self.df['stoptime'].dt.day
        # adding weekday
        self.df['weekday'] = self.df['starttime'].dt.weekday
        self.df['weekday'] = self.df['starttime'].dt.weekday
        # adding starttime y_m_d
        self.df['startime_y_m_d'] = self.df['starttime'].dt.strftime('%Y-%m-%d')

        # one-hot usertype
        self.df['usertype'] = self.df['usertype'].apply(lambda x: 1 if x=='Subscriber' else 0)

        return self.df



class ProcssingImage():
    '''
    the aim for this part to resize the images to 224*224 and delete the alpha channel
    '''

    def __init__(self, image_paths, output_folder):
        self.image_paths = image_paths
        self.output_folder = output_folder

    def process_image(self, idx):
        image_path = self.image_paths[idx]
        original_image = Image.open(image_path)
        
        # delete the alpha chaneel
        original_image  = original_image .convert('RGB') 
        
        width, height = original_image.size
        aspect_ratio = width / height
        
        target_size = (224, int(224 / aspect_ratio))
        resized_image = original_image.resize(target_size, Image.ANTIALIAS)
        
        final_image = Image.new('RGB', (224, 224), (255, 255, 255))
        
        x_offset = (224 - resized_image.width) // 2
        y_offset = (224 - resized_image.height) // 2
        
        final_image.paste(resized_image, (x_offset, y_offset))
        filename = f'RLbike_graph_{idx+1}.png'
        file_path = os.path.join(self.output_folder, filename)
        final_image.save(file_path)



class DataTpyeConversion():
    '''
    the pre-step for translating images into the tensor 
    '''
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

