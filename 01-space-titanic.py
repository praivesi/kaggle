
# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:32.064816Z","iopub.execute_input":"2022-09-20T11:30:32.065311Z","iopub.status.idle":"2022-09-20T11:30:33.326472Z","shell.execute_reply.started":"2022-09-20T11:30:32.065215Z","shell.execute_reply":"2022-09-20T11:30:33.325055Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.svm import SVC

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:33.329045Z","iopub.execute_input":"2022-09-20T11:30:33.330168Z","iopub.status.idle":"2022-09-20T11:30:33.337125Z","shell.execute_reply.started":"2022-09-20T11:30:33.330107Z","shell.execute_reply":"2022-09-20T11:30:33.335741Z"}}
from platform import python_version

print(python_version())

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:33.339027Z","iopub.execute_input":"2022-09-20T11:30:33.340004Z","iopub.status.idle":"2022-09-20T11:30:33.461534Z","shell.execute_reply.started":"2022-09-20T11:30:33.339872Z","shell.execute_reply":"2022-09-20T11:30:33.460323Z"}}
pd.set_option('display.max_rows', None)

train = pd.read_csv('../input/spaceship-titanic/train.csv')
test = pd.read_csv('../input/spaceship-titanic/test.csv')
sample_submission = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:33.464849Z","iopub.execute_input":"2022-09-20T11:30:33.465372Z","iopub.status.idle":"2022-09-20T11:30:33.511191Z","shell.execute_reply.started":"2022-09-20T11:30:33.465312Z","shell.execute_reply":"2022-09-20T11:30:33.509699Z"}}
train.head(20)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:33.512759Z","iopub.execute_input":"2022-09-20T11:30:33.513196Z","iopub.status.idle":"2022-09-20T11:30:33.538772Z","shell.execute_reply.started":"2022-09-20T11:30:33.513154Z","shell.execute_reply":"2022-09-20T11:30:33.537528Z"}}
test.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:33.540402Z","iopub.execute_input":"2022-09-20T11:30:33.540880Z","iopub.status.idle":"2022-09-20T11:30:33.552964Z","shell.execute_reply.started":"2022-09-20T11:30:33.540841Z","shell.execute_reply":"2022-09-20T11:30:33.551779Z"}}
sample_submission.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:33.554319Z","iopub.execute_input":"2022-09-20T11:30:33.555827Z","iopub.status.idle":"2022-09-20T11:30:33.586038Z","shell.execute_reply.started":"2022-09-20T11:30:33.555779Z","shell.execute_reply":"2022-09-20T11:30:33.584742Z"}}
train.info()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:33.587392Z","iopub.execute_input":"2022-09-20T11:30:33.588205Z","iopub.status.idle":"2022-09-20T11:30:33.600561Z","shell.execute_reply.started":"2022-09-20T11:30:33.588160Z","shell.execute_reply":"2022-09-20T11:30:33.599024Z"}}
train['HomePlanet'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:33.602518Z","iopub.execute_input":"2022-09-20T11:30:33.604969Z","iopub.status.idle":"2022-09-20T11:30:33.639741Z","shell.execute_reply.started":"2022-09-20T11:30:33.604919Z","shell.execute_reply":"2022-09-20T11:30:33.638872Z"}}
planet_mapping = {"Earth": 0, "Europa": 1, "Mars": 2}

train_test = [train, test]

for dataset in train_test:
    dataset['HomePlanet'] = dataset['HomePlanet'].map(planet_mapping)

train.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:33.643125Z","iopub.execute_input":"2022-09-20T11:30:33.644041Z","iopub.status.idle":"2022-09-20T11:30:33.693546Z","shell.execute_reply.started":"2022-09-20T11:30:33.644006Z","shell.execute_reply":"2022-09-20T11:30:33.692218Z"}}
train.head()
cabin_splitted = train['Cabin'].str.split("/", n = 2, expand = True)

train['Cabin_deck'] = cabin_splitted[0]
train['Cabin_num'] = cabin_splitted[1]
train['Cabin_side'] = cabin_splitted[2]
train = train.drop(columns = ['Cabin'])
train.head()



# train['CryoSleep'].value_counts()


# train['VIP'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:33.695113Z","iopub.execute_input":"2022-09-20T11:30:33.695458Z","iopub.status.idle":"2022-09-20T11:30:34.099818Z","shell.execute_reply.started":"2022-09-20T11:30:33.695428Z","shell.execute_reply":"2022-09-20T11:30:34.098909Z"}}
dead_deck = train[train['Transported']==0]['Cabin_deck'].value_counts()
saved_deck = train[train['Transported']==1]['Cabin_deck'].value_counts()

dead_num = train[train['Transported']==0]['Cabin_num'].value_counts()
saved_num = train[train['Transported']==1]['Cabin_num'].value_counts()

dead_side = train[train['Transported']==0]['Cabin_side'].value_counts()
saved_side = train[train['Transported']==1]['Cabin_side'].value_counts()

# df = pd.DataFrame([dead_deck, saved_deck, dead_num, saved_num, dead_side, saved_side])
# df.index = ['dead_deck', 'saved_deck', 'dead_num', 'saved_num', 'dead_side', 'saved_side']

df = pd.DataFrame([dead_deck, saved_deck, dead_side, saved_side])
df.index = ['dead_deck', 'saved_deck', 'dead_side', 'saved_side']
df.plot(kind='bar', stacked=True, figsize=(10,5))

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:34.102133Z","iopub.execute_input":"2022-09-20T11:30:34.102599Z","iopub.status.idle":"2022-09-20T11:30:34.112052Z","shell.execute_reply.started":"2022-09-20T11:30:34.102565Z","shell.execute_reply":"2022-09-20T11:30:34.110928Z"}}
train['CryoSleep'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:34.113860Z","iopub.execute_input":"2022-09-20T11:30:34.114184Z","iopub.status.idle":"2022-09-20T11:30:34.132541Z","shell.execute_reply.started":"2022-09-20T11:30:34.114156Z","shell.execute_reply":"2022-09-20T11:30:34.131592Z"}}
train = train.dropna()

for train_data in train:
    train['CryoSleep'] = train['CryoSleep'].astype(int)
    
# train.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:34.134030Z","iopub.execute_input":"2022-09-20T11:30:34.135029Z","iopub.status.idle":"2022-09-20T11:30:34.142850Z","shell.execute_reply.started":"2022-09-20T11:30:34.134996Z","shell.execute_reply":"2022-09-20T11:30:34.141754Z"}}
train['CryoSleep'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:34.144127Z","iopub.execute_input":"2022-09-20T11:30:34.144520Z","iopub.status.idle":"2022-09-20T11:30:34.170655Z","shell.execute_reply.started":"2022-09-20T11:30:34.144488Z","shell.execute_reply":"2022-09-20T11:30:34.169490Z"}}
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:34.171944Z","iopub.execute_input":"2022-09-20T11:30:34.172273Z","iopub.status.idle":"2022-09-20T11:30:34.559752Z","shell.execute_reply.started":"2022-09-20T11:30:34.172243Z","shell.execute_reply":"2022-09-20T11:30:34.558611Z"}}
deck_awake = train[train['CryoSleep']==0]['Cabin_deck'].value_counts()
deck_sleep = train[train['CryoSleep']==1]['Cabin_deck'].value_counts()
side_awake = train[train['CryoSleep']==0]['Cabin_side'].value_counts()
side_sleep = train[train['CryoSleep']==1]['Cabin_side'].value_counts()

df = pd.DataFrame([deck_awake, deck_sleep, side_awake, side_sleep])
df.index = ['deck_awake', 'deck_sleep', 'side_awake', 'side_sleep']
df.plot(kind='bar', stacked=True, figsize=(10,5))


# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:34.561188Z","iopub.execute_input":"2022-09-20T11:30:34.561831Z","iopub.status.idle":"2022-09-20T11:30:34.570255Z","shell.execute_reply.started":"2022-09-20T11:30:34.561796Z","shell.execute_reply":"2022-09-20T11:30:34.569267Z"}}
train['Cabin_deck'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:34.571698Z","iopub.execute_input":"2022-09-20T11:30:34.572229Z","iopub.status.idle":"2022-09-20T11:30:34.585050Z","shell.execute_reply.started":"2022-09-20T11:30:34.572198Z","shell.execute_reply":"2022-09-20T11:30:34.584159Z"}}
train['Cabin_side'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:34.586310Z","iopub.execute_input":"2022-09-20T11:30:34.586838Z","iopub.status.idle":"2022-09-20T11:30:34.894395Z","shell.execute_reply.started":"2022-09-20T11:30:34.586806Z","shell.execute_reply":"2022-09-20T11:30:34.893018Z"}}
# DECK F는 CryoSleep을 False로 채워도 될듯
# DECK G는 CryoSleep을 True로 채워도 될듯

deck_a = train[train['Cabin_deck']=='A']['CryoSleep'].value_counts()
deck_b = train[train['Cabin_deck']=='B']['CryoSleep'].value_counts()
deck_c = train[train['Cabin_deck']=='C']['CryoSleep'].value_counts()
deck_d = train[train['Cabin_deck']=='D']['CryoSleep'].value_counts()
deck_e = train[train['Cabin_deck']=='E']['CryoSleep'].value_counts()
deck_f = train[train['Cabin_deck']=='F']['CryoSleep'].value_counts()
deck_g = train[train['Cabin_deck']=='G']['CryoSleep'].value_counts()

side_s = train[train['Cabin_side']=='S']['CryoSleep'].value_counts()
side_p = train[train['Cabin_side']=='P']['CryoSleep'].value_counts()

df = pd.DataFrame([deck_a, deck_b, deck_c, deck_d, deck_e, deck_f, deck_g, side_s, side_p])
df.index = ['deck_a', 'deck_b', 'deck_c', 'deck_d', 'deck_e', 'deck_f', 'deck_g', 'side_s', 'side_p']
df.plot(kind='bar', stacked=True, figsize=(10,5))

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:34.896212Z","iopub.execute_input":"2022-09-20T11:30:34.896925Z","iopub.status.idle":"2022-09-20T11:30:34.999088Z","shell.execute_reply.started":"2022-09-20T11:30:34.896886Z","shell.execute_reply":"2022-09-20T11:30:34.996701Z"}}
train.tail(100)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.001227Z","iopub.execute_input":"2022-09-20T11:30:35.001734Z","iopub.status.idle":"2022-09-20T11:30:35.010110Z","shell.execute_reply.started":"2022-09-20T11:30:35.001689Z","shell.execute_reply":"2022-09-20T11:30:35.008450Z"}}
# train['Cabin_num'].value_counts()
print(train['Cabin_num'].max())
# train['Cabin_num'].min()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.011936Z","iopub.execute_input":"2022-09-20T11:30:35.013273Z","iopub.status.idle":"2022-09-20T11:30:35.239679Z","shell.execute_reply.started":"2022-09-20T11:30:35.013216Z","shell.execute_reply":"2022-09-20T11:30:35.238631Z"}}
# Europa 사람들이 상대적으로 좀 더 높은 확률로 살았고, Earth 사람들이 상대적으로 높은 확률로 죽은듯

dead = train[train['Transported']==0]['HomePlanet'].value_counts()
saved = train[train['Transported']==1]['HomePlanet'].value_counts()

print('dead => ', dead)
print('saved => ', saved)


df = pd.DataFrame([dead, saved])
df.index = ['dead', 'saved']
df.plot(kind='bar', stacked=True, figsize=(10,5))

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.241676Z","iopub.execute_input":"2022-09-20T11:30:35.242191Z","iopub.status.idle":"2022-09-20T11:30:35.474619Z","shell.execute_reply.started":"2022-09-20T11:30:35.242147Z","shell.execute_reply":"2022-09-20T11:30:35.473652Z"}}
# "Earth": 0, "Europa": 1, "Mars": 2

Earth = train[train['HomePlanet']==0]['Transported'].value_counts()
Europa = train[train['HomePlanet']==1]['Transported'].value_counts()
Mars = train[train['HomePlanet']==2]['Transported'].value_counts()

print('Earth => ', Earth)
print('Europa => ', Europa)
print('Mars => ', Mars)

df = pd.DataFrame([Earth, Europa, Mars])
df.index = ['Earth', 'Europa', 'Mars']
df.plot(kind='bar', stacked=True, figsize=(10,5))

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.476148Z","iopub.execute_input":"2022-09-20T11:30:35.476862Z","iopub.status.idle":"2022-09-20T11:30:35.679275Z","shell.execute_reply.started":"2022-09-20T11:30:35.476819Z","shell.execute_reply":"2022-09-20T11:30:35.678312Z"}}
# "Earth": 0, "Europa": 1, "Mars": 2
# Earth는 CryoSleep을 False로 채워도 될듯

Earth = train[train['HomePlanet']==0]['CryoSleep'].value_counts()
Europa = train[train['HomePlanet']==1]['CryoSleep'].value_counts()
Mars = train[train['HomePlanet']==2]['CryoSleep'].value_counts()

df = pd.DataFrame([Earth, Europa, Mars])
df.index = ['Earth', 'Europa', 'Mars']
df.plot(kind='bar', stacked=True, figsize=(10,5))

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.680791Z","iopub.execute_input":"2022-09-20T11:30:35.681509Z","iopub.status.idle":"2022-09-20T11:30:35.712320Z","shell.execute_reply.started":"2022-09-20T11:30:35.681466Z","shell.execute_reply":"2022-09-20T11:30:35.711095Z"}}
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.713839Z","iopub.execute_input":"2022-09-20T11:30:35.714335Z","iopub.status.idle":"2022-09-20T11:30:35.720648Z","shell.execute_reply.started":"2022-09-20T11:30:35.714304Z","shell.execute_reply":"2022-09-20T11:30:35.719436Z"}}
train = train.drop('Name', axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.722294Z","iopub.execute_input":"2022-09-20T11:30:35.722980Z","iopub.status.idle":"2022-09-20T11:30:35.764209Z","shell.execute_reply.started":"2022-09-20T11:30:35.722936Z","shell.execute_reply":"2022-09-20T11:30:35.762440Z"}}
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.773508Z","iopub.execute_input":"2022-09-20T11:30:35.773943Z","iopub.status.idle":"2022-09-20T11:30:35.788310Z","shell.execute_reply.started":"2022-09-20T11:30:35.773904Z","shell.execute_reply":"2022-09-20T11:30:35.787303Z"}}
train['Destination'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.790104Z","iopub.execute_input":"2022-09-20T11:30:35.790818Z","iopub.status.idle":"2022-09-20T11:30:35.805109Z","shell.execute_reply.started":"2022-09-20T11:30:35.790774Z","shell.execute_reply":"2022-09-20T11:30:35.803902Z"}}
destination_mapping = {"TRAPPIST-1e": 0, "55 Cancri e": 1, "PSO J318.5-22": 2}

train_test = [train, test]

for dataset in train_test:
    dataset['Destination'] = dataset['Destination'].map(destination_mapping)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.806807Z","iopub.execute_input":"2022-09-20T11:30:35.807466Z","iopub.status.idle":"2022-09-20T11:30:35.848169Z","shell.execute_reply.started":"2022-09-20T11:30:35.807423Z","shell.execute_reply":"2022-09-20T11:30:35.846896Z"}}
train.head(20)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:35.850201Z","iopub.execute_input":"2022-09-20T11:30:35.850793Z","iopub.status.idle":"2022-09-20T11:30:36.346177Z","shell.execute_reply.started":"2022-09-20T11:30:35.850747Z","shell.execute_reply":"2022-09-20T11:30:36.345145Z"}}
import seaborn as sns

sns.displot(data = train['VRDeck'], kind="kde")

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:36.347914Z","iopub.execute_input":"2022-09-20T11:30:36.348351Z","iopub.status.idle":"2022-09-20T11:30:36.534154Z","shell.execute_reply.started":"2022-09-20T11:30:36.348308Z","shell.execute_reply":"2022-09-20T11:30:36.533222Z"}}
import matplotlib.pyplot as plt

plt.plot(train['VRDeck'])
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:36.535422Z","iopub.execute_input":"2022-09-20T11:30:36.536276Z","iopub.status.idle":"2022-09-20T11:30:36.564289Z","shell.execute_reply.started":"2022-09-20T11:30:36.536240Z","shell.execute_reply":"2022-09-20T11:30:36.563443Z"}}
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:36.565575Z","iopub.execute_input":"2022-09-20T11:30:36.566545Z","iopub.status.idle":"2022-09-20T11:30:36.766034Z","shell.execute_reply.started":"2022-09-20T11:30:36.566499Z","shell.execute_reply":"2022-09-20T11:30:36.764939Z"}}
plt.plot(train['VRDeck'], train['Transported'], 'ro')
plt.axis([0, 21000, 0, 2])

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:36.767407Z","iopub.execute_input":"2022-09-20T11:30:36.768078Z","iopub.status.idle":"2022-09-20T11:30:36.774039Z","shell.execute_reply.started":"2022-09-20T11:30:36.768031Z","shell.execute_reply":"2022-09-20T11:30:36.772790Z"}}
# train['VRDeck'].value_counts().sort_index()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:36.775869Z","iopub.execute_input":"2022-09-20T11:30:36.776194Z","iopub.status.idle":"2022-09-20T11:30:37.139667Z","shell.execute_reply.started":"2022-09-20T11:30:36.776162Z","shell.execute_reply":"2022-09-20T11:30:37.138519Z"}}
# "Earth": 0, "Europa": 1, "Mars": 2
# Earth는 CryoSleep을 False로 채워도 될듯

Earth = train[train['HomePlanet']==0]['CryoSleep'].value_counts()
Europa = train[train['HomePlanet']==1]['CryoSleep'].value_counts()
Mars = train[train['HomePlanet']==2]['CryoSleep'].value_counts()

df = pd.DataFrame([Earth, Europa, Mars])
df.index = ['Earth', 'Europa', 'Mars']
df.plot(kind='bar', stacked=True, figsize=(10,5))

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:37.141132Z","iopub.execute_input":"2022-09-20T11:30:37.141520Z","iopub.status.idle":"2022-09-20T11:30:37.178463Z","shell.execute_reply.started":"2022-09-20T11:30:37.141486Z","shell.execute_reply":"2022-09-20T11:30:37.177423Z"}}
train_test_data = [train, test]

for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 1, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 1) & (dataset['Age'] <= 3), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 3) & (dataset['Age'] <= 7), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 7) & (dataset['Age'] <= 10), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 13), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 13) & (dataset['Age'] <= 14), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 15), 'Age'] = 6
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 16), 'Age'] = 7
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 19), 'Age'] = 8
    dataset.loc[(dataset['Age'] > 19) & (dataset['Age'] <= 22), 'Age'] = 9
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 26), 'Age'] = 10
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 11
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 12
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 13

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:37.179928Z","iopub.execute_input":"2022-09-20T11:30:37.180517Z","iopub.status.idle":"2022-09-20T11:30:37.727534Z","shell.execute_reply.started":"2022-09-20T11:30:37.180483Z","shell.execute_reply":"2022-09-20T11:30:37.726722Z"}}
# Age 16세 이하는 살 확률이 높지만 나머지 연령대는 죽을 확률이 높음

saved = train[train['Transported']==1]['Age'].value_counts()
dead = train[train['Transported']==0]['Age'].value_counts()

age_0 = train[train['Age'] == 0]['Transported'].value_counts()
age_1 = train[train['Age'] == 1]['Transported'].value_counts()
age_2 = train[train['Age'] == 2]['Transported'].value_counts()
age_3 = train[train['Age'] == 3]['Transported'].value_counts()
age_4 = train[train['Age'] == 4]['Transported'].value_counts()
age_5 = train[train['Age'] == 5]['Transported'].value_counts()
age_6 = train[train['Age'] == 6]['Transported'].value_counts()
age_7 = train[train['Age'] == 7]['Transported'].value_counts()
age_8 = train[train['Age'] == 8]['Transported'].value_counts()
age_9 = train[train['Age'] == 9]['Transported'].value_counts()
age_10 = train[train['Age'] == 10]['Transported'].value_counts()
age_11 = train[train['Age'] == 11]['Transported'].value_counts()
age_12 = train[train['Age'] == 12]['Transported'].value_counts()
age_13 = train[train['Age'] == 13]['Transported'].value_counts()

df = pd.DataFrame([age_0, age_1, age_2, age_3, age_4, age_5, age_6, age_7, age_8, age_9, age_10, age_11, age_12, age_13])
df.index = ['age_0', 'age_1', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7_under_16', 
            'age_8', 'age_9', 'age_10', 'age_11', 'age_12' , 'age_13']
ax = df.plot(kind='bar', stacked=True, figsize=(20,15))

for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height}', (x + width/2, y + height*1.02), ha='center')

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:37.728575Z","iopub.execute_input":"2022-09-20T11:30:37.729065Z","iopub.status.idle":"2022-09-20T11:30:37.735172Z","shell.execute_reply.started":"2022-09-20T11:30:37.729034Z","shell.execute_reply":"2022-09-20T11:30:37.734075Z"}}
train.rename(columns = {'Age' : 'Age_Level'}, inplace=True)
test.rename(columns = {'Age' : 'Age_Level'}, inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:37.738208Z","iopub.execute_input":"2022-09-20T11:30:37.738845Z","iopub.status.idle":"2022-09-20T11:30:37.775486Z","shell.execute_reply.started":"2022-09-20T11:30:37.738773Z","shell.execute_reply":"2022-09-20T11:30:37.774438Z"}}
train_test_data = [train, test]

for dataset in train_test_data:
    dataset['Luxury_sum'] = dataset['RoomService'] + dataset['FoodCourt'] + dataset['ShoppingMall'] + dataset['Spa'] + dataset['VRDeck']
    
train.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:37.777034Z","iopub.execute_input":"2022-09-20T11:30:37.777506Z","iopub.status.idle":"2022-09-20T11:30:37.926044Z","shell.execute_reply.started":"2022-09-20T11:30:37.777462Z","shell.execute_reply":"2022-09-20T11:30:37.924861Z"}}
train_test_data = [train, test]

for col in ['Luxury_sum', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    for dataset in train_test_data:
        dataset.loc[ dataset[col] <= 0, col] = 0
        dataset.loc[(dataset[col] > 0) & (dataset[col] <= 1), col] = 1
        dataset.loc[(dataset[col] > 1) & (dataset[col] <= 3), col] = 2
        dataset.loc[(dataset[col] > 3) & (dataset[col] <= 5), col] = 3       
        dataset.loc[(dataset[col] > 5) & (dataset[col] <= 7), col] = 4
        dataset.loc[(dataset[col] > 7) & (dataset[col] <= 10), col] = 5
        dataset.loc[(dataset[col] > 10) & (dataset[col] <= 100), col] = 6
        dataset.loc[(dataset[col] > 100) & (dataset[col] <= 1000), col] = 7
        dataset.loc[(dataset[col] > 1000) & (dataset[col] <= 10000), col] = 8
        dataset.loc[(dataset[col] > 10000) & (dataset[col] <= 30000), col] = 9
        dataset.loc[(dataset[col] > 30000) & (dataset[col] <= 50000), col] = 10
        dataset.loc[(dataset[col] > 50000) & (dataset[col] <= 100000), col] = 11
        dataset.loc[ dataset[col] > 100000, col] = 12

train.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:30:37.928182Z","iopub.execute_input":"2022-09-20T11:30:37.928575Z","iopub.status.idle":"2022-09-20T11:30:39.407730Z","shell.execute_reply.started":"2022-09-20T11:30:37.928542Z","shell.execute_reply":"2022-09-20T11:30:39.406593Z"}}
# RoomService, FoodCourt, ShoppingMall, Spa, VRDeck 이 0인 사람들이 살아남을 확률이 높고 나머지는 적음
# 이 모든 값들을 합친 Luxury_sum이 0인 사람들은 살아남을 확률이 더 높게 나옴 -> Luxury_sum == 0 이면 살아있는 경향이 개별 Luxury 들이 0일 때보다 높음
# 결론적으로 Luxury_sum == 0 이면 살았다고 예측해야 함

for col in ['Luxury_sum', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa']:
    saved = train[train['Transported']==1][col].value_counts()
    dead = train[train['Transported']==0][col].value_counts()
    
    df = pd.DataFrame([saved, dead])
    df.index = [col + '_saved', col + '_dead']
    df.plot(kind='bar', stacked=True, figsize=(20,15), rot=0)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T11:48:22.119086Z","iopub.execute_input":"2022-09-20T11:48:22.120028Z","iopub.status.idle":"2022-09-20T11:48:22.147999Z","shell.execute_reply.started":"2022-09-20T11:48:22.119975Z","shell.execute_reply":"2022-09-20T11:48:22.146843Z"}}
train.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T12:05:14.500727Z","iopub.execute_input":"2022-09-20T12:05:14.501151Z","iopub.status.idle":"2022-09-20T12:05:15.017597Z","shell.execute_reply.started":"2022-09-20T12:05:14.501116Z","shell.execute_reply":"2022-09-20T12:05:15.016143Z"}}
matched_count = 0
total_count = 0

for index, data in train.iterrows():
    #print(data)
    
    prob = 0.5
    
    planet_threshold = 0.1
    age_threshold = 0.1
    luxury_threshold = 0.2
    
    if data['HomePlanet'] == 0:
        prob -= planet_threshold
    elif data['HomePlanet'] == 1:
        prob += planet_threshold
    
    if data['Age_Level'] <= 7:
        prob += age_threshold
    else:
        prob -= age_threshold
        
    if data['Luxury_sum'] == 0:
        prob += luxury_threshold
    else:
        prob -= luxury_threshold
        
    if prob >= 0.5 and data['Transported']:
        matched_count += 1
    
    if prob < 0.5 and data['Transported'] == False:
        matched_count += 1
        
    total_count += 1
    
predict_rate = matched_count / total_count
print(predict_rate)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T12:13:28.749169Z","iopub.execute_input":"2022-09-20T12:13:28.749716Z","iopub.status.idle":"2022-09-20T12:13:28.788546Z","shell.execute_reply.started":"2022-09-20T12:13:28.749668Z","shell.execute_reply":"2022-09-20T12:13:28.787351Z"}}
test = test.fillna(0)
# for index, data in test.iterrows():
#     for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa' 'VRDeck']
#         if data[col] == 

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T12:13:46.290438Z","iopub.execute_input":"2022-09-20T12:13:46.291518Z","iopub.status.idle":"2022-09-20T12:13:46.366327Z","shell.execute_reply.started":"2022-09-20T12:13:46.291465Z","shell.execute_reply":"2022-09-20T12:13:46.365125Z"}}
test.head(50)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T12:20:00.891680Z","iopub.execute_input":"2022-09-20T12:20:00.892108Z","iopub.status.idle":"2022-09-20T12:20:01.201340Z","shell.execute_reply.started":"2022-09-20T12:20:00.892074Z","shell.execute_reply":"2022-09-20T12:20:01.200144Z"}}
transported = []
for index, data in test.iterrows():
    prob = 0.5
    
    planet_threshold = 0.1
    age_threshold = 0.1
    luxury_threshold = 0.2
    
    if data['HomePlanet'] == 0:
        prob -= planet_threshold
    elif data['HomePlanet'] == 1:
        prob += planet_threshold
    
    if data['Age_Level'] <= 7:
        prob += age_threshold
    else:
        prob -= age_threshold
        
    if data['Luxury_sum'] == 0:
        prob += luxury_threshold
    else:
        prob -= luxury_threshold
        
    if prob >= 0.5:
#         data['Transported'] = True
        transported.append(True)
    else:
#         data['Transported'] = False
        transported.append(False)
    
test['Transported'] = transported

test.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T12:20:10.684639Z","iopub.execute_input":"2022-09-20T12:20:10.685061Z","iopub.status.idle":"2022-09-20T12:20:10.696708Z","shell.execute_reply.started":"2022-09-20T12:20:10.685030Z","shell.execute_reply":"2022-09-20T12:20:10.695385Z"}}
sample_submission.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T12:20:54.580984Z","iopub.execute_input":"2022-09-20T12:20:54.581415Z","iopub.status.idle":"2022-09-20T12:20:54.597024Z","shell.execute_reply.started":"2022-09-20T12:20:54.581378Z","shell.execute_reply":"2022-09-20T12:20:54.595838Z"}}
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test['Transported']
})

submission.to_csv('submission.csv', index=False)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-20T12:21:39.662064Z","iopub.execute_input":"2022-09-20T12:21:39.663224Z","iopub.status.idle":"2022-09-20T12:21:39.681557Z","shell.execute_reply.started":"2022-09-20T12:21:39.663180Z","shell.execute_reply":"2022-09-20T12:21:39.680326Z"}}
final_submission = pd.read_csv('submission.csv')
final_submission.head()

# %% [code]
