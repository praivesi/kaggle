# %% [code] {"execution":{"iopub.status.busy":"2022-09-27T12:09:47.814490Z","iopub.execute_input":"2022-09-27T12:09:47.815031Z","iopub.status.idle":"2022-09-27T12:09:47.831324Z","shell.execute_reply.started":"2022-09-27T12:09:47.814991Z","shell.execute_reply":"2022-09-27T12:09:47.829773Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-09-27T12:09:47.834036Z","iopub.execute_input":"2022-09-27T12:09:47.835072Z","iopub.status.idle":"2022-09-27T12:09:47.859089Z","shell.execute_reply.started":"2022-09-27T12:09:47.835032Z","shell.execute_reply":"2022-09-27T12:09:47.857875Z"}}
pd.set_option('display.max_rows', None)

train = pd.read_csv('/kaggle/input/dfl-bundesliga-data-shootout/train.csv')
sample_submission = pd.read_csv('/kaggle/input/dfl-bundesliga-data-shootout/sample_submission.csv')

# %% [code] {"execution":{"iopub.status.busy":"2022-09-27T12:13:28.656827Z","iopub.execute_input":"2022-09-27T12:13:28.657293Z","iopub.status.idle":"2022-09-27T12:13:28.683597Z","shell.execute_reply.started":"2022-09-27T12:13:28.657257Z","shell.execute_reply":"2022-09-27T12:13:28.682116Z"}}
train.info()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-27T12:15:57.542493Z","iopub.execute_input":"2022-09-27T12:15:57.542934Z","iopub.status.idle":"2022-09-27T12:15:57.551764Z","shell.execute_reply.started":"2022-09-27T12:15:57.542901Z","shell.execute_reply":"2022-09-27T12:15:57.550531Z"}}
train.shape[0]

# %% [code] {"execution":{"iopub.status.busy":"2022-09-27T12:13:53.038983Z","iopub.execute_input":"2022-09-27T12:13:53.039418Z","iopub.status.idle":"2022-09-27T12:13:53.249914Z","shell.execute_reply.started":"2022-09-27T12:13:53.039383Z","shell.execute_reply":"2022-09-27T12:13:53.248587Z"},"jupyter":{"outputs_hidden":true}}
train.value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2022-09-27T12:09:47.862213Z","iopub.execute_input":"2022-09-27T12:09:47.862701Z","iopub.status.idle":"2022-09-27T12:09:47.885509Z","shell.execute_reply.started":"2022-09-27T12:09:47.862654Z","shell.execute_reply":"2022-09-27T12:09:47.884683Z"}}
train.head(100)

# %% [code] {"execution":{"iopub.status.busy":"2022-09-27T12:10:17.341557Z","iopub.execute_input":"2022-09-27T12:10:17.341974Z","iopub.status.idle":"2022-09-27T12:10:17.373401Z","shell.execute_reply.started":"2022-09-27T12:10:17.341943Z","shell.execute_reply":"2022-09-27T12:10:17.372018Z"}}
sample_submission.head(100)

# %% [code]

