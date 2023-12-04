#!/usr/bin/env python
# coding: utf-8

# ## 01 Python Libraries

# In[1]:


from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import os
import gzip


# ## 02a Reading and Loading the Darpa dataset from raw text format

# In[372]:




# Define the days of the week
days_of_week = ["monday7", "tuesday7", "wednesday7", "thursday7", "friday7"]

# Initialize an empty list to store the contents of the files
merged_list = []

# Iterate through each day's folder
for day in days_of_week:
    folder_path = f"./{day}/"  # Update the path according to your directory structure

    # Check if the folder exists
    if os.path.exists(folder_path):
        # Read the contents of bcm.list.gz file
        bcm_file_path = os.path.join(folder_path, "bcm.list.gz")
        if os.path.exists(bcm_file_path):
            with gzip.open(bcm_file_path, "rt") as bcm_file:
                merged_list.extend(bcm_file.readlines())

        # Read the contents of tcpdump.list.gz file
        tcpdump_file_path = os.path.join(folder_path, "tcpdump.list.gz")
        if os.path.exists(tcpdump_file_path):
            with gzip.open(tcpdump_file_path, "rt") as tcpdump_file:
                merged_list.extend(tcpdump_file.readlines())

# Save the merged list to a file named "week2_data.txt"
output_file_path = "week7_data.txt"
with open(output_file_path, "w") as output_file:
    output_file.writelines(merged_list)

print(f"Merged list saved to {output_file_path}")


# In[373]:


import pandas as pd

# Load data from week1_data.txt
week1_file_path = "week1_data.txt"
with open(week1_file_path, "r") as week1_file:
    week1_data = week1_file.readlines()

# Load data from week2_data.txt
week2_file_path = "week2_data.txt"
with open(week2_file_path, "r") as week2_file:
    week2_data = week2_file.readlines()

# Load data from week3_data.txt
week3_file_path = "week3_data.txt"
with open(week3_file_path, "r") as week3_file:
    week3_data = week3_file.readlines()
    
# Load data from week4_data.txt
week4_file_path = "week4_data.txt"
with open(week4_file_path, "r") as week4_file:
    week4_data = week4_file.readlines()
    

# Load data from week5_data.txt
week5_file_path = "week5_data.txt"
with open(week5_file_path, "r") as week5_file:
    week5_data = week5_file.readlines()

# Load data from week6_data.txt
week6_file_path = "week6_data.txt"
with open(week6_file_path, "r") as week6_file:
    week6_data = week6_file.readlines()
    
# Load data from week7_data.txt
week7_file_path = "week7_data.txt"
with open(week7_file_path, "r") as week7_file:
    week7_data = week7_file.readlines()
    
# Merge the two lists
data_full_7wks = week1_data + week2_data + week3_data + week4_data + week5_data + week6_data +week7_data

# Save the merged data to a text file
output_txt_path = "data_full_7wks.txt"
with open(output_txt_path, "w") as output_txt:
    output_txt.writelines(data_full_7wks)

print(f"data_full_7wks data saved to {output_txt_path}")


# In[374]:


len(data_full_7wks)


# ### 02b Converting processed data into CSV

# In[3]:


# Read merged_data.txt into a DataFrame
merged_data_file = "04_dataset_7weeks.txt"
df_full = pd.read_csv(merged_data_file, delim_whitespace=True, header=None)

# Select specific columns from df_full, not df
df_full = df_full.iloc[:, [1,2,3,4,5,6,7,8,9]]

# Rename columns
df_full.columns = ['date', 'time', 'duration', 'server', 'sourcePort', 'destinationPort','srcIP','destIP', 'anomaly']

# Print the resulting DataFrame
df_full


# In[6]:



# Read merged_data.txt into a DataFrame
#merged_data_file = "data_1_2.txt"
merged_data_file = "04_dataset_7weeks.txt"

df = pd.read_csv(merged_data_file, delim_whitespace=True, header=None)

# Select specific columns
df = df.iloc[:, [1, 2, 5, 6, 9]]

# Rename columns
df.columns = ['date', 'time', 'source', 'destination', 'anomaly']

# Print the resulting DataFrame
df


# In[7]:


print(df[1:2])


# ## 03 Visualizing   dataset and feature engineering

# In[8]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Assuming you have loaded your dataset into a DataFrame named 'df'
# If not, load the dataset as described in the previous responses

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph using the 'source' and 'destination' columns
for _, row in df[0:20000].iterrows():
    G.add_edge(row['source'], row['destination'])

# Plot the directed graph
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G)  # You can choose a different layout algorithm
nx.draw(G, pos, with_labels=True, font_size=8, node_size=1000, font_color="black", font_weight="bold", node_color="skyblue", arrowsize=10)

plt.title("DARPA-1998: Directed Graph of IP Addresses")
# Save the figure
plt.savefig("DARPA_directed_graph.png")
plt.show()


# In[378]:


# exclude collapsing data
tcp = df
tcp = tcp[tcp.date != '07/32/1998']

# acquire datetime information
tcp['date_time'] = pd.to_datetime(tcp['date'] + ' ' + tcp['time'], errors='coerce', format='%m/%d/%Y %H:%M:%S')
tcp = tcp.drop(['date', 'time'], axis=1)


# In[379]:


# exclude collapsing data
tcp = df
tcp = tcp[tcp.date != '07/32/1998']

# acquire datetime information
tcp['date_time'] = tcp['date'] + ' ' + tcp['time']
tcp = tcp.drop('date', axis=1)
tcp = tcp.drop('time', axis=1)
tcp['date_time'] = pd.to_datetime(tcp['date_time'],errors='coerce',  format='%m/%d/%Y %H:%M:%S')


# In[380]:


tcp.anomaly.value_counts()


# In[381]:


tcp


# In[382]:


# calculate how many hours passed since the initial time
initial_time = tcp['date_time'].min()
tcp['date_time'] = tcp['date_time'] - initial_time
tcp['hours_past'] = tcp['date_time'].dt.days * 24 + tcp['date_time'].dt.seconds//3600
tcp = tcp.drop('date_time', axis=1)
tcp = tcp.sort_values('hours_past')


# In[383]:


tcp.hours_past.fillna(0, inplace=True)
# Remove rows with '-' values across the entire DataFrame
tcp = tcp.replace('-', pd.NA).dropna()
len(tcp)


# In[389]:


# Search for special characters in the column
special_char_rows = tcp[tcp['source'].astype(str).str.contains('[^0-9]')]
special_char_rows2 = tcp[tcp['destination'].astype(str).str.contains('[^0-9]')]

# Display the DataFrame with rows containing special characters
print(special_char_rows)
print(special_char_rows2)


# In[391]:


# remove such rows
#tcp.loc[477972]

# Remove the row with index 49724
tcp = tcp.drop(477972, axis=0)


# In[392]:


# Convert the column to integers
tcp['source'] = tcp['source'].apply(lambda x: int(''.join(char for char in str(x) if char.isdigit())))
tcp['destination'] = tcp['destination'].apply(lambda x: int(''.join(char for char in str(x) if char.isdigit())))

# Search for special characters in the column
special_char_rows = tcp[tcp['source'].astype(str).str.contains('[^0-9]')]
special_char_rows2 = tcp[tcp['destination'].astype(str).str.contains('[^0-9]')]

# Display the DataFrame with rows containing special characters
print(special_char_rows)
print(special_char_rows2)


# In[393]:


tcp.destination


# In[ ]:





# In[394]:


tcp


# In[395]:


# Find rows where the special character is present in any column
rows_with_special_char = tcp[tcp.applymap(lambda x: '<' in str(x)).any(axis=1)]

# Display the rows with the special character
print(rows_with_special_char)


# In[396]:


graphs = tcp.loc[:, ['source', 'destination', 'hours_past']]
graphs = graphs.values


# In[397]:


len(graphs)


# ## 04  SpotLight Algorithm in python

# In[398]:


class SpotLight:
    def __init__(self, graphs):
        '''
        input should be Nx3 array. (N: number of total edges)
        first column: source node
        second column: destination node
        third column: timestamp
        '''
        self.graphs = graphs
        self.timestamps = np.unique(graphs[:,2])
        self.num_of_timestamp = self.timestamps.shape[0]

    def sketch(self, K=50, p=0.2, q=0.2):
        '''
        K: number of subgraphs
        p: source sampling probability
        q: destination sampling probability
        '''
        sketched_vectors = np.empty((0, K), int)
        for i in tqdm(range (self.num_of_timestamp)):
            sketched_vector = np.empty((0, K), int)
            graph = self.graphs[self.graphs[:, 2] == (self.timestamps[i])]
            self.source_nodes = np.unique(graph[:,0]).reshape((-1, 1))
            self.dest_nodes = np.unique(graph[:,1]).reshape((-1, 1))
            self.hashing(K, p, q)
            for j in range(graph.shape[0]):
                source = graph[j, 0]
                dest = graph[j, 1]
                sources_are_in_subgraphs = (self.subgraphs_source[self.subgraphs_source[:, 0] == source])[:,1:]
                dests_are_in_subgraphs = (self.subgraphs_dest[self.subgraphs_dest[:, 0] == dest])[:,1:]
                sketched_vector = np.append(sketched_vector, sources_are_in_subgraphs * dests_are_in_subgraphs, axis=0)
            sketched_vector = np.sum(sketched_vector, axis=0).reshape((1, K))
            sketched_vectors = np.append(sketched_vectors, sketched_vector, axis=0)
        return sketched_vectors
        

    def hashing(self, K, p, q):
        self.subgraphs_source = np.random.choice([0,1], [self.source_nodes.shape[0], K], p = [1-p, p])
        self.subgraphs_source = np.concatenate((self.source_nodes, self.subgraphs_source), axis=1)
        self.subgraphs_dest = np.random.choice([0,1], [self.dest_nodes.shape[0], K], p = [1-q, q])
        self.subgraphs_dest = np.concatenate((self.dest_nodes, self.subgraphs_dest), axis=1)


# In[399]:


# sketching tcodump data to spotlight space.
SL = SpotLight(graphs)
v_g = SL.sketch(50, 0.2, 0.2)


# In[400]:


v_g.shape


# In[410]:


# Convert the NumPy array to a Pandas DataFrame
spotligh_space = pd.DataFrame(v_g)
# Save the DataFrame to a CSV file
spotligh_space.to_csv('spotlight_space.csv', index=False)


# ## 04a Anomaly Score with Isolation Forest

# In[435]:


# anomaly detection based on Isolation Forest
from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=100, max_samples=80, contamination=0.4)
clf.fit(v_g)
detected = clf.score_samples(v_g)
detected = detected * -1


# In[436]:


# groupby timestamp, timestamps which contain more than 1000 anomalous communication are anomalous timestamps
truth = tcp.groupby('hours_past').sum()
truth = ((truth.anomaly.values > 1000)*1)


# In[437]:


len(truth)


# In[448]:


len(detected)


# In[439]:


from sklearn.metrics import precision_recall_curve, auc
precision_1, recall_1, thresholds = precision_recall_curve(truth, detected)
print('Area Under Curve:', auc(recall_1, precision_1))


# In[440]:


# draw recall precision curve
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
pyplot.plot(recall_1, precision_1, marker='.')


# In[408]:


# draw recall precision curve
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
pyplot.plot(recall, precision, marker='.')
# Save the plot
pyplot.savefig('recall_precision_curve.png')

# Display the plot
pyplot.show()


# ## 4b Random Cut Trees.

# In[441]:


from sklearn.ensemble import RandomTreesEmbedding
# Build the Random Cut Trees model

# RandomTreesEmbedding is used here, but it's essentially a Random Cut Trees model.
rct_model = RandomTreesEmbedding(n_estimators=100, max_depth=10, random_state=42)
rct_model.fit(v_g)
#detected = rct_model.score_samples(v_g)
#detected = detected * -1
# Use decision_function for scoring
decision_scores = rct_model.apply(v_g).sum(axis=1)

# Convert decision scores to anomaly scores (multiply by -1 to obtain positive scores)
#anomaly_scores = -decision_scores
anomaly_scores = decision_scores * 1
# Print the anomaly scores
print("Anomaly Scores:")
print(anomaly_scores)


# In[443]:


from sklearn.metrics import precision_recall_curve, auc
precision_2, recall_2, thresholds = precision_recall_curve(truth, anomaly_scores)
print('Area Under Curve:', auc(recall_2, precision_2))


# ## 05 Comparing IsolatedForest vs RandomCutTree for Anomaly Detection

# In[447]:


from matplotlib import pyplot

# Assuming you have the data for recall and precision in the variables recall_1, precision_1, recall_2, precision_2

# Plot the first set of recall-precision data
pyplot.plot(recall_1, precision_1, marker='.', label='SL: IsolatedForest')

# Plot the second set of recall-precision data
pyplot.plot(recall_2, precision_2, marker='.', label='SL: RandomCutTree')

# Label the x-axis
pyplot.xlabel('Recall')

# Label the y-axis
pyplot.ylabel('Precision')

# Add a title to the plot
pyplot.title('Recall-Precision Curve')

# Display a legend indicating which line corresponds to which model
pyplot.legend()

# Save the plot (you can uncomment this line if you want to save the plot as an image file)
pyplot.savefig('03_recall_precision_curve_both.png')

# Display the plot
pyplot.show()


# In[ ]:




