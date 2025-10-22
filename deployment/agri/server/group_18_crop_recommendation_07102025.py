#!/usr/bin/env python
# coding: utf-8

# #### <h1 align=center><font size = 5>Capstone Project -18 : Crop Recommendation System</font></h1>
# <h2 align=center><font size = 5>AIML Certification Programme</font></h2>

# ## Team <br>
# 1. Anuj Alex (2024AIML009)​
# 2. Gurbachan Singh Kalwan (2024AIML004)​
# 3. Krishna Murthy P (2024AIML078)​
# 4. Sidharth Gupta (2024AIML017)
# 5. Sree Rama Kumar Yeddanapudi (2024AILML008)​
# 
# Mentor: Prof. Aniruddha Dasgupta

# In[5]:


import numpy as np # linear algebra
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import re
from rapidfuzz import process, fuzz
from joblib import dump


# ### 1. Loading and Reading Dataset

# In[6]:


df = pd.read_csv("data/AP_data.csv")
df.rename(columns={"Extent\n(AC)":"Farm_Acres","Crop before":"Crop_Sown"},inplace=True)
df['District'] = df['District'].replace({
    "Anantapur": "Ananthapur",
    "S.P.S.Nellore": "Nellore",
    "S.P.S. Nellore": "Nellore",
    "Kadapa YSR": "Kadapa"
})
df.head()


# In[7]:


dist_master = pd.read_csv("data/AP_district_level_master.csv")
dist_master.head()


# ### 1.1 Merge datasets

# In[8]:


# Merge the DataFrames using left_on and right_on
rain_df= dist_master[["District","Kharif_rain",	"Rabi_rain",	"Zaid_rain"]].drop_duplicates()
# print(rain_df.head())
merged_df = pd.merge(df,rain_df, left_on='District', right_on='District',how='left')


# ### 1.2 Remove unnecessary columns

# In[9]:


merged_df.drop(columns=["Sl no", "Date", "Farmer No", "Macro/ Micro nutrient", "Farmer Name", "Fathers Name", "Time"
                        , "Recommended Sowing Time", "Season", "Farm_Acres", "Survey No.","Latitude","Longitude"], inplace=True)


# In[10]:


orig_df = df
df = merged_df


# ### 1.3 Check basic data statistics (shape, info, describe)

# In[11]:


df.shape


# In[12]:


df.describe()


# In[13]:


df.info()


# In[14]:


cols_to_convert = ['OC', 'Avail-S', 'Avail-B']
for col in cols_to_convert:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')


# ### 1.4 Display unique and sample values for columns

# In[15]:


for col in df.columns:
    print(col, df[col].nunique(),df[col].unique()[0:20],'\n')


# ### 2. Data Preprocessing

# ### 2.1 Check for Data Quality Issues
# 
# * duplicate data
# * missing data
# * data inconsistencies

# In[16]:


# duplicate data
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")


# In[17]:


# missing data
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]
print(missing_data.sort_values(ascending = False))


# ### 2.2 Handle missing values for numerical attributes

# In[18]:


numerical_cols = df.select_dtypes(include=['float64','int64']).columns


# In[19]:


numerical_cols


# In[20]:


for col in numerical_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)


# In[21]:


# missing data
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]
print(missing_data.sort_values(ascending = False))


# ### 2.3 Handle missing values for Soil

# In[22]:


# --- Master Soil categories ---
master_categories = [
    "Black", "Red", "Sandy", "Loam", "Clay",
    "Brown", "Yellow", "White", "Laterite",
    "Saline", "Alkaline", "Alluvial",
    "Gravel/Stony", "Mixed", "Other"
]

# --- Known corrections / synonyms ---
direct_map = {
    # Misspellings
    "RED": "Red",
    "red": "Red",
    "Red soil": "Red",
    "res": "Red",
    "redsoil": "Red",
    "redsoils": "Red",
    "red sandy loam": "Red",
    "red sandy": "Red",
    "red sandy\\": "Red",
    "redsandy": "Red",
    "redsandylo": "Red",
    "redsand": "Red",
    "redloam": "Red",
    "redbrown": "Red",
    "red grey": "Red",
    "red masari": "Red",

    "black soil": "Black",
    "Black Soil": "Black",
    "BLACK": "Black",
    "black clay": "Black",
    "black sandy": "Black",
    "black sandy loam": "Black",
    "black loam": "Black",
    "deepblack": "Black",
    "deep black": "Black",
    "normal bla": "Black",
    "black mix": "Black",
    "black mara": "Black",


    "sandi": "Sandy",
    "sanday": "Sandy",
    "sanday+bla": "Sandy",
    "sanday mix": "Sandy",
    "sand mixed": "Sandy",
    "sand mix": "Sandy",
    "sandy loam": "Sandy",
    "sandy mixe": "Sandy",
    "sandy with": "Sandy",
    "sandy brow": "Sandy",
    "sandy whit": "Sandy",


    "clayey loam": "Clay",
    "clay soil": "Clay",
    "caly soil": "Clay",
    "clay-sandy": "Clay",
    "silty clay": "Clay",

    "broan clay": "Brown",
    "brown light": "Brown",
    "brown dark": "Brown",
    "light brow": "Brown",
    "dark brown": "Brown",

    "alkhaline": "Alkaline",
    "alkline": "Alkaline",
    "alkline +": "Alkaline",
    "black alka": "Alkaline",

    "saline soi": "Saline",
    "salain": "Saline",
    "salty": "Saline",
    "salain mix": "Saline",
    "saline mix": "Saline",

    "laterite s": "Laterite",
    "laterite l": "Laterite",
    "laterite m": "Laterite",
    "laterite u": "Laterite",
    "latritate": "Laterite",
    "red lateri": "Laterite",

    "loamy soil": "Loam",
    "loomy": "Loam",
    "loami": "Loam",
    "laomy": "Loam",
    "soil loamy": "Loam",
    "loamy brow": "Loam",
    "ORTHIDS": "Loam",

    "white gara": "White",
    "white mixe": "White",
    "white soil": "White",
    "whitebrown": "White",
    "white sand": "White",
    "white red": "White",
    "white yell": "White",
    "brown whit": "White",

    "yellowblac": "Yellow",
    "yellowbrow": "Yellow",
    "yellow red": "Yellow",
    "white yello": "Yellow",

    # Loam/Alluvial
    "alluvial s": "Alluvial",
}
direct_map.update({
    # Sandy soils
    "SANDY ALFISOL": "Sandy",
    "SANDY ALFISOLS": "Sandy",
    "PSSAMENTS": "Sandy",
    "PSSAMNETS": "Sandy",
    "INNCEPTISOLS": "Sandy",
    "INSEPTISOLS": "Sandy",

    # Loam soils / Alfisols / Inceptisols variants
    "ORTHIDS": "Loam",
    "LOAMY ALFISOLS": "Loam",
    "LOAMY ALFISOL": "Loam",
    "USTALF/USTOLLS": "Loam",
    "UDUPTS/UDALFS": "Loam",
    "UDOLLS/UDALFS": "Loam",
    "INCEPTISOLS": "Loam",

    # Black soils / Vertisols variants
    "VERTISOLS": "Black",
    "VERTIC SOILS": "Black",
    "VERTIC SOLS": "Black",
    "VERTI SOLS": "Black",
    "VRTIC SOILS": "Black",
    "VERRTISOLS": "Black",
    "VERTIC OSILS": "Black",
})
# --- Local overrides (dialect → base class) ---
overrides = {
    "chowdu": "Red",
    "nalla regadi": "Black",
    "regadi": "Red",
    "sowdu": "Red",
    "sudda": "Red",
    "thella kattu": "White",
    "sudda neela": "Clay",
    "tella masaka": "White",
    "erra maska": "Red",
    "savudu": "Red",
    "garuku": "Other",
    "garasu": "Red",
    "garasu mix": "Red",
    "garsu mix": "Red",
    "mosari": "Red",
    "masari": "Red",
    "masali": "Red",
    "masale": "Red",
    "masori": "Red",
    "madikattu": "Red",
    "maradi": "Red",
    "marad": "Red",
    "mardi": "Red",
    "marali": "Red",
    "moram": "Red",
    "maralugodu": "Red",

    "murrum": "Black",
    "murum soil": "Black",
    "medium bla": "Black",
    "m black": "Black",
    "black muri": "Black",
    "humpli bla": "Black",

    "kari": "Black",
    "kapu": "Black",
    "kappu": "Black",

    "kemp": "Red",
    "kempu": "Red",
    "k-r": "Red",
    "r-k": "Red",
    "r-m": "Red",
    "m-r": "Red",
}


# In[23]:


print(master_categories)


# In[24]:


print(direct_map)


# In[25]:


def clean_text(txt: str) -> str:
    txt = str(txt).lower().strip()
    txt = re.sub(r"soil", "", txt)
    txt = re.sub(r"[^a-z\s\+\-]", "", txt)
    return txt.strip()


# In[26]:


def standardize_soil(raw: str) -> str:
    if not raw or not isinstance(raw, str) or raw.strip() == "":
        return "Other"
    text = raw.lower()
    text = clean_text(raw)
    for key, val in overrides.items():
        if key in text:
            return val
    for key, val in direct_map.items():
        if key in text:
            return val
    # if text in direct_map:
    #     return direct_map[text]

    match, score, _ = process.extractOne(text, master_categories, scorer=fuzz.WRatio)
    if score >= 80:
        return match
    return "Other"


# In[27]:


# Test Soil Function works for samples
samples = [
    "Black Soil", "redsoil", "Chowdu", "Alkline +",
    "Saline Soi", "Laterite m", "Broan Clay",
    "Sanday+bla", "White gara", "Masari", "Murum Soil",
    "nalla regadi", "redsandylo", "Random Gibberish", "", None
    ,"RED", "BLACK", "Red soil", "Black Soil"

]

print("--- Soil Type Standardization Examples ---")
for s in samples:
    # For each sample, print the original string and its standardized version
    standardized_value = standardize_soil(s)
    print(f"'{s}' → '{standardized_value}'")


# ### 2.4 Handle missing values for Crop

# In[28]:


crop_map = {
    # Cereals (Grains & Millets)
    "maize": "Maize", "mazi": "Maize", "sweetcorn": "Maize",
    "jowar": "Sorghum", "jonna": "Sorghum", "mahendra jonna": "Sorghum",
    "pacha jonna": "Sorghum", "erra jonna": "Sorghum",
    "bajra": "Pearl Millet",
    "korra": "Foxtail Millet",
    "dhanyalu": "Other Millet",
    "ragi": "Ragi",
    "vari": "Rice", "paddy": "Rice", "paady": "Rice",
    "rice": "Rice",
    "millet": "Other Millet",
    "finger millet": "Ragi",
    "pearl millet": "Pearl Millet",
    "wheat": "Wheat",
    "barley": "Barley",

    # Pulses
    "bengalgram": "Chickpea", "senaga": "Chickpea", "erra senaga": "Chickpea",
    "chickpea": "Chickpea",
    "red gram": "Pigeonpea", "redgram": "Pigeonpea",
    "pigeonpea": "Pigeonpea",
    "green gram": "Green Gram", "mung": "Green Gram",
    "black gram": "Black Gram", "blackgram": "Black Gram",
    "horse gram": "Horse Gram", "horsegram": "Horse Gram",
    "cowpea": "Cowpea", "cow pea": "Cowpea",
    "rajma": "Rajma", "peasara": "Other Pulse", "pulse": "Other Pulse",
    "minor pulses": "Other Pulse",

    # Oilseeds
    "ground nut": "Groundnut", "groundnut": "Groundnut", "g.nut": "Groundnut",
    "grounat": "Groundnut", "ground nat": "Groundnut",
    "veru senaga": "Groundnut",
    "castor": "Castor", "clastor": "Castor",
    "sesamum": "Sesame", "sesumum": "Sesame",
    "sunflower": "Sunflower",
    "linseed": "Linseed",
    "rapeseed": "Rapeseed and Mustard", "mustard": "Rapeseed and Mustard",
    "soyabean": "Soyabean", "soybean": "Soyabean",
    "safflower": "Safflower",

    # Cash crops
    "cotton": "Cotton", "cottan": "Cotton",
    "sugarcane": "Sugarcane", "suger cane": "Sugarcane",
    "sugar cane": "Sugarcane",
    "tobacco": "Tobacco", "pogaku": "Tobacco",
    "oil palm": "Oil Palm",
    "eucalyptus": "Eucalyptus", "eucaliptus": "Eucalyptus",

    # Fruits
    "banana": "Banana", "cocnut": "Coconut", "coconut": "Coconut",
    "papaya": "Papaya", "anaar": "Pomegranate",
    "mango": "Mango",
    "citrus": "Citrus",
    "lime": "Lime", "lemon": "Lime",
    "cashew": "Cashew", "cashewnut": "Cashew", "cashew nut": "Cashew",
    "cashew raina": "Cashew",
    "cocoa": "Cocoa", "cocoa+ coconut": "Cocoa",
    "fruits": "Fruits",

    # Vegetables
    "brinjal": "Brinjal",
    "tomato": "Tomato", "tamato": "Tomato",
    "benda": "Okra", "okra": "Okra",
    "cabbage": "Cabbage", "cabage": "Cabbage",
    "cucumber": "Cucumber",
    "potato": "Potato", "potatao": "Potato", "potatoes": "Potato",
    "onion": "Onion", "onian": "Onion", "oniyan": "Onion",
    "ridge guard": "Ridge Gourd", "donda": "Ridge Gourd",
    "yam": "Yam",
    "vegetable": "Vegetables", "vegetables": "Vegetables",

    # Spices & condiments
    "chilli": "Chilli", "chill": "Chilli", "chillies": "Chilli",
    "chilly": "Chilli", "mirchi": "Chilli",
    "turmeric": "Turmeric", "turmaric": "Turmeric",
    "coriandam": "Coriander",

    # Plantation crops
    "mulberry": "Mulberry", "mulbarry": "Mulberry", "mulberrry": "Mulberry",
    "coffee": "Coffee",
    "red sandal": "Sandalwood", "sandal": "Sandalwood",

    # Others
    "fodder crops": "Fodder", "fodder": "Fodder",
    "flowers": "Flowers",
    "prawns": "Aquaculture",
}


# In[29]:


def standardize_crop(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip(): return "Other"
    text = raw.lower()
    text = re.sub(r'[/;()$$$$+]', ',', text)
    text = re.sub(r'\b(intercrop|and|crops?)\b', ',', text, flags=re.IGNORECASE)
    potential_crops = [c.strip() for c in text.split(',') if c.strip()]
    standardized_crops = set()
    for crop_text in potential_crops:
        if crop_text in crop_map:
            standardized_crops.add(crop_map[crop_text])
            continue
        match, score, _ = process.extractOne(crop_text, crop_map.keys(), scorer=fuzz.WRatio)
        if score >= 85:
            standardized_crops.add(crop_map[match])
    if not standardized_crops: return "Other"
    crop_str = str(list(standardized_crops)[0])
    return crop_str


# In[30]:


# # Test Crop Function works for samples
# tests = [
#     "Cashewnut, Mango (Intercrop Maize And Seasamum)",
#     "Topioca, Cashew",
#     "Paddy/Maize/G.Nut",
#     "Cotton;Red Gram",
#     "Oil Palm+Coconut",
#     "Paddy/ Sugarcane",
#     "Mirchi",
#     "Erra Senaga"
# ]

# for t in tests:
#     print(t, "→", standardize_crop(t))


# In[31]:


# 2. Standardize Soil Type
df['Soil_Type_Standard'] = df['Soil type'].apply(standardize_soil)
df['Crop_Sown_Standard'] = df['Crop_Sown'].apply(standardize_crop)


# In[32]:


print(df.head().to_string())
print(df.tail().to_string())



# In[33]:


# missing data

missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]
print(missing_data.sort_values(ascending = False))


# ### 3. Exploratory Data Analysis

# ### 3.1 EDA - Soil Distribution

# In[34]:


plt.figure(figsize=(6, 4))
soil_counts_orig = orig_df['Soil type'].value_counts()
sns.barplot(x=soil_counts_orig.index, y=soil_counts_orig.values)
plt.title('Soil Distribution (Class Balance) before Feature Engg')
plt.xlabel('Soil Type')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[35]:


plt.figure(figsize=(6, 4))
soil_counts = df['Soil_Type_Standard'].value_counts()
sns.barplot(x=soil_counts.index, y=soil_counts.values)
plt.title('Soil Distribution (Class Balance) after Feature Engg')
plt.xlabel('Soil Type')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
plt.show()


# ### 3.2 EDA - Crop Distribution

# In[36]:


plt.figure(figsize=(6, 4))
soil_counts_orig = orig_df['Crop_Sown'].value_counts()
sns.barplot(x=soil_counts_orig.index, y=soil_counts_orig.values)
plt.title('Crop Distribution (Class Balance) before Feature Engg')
plt.xlabel('Crop Sown')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[37]:


top_25_crops = df['Crop_Sown_Standard'].value_counts().head(25).index.tolist()
print(top_25_crops)
# # Replace crops not in top 5 with 'others'
df['Crop_Sown_Standard'] = df['Crop_Sown_Standard'].apply(lambda x: x if x in top_25_crops else 'Other')
print(df['Crop_Sown_Standard'].unique())


# In[38]:


plt.figure(figsize=(11, 4))
crop_counts = df['Crop_Sown_Standard'].value_counts()
sns.barplot(x=crop_counts.index, y=crop_counts.values)
plt.title('Crop Distribution (Class Balance) after Feature Engg')
plt.xlabel('Crop Sown')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[39]:


# # Set plotting style
# sns.set(style="whitegrid")

# # ✔ Distribution of Nutrients
# print("\n--- EDA: Distribution of Nutrients ---")
# # Update nutrient_cols to match the columns in df_processed (using hyphens)
# nutrient_cols = ['pH', 'EC', 'OC', 'Avail-P', 'Exch-K', 'Avail-S', 'Avail-Zn', 'Avail-B', 'Avail-Fe', 'Avail-Cu', 'Avail-Mn']

# plt.figure(figsize=(15, 12))
# for i, col in enumerate(nutrient_cols):
#     plt.subplot(4, 3, i + 1) # Adjust subplot layout
#     sns.histplot(df[col], kde=True, bins=20) # Use df_processed instead of data
#     plt.title(f'Distribution of {col}')
# plt.tight_layout()
# plt.show()


# In[40]:


sns.set(style="whitegrid")
nutrient_cols = ['pH', 'EC', 'OC', 'Avail-P', 'Exch-K', 'Avail-S', 'Avail-Zn', 'Avail-B', 'Avail-Fe', 'Avail-Cu', 'Avail-Mn']
df[nutrient_cols].hist(bins=30, figsize =(20,20))
plt.show()


# In[41]:


plt.figure(figsize=(20, 10))
sns.heatmap(df[numerical_cols].corr(), cmap="Blues",annot=True)


# In[45]:


print(df.head().to_string())
# Remove Categorical variables
df_new = df.copy(deep=True)
#Encode 3 categorical variables which need to be converted to numerical format
le = LabelEncoder()  
df_new['District_encoded']=le.fit_transform(df_new['District'])
dump(le,"District_LE.joblib")
df_new['Mandal_encoded']=le.fit_transform(df_new['Mandal'])
dump(le,"Mandal_LE.joblib")
df_new['Village_encoded']=le.fit_transform(df_new['Village'])
dump(le,"Village_LE.joblib")
df_new['Soil_Type_Standard_encoded']=le.fit_transform(df_new['Soil_Type_Standard'])
dump(le,"Soil_Type_Standard_LE.joblib")
df_new['Crop_Sown_Standard_encoded']=le.fit_transform(df['Crop_Sown_Standard'])
dump(le,"Crop_Sown_Standard_LE.joblib")
df_new = df_new.drop(['District', 'Mandal', 'Village',"Crop_Sown","Soil type","Crop_Sown_Standard","Soil_Type_Standard"], axis=1)
print("Print Clean DF")
print(df_new.head().to_string())



# In[46]:


#Split the dataset into train and test with stratification
X = df_new.drop("Crop_Sown_Standard_encoded", axis=1)
y = df_new["Crop_Sown_Standard_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y  # stratify based on y
)

y_test_c = y_test.copy(deep=True)


# In[47]:


#Data scaling
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train.to_numpy())
dump(scaler,"Standard_Scaler.joblib")
var_list = list(X_train.columns)
append_str = '_N'
var_list_N = [sub + append_str for sub in var_list]

train_Norm = pd.DataFrame(scaled_train,
                        columns=var_list_N)
X_train_scaled = train_Norm.to_numpy()

scaled_test = scaler.transform(X_test.to_numpy())
test_Norm = pd.DataFrame(scaled_test,
                        columns=var_list_N)
X_test_scaled = test_Norm.to_numpy()


# In[48]:


pca = PCA(n_components=X_train_scaled.shape[1]//2+1) ## Limiting # of PCA components to half of X dimensions
print(X_train_scaled.shape)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print("PCA_Reg:", np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100))
print("This appears to indicate that more than half the features are important to explain the variability in crop selection")


# In[49]:


#import Gaussian NB Classifier

#Setup a gnb classifier
model_nb = GaussianNB()

#Fit the model
# model_nb.fit(X_train_pca, y_train)
# y_pred_nb = model_nb.predict(X_test_pca)

model_nb.fit(X_train_scaled, y_train)
y_pred_nb = model_nb.predict(X_test_scaled)

##Compute accuracy on the training set
print("Gaussian NB Train Accuracy:",model_nb.score(X_train_scaled, y_train))

# #Compute accuracy on the test set
print("Gaussian NB Test Accuracy:",model_nb.score(X_test_scaled, y_test_c))

# #Evaluate NB model
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

model_nb.fit(X_train_pca, y_train)
y_pred_pca_nb = model_nb.predict(X_test_pca)
##Compute accuracy on the training set
print("Gaussian NB Train Accuracy PCA:",model_nb.score(X_train_pca, y_train))

# #Compute accuracy on the test set
print("Gaussian NB Test Accuracy PCA:",model_nb.score(X_test_pca, y_test_c))
# 6. Evaluate PCA model
print("Accuracy:", accuracy_score(y_test, y_pred_pca_nb))
print("Classification Report:\n", classification_report(y_test_c, y_pred_pca_nb))


# In[50]:


# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_pca, y_train)
y_pred_rf = rf.predict(X_test_pca)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_pca, y_train)
y_pred_gb = gb.predict(X_test_pca)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Gradient Boosting Classification Report:\n", classification_report(y_test, y_pred_gb))


# ## Creating a Forward Selection Strategy for PCA Feature Selection

# In[51]:


def plot_complexity_curve(k_list, knn_model, x_train, x_test, y_train, y_test):
    train_scores = []
    test_scores = []
    knn_test_scores = pd.DataFrame()
    # For each k
    for k in k_list:
        # Initialize, fit, predict
        knn = knn_model(k, weights=knn_wt, metric='manhattan')
        knn.fit(x_train, y_train)
        test_scores.append(knn.score(x_test, y_test))

    knn_test_scores["K_value"] = pd.Series(k_list)
    knn_test_scores["TestScore"] = pd.Series(test_scores)
    # print("KNN Fit done. Returning DF with scores")
    return knn_test_scores


# In[53]:


np.random.seed(12345)
knn_wt = 'distance'


all_res = pd.DataFrame()
all_preds = pd.DataFrame()
#
for i in range(1,X_train_pca.shape[1]+1):
    res=pd.DataFrame()
    preds = pd.DataFrame()
    preds["y_act"] = y_test_c  

    # print( X_train_pca[:, :i].shape,X_test_pca[:, :i].shape, y_train.shape,
    #                                 y_test.shape)
    # ## KNN Classifier
    k_score = plot_complexity_curve(range(1,50), KNeighborsClassifier, X_train_pca[:, :i], X_test_pca[:, :i], y_train,
                                    y_test_c)
    k_df = k_score[k_score["TestScore"] >= 0.995 * k_score["TestScore"].max()]
    k_opt = int(k_df["K_value"].min())
    knn_model = KNeighborsClassifier(n_neighbors=k_opt, weights=knn_wt, metric='manhattan').fit(X_train_pca[:, :i],
                                                                                               y_train)
    # Score
    y_pred_knn = knn_model.predict(X_test_pca[:, :i])
    knn_accuracy = np.round(accuracy_score(y_test_c, y_pred_knn),4)

    #Setup a gnb classifier
    model_nb = GaussianNB()

    model_nb.fit(X_train_pca[:, :i], y_train)
    y_pred_nb = model_nb.predict(X_test_pca[:, :i])
    nb_accuracy = np.round(accuracy_score(y_test_c, y_pred_nb),4)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_pca[:, :i], y_train)
    y_pred_rf = rf.predict(X_test_pca[:, :i])
    rf_accuracy = np.round(accuracy_score(y_test_c, y_pred_rf),4)


    ##Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train_pca[:, :i], y_train)
    y_pred_gb = gb.predict(X_test_pca[:, :i])
    gb_accuracy = np.round(accuracy_score(y_test_c, y_pred_gb),4)


    ### logistic Regression
    LR = LogisticRegression(
    # multi_class='multinomial',  # Use 'multinomial' for softmax regression
    solver='lbfgs',             # Solver that supports multinomial
    max_iter=1000)
    LR.fit(X_train_pca[:, :i], y_train)
    # Predict
    y_pred_LR = LR.predict(X_test_pca[:, :i])
    LR_accuracy = np.round(accuracy_score(y_test_c, y_pred_LR),4)



    preds["KNN_pred"] = y_pred_knn
    preds["NB_pred"] = y_pred_nb
    preds["RF_pred"] = y_pred_rf
    preds["GB_pred"] = y_pred_gb
    preds["LR_pred"] = y_pred_LR
    preds["PCA_Cols"] = i  


    res["PCA_Cols"]=pd.Series(i)
    res["Knn_Kopt"]=pd.Series(k_opt)
    res["Knn_Acc"]=pd.Series(knn_accuracy)
    res["NaiveBayes_Acc"]=pd.Series(nb_accuracy)
    res["RF_Acc"]=pd.Series(rf_accuracy)
    res["GB_Acc"]=pd.Series(gb_accuracy)
    res["LogReg_Acc"]=pd.Series(LR_accuracy)

    all_res = pd.concat([all_res,res],ignore_index=True)
    all_preds = pd.concat([all_preds,preds],ignore_index=True)
    print("PCA Cols:",str(i), "Knn:",k_opt, knn_accuracy, "Naive Bayes:",nb_accuracy, "Random Forest:",rf_accuracy,"Gradient Boost", gb_accuracy,"Logistic Regression:",LR_accuracy)




# ### Plotting of results

# In[54]:


all_preds.to_csv("y_test_with_pred.csv")
all_res.to_csv("All_results.csv")

# Melt the dataframe to long format
all_res_melted = all_res.melt(
    id_vars="PCA_Cols",
    value_vars=["Knn_Acc", "NaiveBayes_Acc", "RF_Acc", "GB_Acc", "LogReg_Acc"],
    var_name="Model",
    value_name="Accuracy"
)

# Find the best model/component (highest accuracy)
best_row = all_res_melted.loc[all_res_melted["Accuracy"].idxmax()]
best_x = best_row["PCA_Cols"]
best_y = best_row["Accuracy"]
best_model = best_row["Model"]

# Create the plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=all_res_melted, x="PCA_Cols", y="Accuracy", hue="Model", marker="o")

# Highlight the best point
plt.scatter(best_x, best_y, color='red', s=150, edgecolor='black', zorder=5, label="Best Model")

# Annotate the best point
plt.text(best_x + 0.3, best_y, f'{best_model}\nAcc: {best_y}', color='red', fontsize=10)

# Add titles and labels
plt.title("Model Accuracy vs Number of PCA Components")
plt.xlabel("Number of PCA Components (PCA_Cols)")
plt.ylabel("Accuracy")
plt.xticks(all_res['PCA_Cols'].unique())
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ## Create Soft voting Ensemble Model

# In[55]:


###Selected from previous PCA Feature selection
best_x = 11
k_opt = 16
knn = KNeighborsClassifier(n_neighbors=k_opt, weights='distance', metric='manhattan')
nb = GaussianNB()
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
lr = LogisticRegression(solver='lbfgs', max_iter=1000)

ensemble = VotingClassifier(
    estimators=[
        ('KNN', knn),
        # ('NB', nb),
        ('RF', rf),
        ('GB', gb),
        # ('LR', lr)
    ],
    voting='soft'  
)

# Fit on training data
ensemble.fit(X_train_pca[:, :best_x], y_train)  # best_x from earlier (optimal PCA cols)
dump(ensemble,"Soft_Voting_Ensemble_Model.pkl")
dump(knn,"KNN_Model.pkl")
dump(rf,"Random_Forest.pkl")
dump(gb,"Gradient_Boosting.pkl")

# Predict and evaluate
y_pred_ensemble = ensemble.predict(X_test_pca[:, :best_x])
ensemble_accuracy = accuracy_score(y_test_c, y_pred_ensemble)

print(f"Ensemble Accuracy: {np.round(ensemble_accuracy, 4)}")
print("Ensemble Classification Report:\n", classification_report(y_test, y_pred_ensemble))


# In[ ]:





# In[ ]:




