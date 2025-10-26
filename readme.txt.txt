============================================================
            Model Generation for Crop Recommendation
============================================================

Project Title: Crop Recommendation System for Indian Agriculture
Group: 18
Members: 
[Anuj Alex (2024AIML009)​
Gurbachan Singh Kalwan (2024AIML004)​
Krishna Murthy P (2024AIML078)​
Sidharth Gupta (2024AIML017)
Sree Rama Kumar Yeddanapudi (2024AILML008)​]

------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------
This project builds and evaluates a machine learning model to provide Top-3 crop recommendations based on soil health, location, and climate data from Andhra Pradesh. The final model is an ensemble classifier trained on a focused dataset with advanced feature engineering.

------------------------------------------------------------
2. CONTENTS OF THE SUBMISSION
------------------------------------------------------------
This submission is organized into the following folders:

/1_Code/
  - crop_recommendation.ipynb: The main Jupyter Notebook containing the full, end-to-end workflow with explanations and visualizations.
  - crop_recommendation.py: A clean Python script version for direct execution.

/2_Data/
  - AP_data.csv: The primary soil health card dataset.
  - AP_district_level_master.csv: The supplementary district-level rainfall data.

/3_Model/
  - final_crop_model.pkl: The final, trained ensemble model saved as a pickle file, ready for deployment.

/4_Documentation/
  - README.txt: This instruction file.
  - Group_18_Crop Recommendation System_Report.pdf: The final project report document.

------------------------------------------------------------
3. STEPS TO REPRODUCE THE WORK
------------------------------------------------------------

**Prerequisites:**
- Python 3.8+
- A Python environment (e.g., Anaconda, venv).
- The required libraries can be installed via pip.

**Step 1: Install Required Libraries**
Open your terminal or command prompt and run the following command to install all necessary packages:
pip install pandas numpy seaborn matplotlib scikit-learn rapidfuzz imblearn lightgbm xgboost catboost

**Step 2: Place Data Files**
Ensure that the `AP_data.csv` and `AP_district_level_master.csv` files from the `/2_Data/` folder are placed in the same directory as the code files.

**Step 3: Execute the Code**
You have two options to run the project:

  A) Using the Jupyter Notebook (Recommended for review):
     1. Open `crop_recommendation.ipynb` in Jupyter Lab or Jupyter Notebook.
     2. From the menu, select "Kernel" -> "Restart & Run All".
     3. The notebook will execute from top to bottom, generating all analyses, visualizations, and saving the final `final_crop_model.pkl` file in the same directory.

  B) Using the Python Script:
     1. Open a terminal or command prompt.
     2. Navigate to the `/1_Code/` directory.
     3. Run the script using the command: python crop_recommendation.py
     4. The script will print its progress to the console and save all generated plots and the final model file in the same directory.

------------------------------------------------------------
4. ABOUT THE FINAL MODEL
------------------------------------------------------------
The final deployable model is saved as `final_crop_model.pkl`. This file contains the trained soft-voting ensemble classifier. It can be loaded using the `pickle` library in Python and used to make predictions on new, preprocessed data.

------------------------------------------------------------

CROP Recommendation Deployment

------------------------------------------------------------


Crop Advisor – Local Setup & Run
================================

This project has two parts:
1. server/      -> FastAPI backend that loads the trained model and exposes prediction APIs.
2. crop-chat/   -> React (Vite) frontend with a chat-style UI.


1) PREREQUISITES
----------------
- Python 3.10+
- Node.js 18+ and npm (or yarn/pnpm)
- (Optional) Git
- Works on Windows / macOS / Linux


2) FOLDER STRUCTURE
-------------------
/deployment/agri/
├─ server/
│  ├─ main.py
│  ├─ soft_voting_classifier.pkl      # trained model bundle (joblib/pickle)
│  ├─ commodity_prices.sqlite3        # SQLite DB with min/max prices (optional but recommended)
│  ├─ i18n/
│  │  └─ en.json                      # language strings
│  ├─ flow/
│  │  └─ default.json                 # slot/flow config
│  └─ requirements.txt                # Python deps
└─ crop-chat/
   ├─ src/
   │  ├─ App.jsx
   │  ├─ i18n.js
   │  ├─ index.css
   │  └─ main.jsx
   ├─ index.html
   ├─ package.json
   ├─ vite.config.js
   └─ .env.local


3) BACKEND – FASTAPI
--------------------
Create and activate a virtual environment:
-------------------------------------------
cd server
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

Install dependencies:
---------------------
Create a requirements.txt (if not present):
fastapi
uvicorn[standard]
joblib
pandas
numpy
scikit-learn
catboost

Then run:
pip install -r requirements.txt

Run the FastAPI server:
-----------------------
uvicorn main:app --reload --port 8000

API Endpoints:
--------------
Health:            http://127.0.0.1:8000/api/health
Predict:           http://127.0.0.1:8000/api/predict
District list:     http://127.0.0.1:8000/api/meta/districts


Example /predict Request:
-------------------------
curl -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" -d '{
  "District": "Kadapa",
  "Soil_Type_Standard": "Red",
  "pH": 6.5,
  "EC": 1.2,
  "OC": 1.1,
  "Avail-P": 20,
  "Exch-K": 150,
  "Avail-Ca": 2000,
  "Avail-Mg": 200,
  "Avail-S": 15,
  "Avail-Zn": 1.0,
  "Avail-B": 0.5,
  "Avail-Fe": 10,
  "Avail-Cu": 1.0,
  "Avail-Mn": 10
}'

Sample Response:
----------------
{
  "top_k": ["Rice","Maize","Groundnut"],
  "top_k_with_prices": [
    {"commodity":"Rice","min_price":1450.0,"max_price":2200.0},
    {"commodity":"Maize","min_price":1100.0,"max_price":1800.0},
    {"commodity":"Groundnut","min_price":4200.0,"max_price":5500.0}
  ]
}


4) FRONTEND – REACT (VITE)
--------------------------
Install dependencies:
---------------------
cd crop-chat
npm install

Set the API base URL:
---------------------
Create a file .env.local in crop-chat/ with:
VITE_API_BASE=http://127.0.0.1:8000/api

Run the development server:
---------------------------
npm run dev

Access the application in your browser:
---------------------------------------
http://127.0.0.1:5173/

You should see the "Crop Advisor" chatbot UI.


5) PRODUCTION BUILD (OPTIONAL)
------------------------------
npm run build
This generates static files under crop-chat/dist/


6) TROUBLESHOOTING
------------------
- Run uvicorn from inside the server folder.
- Ensure soft_voting_classifier.pkl is present.
- Allow CORS if browser blocks API calls.
- Update ports if 8000 or 5173 are in use.


7) DESIGN HIGHLIGHTS
--------------------
- Cross-platform: Works on mobile and desktop browsers.
- Scalable: FastAPI handles async high-volume requests.
- Modular: Model and UI can be updated independently.
- Low latency: Preloaded model responds within milliseconds.
- Extensible: Future integrations with weather APIs or IoT sensors.




