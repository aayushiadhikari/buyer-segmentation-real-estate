# 🏠 Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

An AI-powered real estate analytics project that uses machine learning-based customer segmentation to identify different buyer profiles and investment behaviors.

The project combines data preprocessing, exploratory data analysis, feature engineering, K-Means clustering, buyer profiling, and an interactive Streamlit dashboard to generate actionable real estate market intelligence.

---

## 📌 Project Overview

Real estate markets contain buyers with different characteristics, financial capabilities, investment purposes, geographic preferences, and purchasing behaviors.

Treating all buyers as a single group can lead to:

- Inefficient marketing strategies
- Generic property recommendations
- Poor customer targeting
- Missed investment opportunities

This project uses machine learning-based clustering to identify hidden patterns among buyers and convert them into meaningful buyer segments.

---

## 🎯 Objectives

The main objectives of this project are to:

- Analyze buyer demographics and characteristics
- Understand property preferences and purchasing behavior
- Identify investment patterns among buyers
- Analyze financial and financing-related attributes
- Study geographic differences in buyer behavior
- Create meaningful buyer segments using unsupervised learning
- Build an interactive dashboard for real estate market intelligence
- Support targeted marketing and buyer profiling

---

## 📂 Project Structure

```text
buyer-segmentation-real-estate/
│
├── app/
│   └── app.py
│
├── data/
│   ├── clients.csv
│   ├── properties.csv
│   ├── master_dataset.csv
│   └── final_clustered_dataset.csv
│
├── models/
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   └── 03_Model_Building.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md

🧹 Data Preprocessing

The preprocessing stage includes:

Dataset structure inspection
Missing value analysis and handling
Data type correction
Duplicate checking
Cleaning inconsistent data
Processing client and property information
Combining client and property datasets
Creating a buyer-level analytical dataset

The processed dataset is used for exploratory analysis and machine learning.

🔍 Exploratory Data Analysis

EDA is performed to understand buyer and property characteristics and identify important patterns.

The analysis covers:

Buyer characteristics
Property preferences
Purchase-related behavior
Financial attributes
Geographic patterns
Numerical feature distributions
Relationships between buyer and property attributes
Investment behavior

Visualizations are used to identify trends and relationships in the data.

⚙️ Feature Engineering

Buyer-level features are created from client and property information.

The engineered features represent:

Buyer characteristics
Property preferences
Purchase behavior
Financial capacity
Investment-related behavior
Property ownership/purchase patterns

These features form the input to the clustering model.

🤖 Machine Learning
K-Means Clustering

K-Means is used as the primary unsupervised machine learning algorithm for buyer segmentation.

The workflow consists of:

Selecting relevant buyer-level features
Preparing the feature matrix
Encoding categorical variables where required
Scaling numerical features
Training the K-Means clustering model
Assigning buyers to clusters
Profiling cluster characteristics
Mapping cluster results to interpretable buyer segments

The trained K-Means model and feature scaler are saved in the models/ directory.

👥 Buyer Segmentation

The clustering results are converted into interpretable buyer segments.

Cluster numbers themselves do not represent meaningful buyer types. Therefore, the characteristics of each cluster are analyzed before assigning meaningful business interpretations.

The segmentation considers factors such as:

Buyer demographics
Acquisition purpose
Financing behavior
Investment amount
Property characteristics
Geographic distribution
Purchase behavior

This makes the machine learning output easier to understand from a real estate business perspective.

📊 Key Analytical Features

The final analytical dataset contains buyer-level information that supports analysis of:

Age
Client type
Gender
Country
Region
Acquisition purpose
Loan behavior
Satisfaction
Total investment
Number of properties
Average property characteristics
Buyer segment

🌐 Streamlit Application

The interactive Streamlit application is located at:

app/app.py

The dashboard provides an interactive interface for exploring buyer segmentation and investment behavior.

Dashboard Features
Total buyer count
Total investment
Average buyer age
Average satisfaction score
Total properties
Buyer type distribution
Acquisition purpose analysis
Age distribution
Investment by buyer segment
Investment vs. property characteristics
Investment by country
Investment by region
Buyer segmentation insights

Users can explore the results through interactive visualizations and filters.

🛠️ Technology Stack
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Plotly
Jupyter Notebook
Streamlit
Joblib
Git
GitHub

🚀 How to Run the Project
1. Clone the repository
git clone https://github.com/aayushiadhikari/buyer-segmentation-real-estate.git
2. Navigate to the project directory
cd buyer-segmentation-real-estate
3. Create and activate a virtual environment
python -m venv venv
Windows:
venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
5. Run the Streamlit application
streamlit run app/app.py

The application will open in your browser.

📓 Notebooks
01_Data_Preprocessing.ipynb

Contains the data inspection, cleaning, preprocessing, and creation of the buyer-level analytical dataset.

02_EDA.ipynb

Contains exploratory analysis and visualizations of buyer, property, financial, and geographic characteristics.

03_Model_Building.ipynb

Contains feature preparation, scaling, K-Means clustering, cluster analysis, buyer segmentation, and model saving.

💾 Saved Machine Learning Models

The trained machine learning artifacts are stored in the models/ directory:

models/
├── kmeans_model.pkl
└── scaler.pkl

These artifacts allow the Streamlit application to use the trained clustering model without retraining it every time.

📈 Business Applications

The resulting buyer segmentation can support:

Targeted marketing campaigns
Personalized property recommendations
Buyer profiling
Customer prioritization
Investment opportunity identification
Geographic market analysis
Financing strategy analysis
Real estate business intelligence
Data-driven customer engagement

⚠️ Limitations
The segmentation is based on the available dataset and selected features.
K-Means clustering identifies statistical patterns and does not represent officially defined customer categories.
Cluster interpretations depend on the characteristics of the underlying data.
The model should be retrained when substantially different or new buyer data becomes available.
The results are intended for analytical and decision-support purposes.

🔮 Future Improvements

Potential future enhancements include:

Testing additional clustering algorithms
Improving cluster validation using multiple evaluation metrics
Adding automated model selection
Incorporating additional financial and property features
Developing personalized property recommendation systems
Adding real-time or continuously updated data
Deploying the application on a cloud platform
Adding advanced geographic visualizations


📄 Project Context

This project was developed as part of the Unified Mentor project for Parcl Co. Limited, focusing on machine learning-based buyer intelligence and real estate market analytics.

👩‍💻 Author

Aayushi Adhikari
