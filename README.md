 <img width="571" alt="Screenshot 2025-06-27 at 3 51 17 PM" src="https://github.com/user-attachments/assets/ba232535-6f24-46e9-8796-3394bc95903a" />
# student marks predictor– FastAPI + ML + Docker

Welcome to **Student Marks Predictor**!  
This isn't just another Python app — it's a smart prediction engine 🚀 built with FastAPI and powered by multiple machine learning models. You give it a few student details… and 💥 out comes a mark prediction!

---

## 🧠 What’s This All About?

This project:
- Takes in **student info** like test scores, lunch type, and parental education 🎒
- Tries **multiple ML models** (Linear Regression, Random Forest, etc.) and **picks the best** 🧠
- Serves predictions using a fast and modern **FastAPI backend** ⚡
- Is fully **Dockerized** for easy setup anywhere 🐳

---

## 🗂️ Project Structure

student-marks-prediction/
├── app/ ← FastAPI app & ML logic
│ ├── main.py ← Launches the app
│ ├── components/ ← Ingestion, preprocessing, model training
│ ├── utils.py ← Reusable helpers
├── artifacts/ ← Trained model & preprocessor (auto-generated)
├── Dockerfile ← For containerizing everything
├── requirements.txt ← All dependencies
└── README.md ← You are here!


---

## 🚀 Run It in 1-2-3

1. **Clone this repo**:
```bash
git clone https://github.com/your-username/student-marks-prediction.git
cd student-marks-prediction
Build the Docker image:
docker build -t student-predictor .
Run the container:
docker run -p 8080:8080 student-predictor
Open the Swagger docs:
Visit
http://localhost:8080/docs
🎯 and test the API with your inputs!

🔍 How It Works

Student data is submitted to /predictdata
The app loads:
A preprocessor (for encoding, scaling, etc.)
The best ML model chosen after comparing several during training
Prediction is made and returned like:
{
  "predicted_marks": 83.2
}
🔬 Models Tried

During training, these models were compared:

Linear Regression
Decision Tree
Random Forest
Gradient Boosting
XGBoost
CatBoost
✅ The best one based on R² score is saved and used for predictions.

🧰 Tech Stack

Tool	Purpose
FastAPI	High-speed modern API backend ⚡
Scikit-learn	Model training & preprocessing 🔧
Pandas	Data wrangling 📊
Docker	Containerization 🐳
Matplotlib/Seaborn	Exploratory visuals 🧠
📦 For Manual Setup (without Docker)

pip install -r requirements.txt
uvicorn app.main:app --reload
🧪 Sample Input

{
  "gender": "female",
  "race_ethnicity": "group C",
  "parental_level_of_education": "associate's degree",
  "lunch": "free/reduced",
  "test_preparation_course": "none",
  "reading_score": 68,
  "writing_score": 70
}
