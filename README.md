🏠 Housing Price Prediction using Regression


This project applies machine learning regression algorithms to predict housing prices based on features such as area, number of bedrooms, presence of amenities, and more. The model is trained and evaluated using both Linear Regression and Random Forest Regressor, with hyperparameter tuning via GridSearchCV.


📁 Project Structure


housing-price-prediction/
│
├── Housing.csv                # Dataset used for training/testing
├── housing\_model.ipynb        # Jupyter notebook or script
├── README.md                  # Project documentation



 📊 Dataset Description

The dataset contains the following features:

- `area`: Total area of the house (in sq. ft.)
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `stories`: Number of floors
- `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`: Yes/No features
- `furnishingstatus`: Categorical (`furnished`, `semi-furnished`, `unfurnished`)
- `prefarea`: Whether house is in a preferred area
- `price`: Target variable — the sale price of the house


 ✅ Project Steps

### 1. Data Loading & Exploration
- Load the CSV file using `pandas`
- Check for data types, missing values, and basic statistics

### 2. Data Preprocessing
- Convert categorical variables using `get_dummies` and `map()`
- Normalize numerical features using `StandardScaler`
- Drop original columns after transformation

### 3. Visualization
- Histograms for data distribution
- Heatmaps for feature correlation

### 4. Model Building
- Train a **Linear Regression** model as baseline
- Train a **Random Forest Regressor** to improve accuracy
- Use **GridSearchCV** to optimize the random forest

### 5. Model Evaluation
- Evaluate using R² score
- (Optional) Add MAE and RMSE for better performance insights

---

## 🧪 Example Results

- **Linear Regression R² Score**: ~`0.xx`
- **Random Forest R² Score**: ~`0.xx`
- **Best Random Forest (tuned) R² Score**: ~`0.xx`



## ⚙️ Requirements

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
````

---

## 🚀 How to Run

1. Clone the repository
2. Place `Housing.csv` in the root directory or correct path
3. Run the notebook:

```bash
jupyter notebook housing_model.ipynb
```

Or run the script:

```bash
python housing_model.py
```

---

## 📈 Future Improvements

* Add model saving with `joblib` or `pickle`
* Explore other regression models (e.g., XGBoost, GradientBoosting)
* Add error metrics like MAE, RMSE
* Deploy using Flask or Streamlit
* Use cross-validation with all models

---

## 📌 Author

**Jamiu Kareem**
kjamiu4@gmail.com

---

## 📝 License

This project is open-source and free to use for educational purposes.


