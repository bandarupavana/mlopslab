import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("B1_toyato_feature.csv")

# Convert 'Price' into a classification target (example: high vs low price)
median_price = df["Price"].median()
df["Price_Class"] = df["Price"].apply(lambda x: 1 if x >= median_price else 0)

# Prepare data
X = df.drop(columns=["Price", "Price_Class"])
y = df["Price_Class"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# TPOT Classifier
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    random_state=42,
    n_jobs=-1,
    cv=5
)

# Train model
tpot.fit(X_train, y_train)

# Evaluate
accuracy = tpot.score(X_test, y_test)
print(f"Test Set Accuracy: {accuracy:.2f}")

# Export best model
tpot.export("best_classifier_pipeline.py")


#docker build -t automl-tpot .
#docker run --rm -v %cd%:/app automl-tpot python automl_script.py
#docker run --rm -v ${PWD}:/app automl-tpot python automl_script.py