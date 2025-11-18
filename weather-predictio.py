import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# 1. DATASET WITH 4 CITIES

data = {
    "place": [
        "Delhi","Mumbai","Chennai","Delhi","Mumbai","Chennai","Delhi","Mumbai",
        "Bangalore","Bangalore","Bangalore"
    ],
    "temperature": [22,30,34,21,29,33,23,31,28,27,29],
    "humidity": [90,75,65,92,72,60,88,70,80,85,78],
    "weather": [
        "Rain","Sunny","Sunny","Rain","Sunny","Sunny","Rain","Sunny",
        "Rain","Rain","Sunny"
    ]
}

df = pd.DataFrame(data)

# encode place
df["place_code"] = df["place"].astype("category").cat.codes

X = df[["place_code", "temperature", "humidity"]]
y = df["weather"]

# 2. TRAIN MODEL

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)


# 3. USER INPUT

place_input = input("Enter Place (Delhi/Mumbai/Chennai/Bangalore): ")

if place_input not in df["place"].unique():
    print("Place not found in dataset!")
    exit()

place_code = df[df["place"] == place_input]["place_code"].iloc[0]

temp = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))

# Convert to DF to avoid warnings
input_df = pd.DataFrame([{
    "place_code": place_code,
    "temperature": temp,
    "humidity": humidity
}])


# 4. ML PREDICTION

pred = model.predict(input_df)[0]
probs = model.predict_proba(input_df)[0]

rain_prob = probs[list(model.classes_).index("Rain")] * 100


# 5. HUMIDITY LEVEL

if humidity > 85:
    humidity_level = "High Humidity"
elif humidity > 60:
    humidity_level = "Moderate Humidity"
else:
    humidity_level = "Low Humidity"


# 6. RAIN TIME PREDICTION (ESTIMATION)

if rain_prob > 80:
    rain_time = "Evening (5 PM - 9 PM)"
elif rain_prob > 60:
    rain_time = "Afternoon (12 PM - 5 PM)"
elif rain_prob > 40:
    rain_time = "Morning (6 AM - 12 PM)"
else:
    rain_time = "No rain expected today"


# 7. OUTPUT

print("\n------- WEATHER REPORT -------")
print("Place:", place_input)
print("Weather Type:", pred)
print("Rain Chance:", f"{rain_prob:.2f}%")
print("Humidity Level:", humidity_level)
print("Estimated Rain Time:", rain_time)

print("\nBreakdown Probabilities:")
for label, p in zip(model.classes_, probs):
    print(f"{label}: {p*100:.2f}%")


# 8. CONFUSION MATRIX

cm = confusion_matrix(y_test, model.predict(X_test))

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()