import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset
data = pd.read_csv(
    ".\MyAnalysis\TensorHulk\Classification\Cardiovascular disease prediction\Cardio_dataset.csv",
    delimiter=";",
)


# custom print method
def cprint(*msg: str):
    """Custom print method to decor output console"""
    print("-" * 120, "".join([str(m) for m in msg]), "-" * 120, sep="\n")


cprint("Total no. of rows = ", len(data), "Total no. of features = ", len(data.columns))
cprint("Column Headers  : ", data.columns)
cprint("Dataset\n", data.head())

# printing the unique values in each column
for column in data:
    series = data[column]
    count = len(list(series.value_counts()))
    print("No of unique values in column :", series.name, "\t=", count)


cprint(
    "OBSERVATION => gender, smoke, alco, active are categorical variables (cardio target)"
)

cprint("ACTION =>[id] column droped, since it is a classification problem")
data.drop(["id"], axis=1, inplace=True)

cprint("ACTION => [age] is mentioned in no of days, hence divided by 365")
data["age"] = (data["age"] / 365).round().astype(int)

data.drop_duplicates(inplace=True)
cprint("Total number of rows after droping duplicates = ", len(data))

# One-Hot Encoding
data = pd.get_dummies(data, columns=["gender", "cholesterol", "gluc"])
print(data.head())

# Split into features and labels
X = data.drop(["cardio"], axis=1)
y = data["cardio"]

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a logistic regression model
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cprint("ACCURACY:", accuracy)

# Save the model
with open("cardio_model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Load the model and make predictions on new data
with open("cardio_model.pkl", "rb") as f:
    model = pickle.load(f)

new_data = pd.DataFrame(
    {
        "age": [50],
        "height": [165],
        "weight": [75],
        "ap_hi": [120],
        "ap_lo": [80],
        "smoke": [1],
        "alco": [0],
        "active": [1],
        "gender_1": [0],
        "gender_2": [1],
        "cholesterol_1": [0],
        "cholesterol_2": [1],
        "cholesterol_3": [0],
        "gluc_1": [1],
        "gluc_2": [0],
        "gluc_3": [0],
    }
)

x = scaler.transform(new_data)
new_data = scaler.transform(new_data)

prediction = model.predict(new_data)

if prediction == 0:
    cprint("PREDICTION => ABSCENCE OF CARDIO VASCULAR DISEASE")
else:
    cprint("PREDICTION => PRESENCE OF CARDIO VASCULAR DISEASE")