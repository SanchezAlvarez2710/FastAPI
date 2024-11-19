from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

data = pd.read_csv("test.csv")
data = data.rename(columns={"Var_1": "Category"})

data['Work_Experience'] = data['Work_Experience'].fillna(data['Work_Experience'].median())
data['Family_Size'] = data['Family_Size'].fillna(data['Family_Size'].median())
categorical_columns = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Category', 'Spending_Score']
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(columns=["ID", "Category"])
y = data["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')
knn.fit(X_train, y_train)

app = FastAPI()

class PredictionInput(BaseModel):
    n_neighbors: int
    metric: str
    weights: str
    features: list[float]

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if len(input_data.features) != X_train.shape[1]:
        raise HTTPException(status_code=400, detail=f"Se esperan {X_train.shape[1]} características.")
    
    try:
        knn_custom = KNeighborsClassifier(
            n_neighbors=input_data.n_neighbors,
            metric=input_data.metric,
            weights=input_data.weights
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    knn_custom.fit(X_train, y_train)
    
    prediction = knn_custom.predict([input_data.features])
    predicted_category = label_encoders["Category"].inverse_transform(prediction)[0]
    
    return {"predicted_category": predicted_category}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
