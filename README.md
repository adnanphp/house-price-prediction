#  House Price Prediction API

A production-ready machine learning API for predicting house prices using the California Housing dataset.



**API URL:** [https://house-price-prediction-lgwa.onrender.com]

##  Features

-  Machine Learning Model (Gradient Boosting with hyperparameter tuning)
-  REST API with FastAPI
-  Docker containerization
-  Deployed on Render.com
-  Interactive Swagger Documentation

## Tech Stack

- **Python 3.10** + FastAPI
- **scikit-learn** (Gradient Boosting, Random Forest, Linear Regression)
- **Docker** + Docker Hub
- **Render.com** for deployment

##  Model Performance

| Model | R² Score | RMSE |
|-------|----------|------|
| Linear Regression | 0.5758 | 0.7456 |
| Random Forest | 0.8053 | 0.5051 |
| Gradient Boosting (Tuned) | **0.8354** | **0.5422** |

##  Local Development

### Run with Docker

```bash
docker pull adnanphp/house-price-api
docker run -p 8000:8000 adnanphp/house-price-api
```

### Test the API

#### Health Check

```bash
curl https://house-price-prediction-lgwa.onrender.com/
```

#### Predict House Price

```bash
curl -X POST https://house-price-prediction-lgwa.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322.0,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

### Example Response

```json
{
  "predicted_price": 4.509417906267169,
  "features": {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322.0,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }
}
```
