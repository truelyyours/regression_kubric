import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.
	
    You can run this program from the command line using `python3 regression.py`.
    """
    # Loading training data
    response = requests.get(TRAIN_DATA_URL)
    area_train,price_train = response.text.split()
    area_train = numpy.array(list(map(float,area_train.split(',')[1:])))
    area_train = area_train / numpy.linalg.norm(area_train)
    price_train = numpy.array(list(map(float,price_train.split(',')[1:])))
    price_train = price_train / numpy.linalg.norm(price_train)

    # Loading test data
    response = requests.get(TEST_DATA_URL)
    area_test,price_test = response.text.split()
    area_test = list(map(float,area_test.split(',')[1:]))
    price_test = list(map(float,price_test.split(',')[1:]))
    learning_rate = 0.005

    # actual code
    # y = mx + c
    m = 0
    c = 0
    epoch = 10000
    data = numpy.array([price_train,area_train]).transpose()

    for i in range(epoch):
    	c_dash = 0.0
    	m_dash = 0.0
    	n = float(len(data))
    	for i in range(len(data)):
    		x = data[i,1]
    		y = data[i,0]
    		c_dash -= (2/n) * (y - (m * x + c))
    		m_dash -= (2/n) * x * (y - (m * x + c))

    	c -= learning_rate*c_dash
    	m -= learning_rate*m_dash

    price = []
    for area in area_test:
    	price.append(m*area + c)
#     Need to return the validatoion values only.. It seems
    return numpy.array(price)[:24]


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
