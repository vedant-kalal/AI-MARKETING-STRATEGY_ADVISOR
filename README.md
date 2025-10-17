🚀 AI Powered Bank Marketing Strategy Advisor
======================== 
> " Revolutionizing marketing strategies with AI-powered predictions and advice"

📖 Description
-------------
The Bank Marketing Strategy Advisor is a cutting-edge Python project that leverages machine learning and natural language processing to predict customer subscription probabilities and provide personalized marketing advice. This innovative tool is designed to help banks and financial institutions optimize their marketing efforts, improve customer engagement, and increase subscription rates.

The project utilizes a combination of machine learning algorithms, including ExtraTreesClassifier and RandomizedSearchCV, to predict customer subscription probabilities based on demographic and behavioral data. The predicted probabilities are then used to generate personalized marketing advice using the Gemini AI model. The project also features a user-friendly web interface built with Streamlit, allowing users to input customer data and receive predicted probabilities and marketing advice.

The Bank Marketing Strategy Advisor has numerous applications in the banking and financial services industry, including customer acquisition, retention, and upselling. By providing personalized marketing advice, the project can help banks tailor their marketing efforts to individual customers, improving the overall customer experience and driving business growth.

The project consists of several key components, including data preprocessing, model training, and prediction. The data preprocessing step involves cleaning and transforming the customer data, including handling missing values and encoding categorical variables. The model training step involves training the machine learning models using the preprocessed data, and the prediction step involves using the trained models to predict customer subscription probabilities.

The project also features a robust testing framework, including unit tests and integration tests, to ensure that the code is reliable and functions as expected. The testing framework includes tests for the machine learning models, the data preprocessing step, and the web interface, providing comprehensive coverage of the project's functionality.

In addition to its technical features, the Bank Marketing Strategy Advisor also provides a number of benefits to users, including improved marketing effectiveness, increased customer engagement, and enhanced business insights. By providing personalized marketing advice, the project can help banks and financial institutions optimize their marketing efforts, improve customer satisfaction, and drive business growth.

The project's web interface is user-friendly and easy to navigate, allowing users to input customer data and receive predicted probabilities and marketing advice. The interface also provides a number of visualizations and charts, including bar charts and histograms, to help users understand the predicted probabilities and marketing advice.

✨ Features
---------
The following are some of the key features of the Bank Marketing Strategy Advisor:
1. **Machine Learning-powered Predictions**: The project uses machine learning algorithms to predict customer subscription probabilities based on demographic and behavioral data.
2. **Personalized Marketing Advice**: The project generates personalized marketing advice using the Gemini AI model, based on the predicted subscription probabilities.
3. **User-friendly Web Interface**: The project features a user-friendly web interface built with Streamlit, allowing users to input customer data and receive predicted probabilities and marketing advice.
4. **Data Preprocessing**: The project includes a data preprocessing step, which involves cleaning and transforming the customer data, including handling missing values and encoding categorical variables.
5. **Model Training**: The project includes a model training step, which involves training the machine learning models using the preprocessed data.
6. **Prediction**: The project includes a prediction step, which involves using the trained models to predict customer subscription probabilities.
7. **Testing Framework**: The project includes a robust testing framework, including unit tests and integration tests, to ensure that the code is reliable and functions as expected.
8. **Visualizations and Charts**: The project's web interface provides a number of visualizations and charts, including bar charts and histograms, to help users understand the predicted probabilities and marketing advice.

🧰 Tech Stack Table
-------------------
| Component | Technology |
| --- | --- |
| Frontend | Streamlit |
| Backend | Python |
| Machine Learning | scikit-learn, joblib |
| Natural Language Processing | Gemini AI |
| Data Preprocessing | pandas, numpy |
| Testing Framework | unittest, pytest |



## 🗂️ Project Structure
-------------------

```text
├── .gitignore
├── README.md
├── notebooks
│   └── 01_EDA_and_Preprocessing.ipynb
├── requirements.txt
├── setup.cfg
└── src
    ├── Gemeni_Web_App
    │   ├── __init__.py
    │   ├── agent_advisor.py
    │   ├── app.py
    │   └── web_prediction.py
    ├── Model_selection&Tuning
    │   ├── __init__.py
    │   ├── model_selection.py
    │   ├── pipeline.py
    │   ├── prediction.py
    │   └── tuning.py
    └── __init__.py
```
The project is structured into the following folders:
* **src**: This folder contains the source code for the project, including the machine learning models, data preprocessing, and web interface.
* **models**: This folder contains the trained machine learning models.
* **data**: This folder contains the customer data used for training and testing the models.
* **tests**: This folder contains the testing framework, including unit tests and integration tests.
* **app**: This folder contains the web interface built with Streamlit.

Each folder has a brief description:
* **src/**: This is the main source code folder, containing all the Python files for the project.
* **models/**: This folder stores the trained machine learning models, which are used for making predictions.
* **data/**: This folder contains the customer data used for training and testing the models.
* **tests/**: This folder contains all the test files, including unit tests and integration tests.
* **app/**: This folder contains the Streamlit app, which provides a user-friendly web interface for the project.

⚙️ How to Run
-------------
To run the project, follow these steps:
1. **Setup**: Install the required dependencies, including Python, Streamlit, scikit-learn, and Gemini AI.
2. **Environment**: Create a new virtual environment using conda or virtualenv, and activate it.
3. **Build**: Build the project by running the `python setup.py build` command.
4. **Deploy**: Deploy the project by running the `streamlit run app.py` command.
5. **Run**: Run the project by navigating to the web interface in a web browser.

🧪 Testing Instructions
-------------------
To test the project, follow these steps:
1. **Unit Tests**: Run the unit tests by executing the `python -m unittest tests/test_models.py` command.
2. **Integration Tests**: Run the integration tests by executing the `python -m pytest tests/test_app.py` command.
3. **Test Data**: Use the test data provided in the **data** folder to test the project.

📦 API Reference
-------------
The project provides a RESTful API for accessing the predicted probabilities and marketing advice. The API endpoints are as follows:
* **/predict**: Returns the predicted subscription probability for a given customer.
* **/advice**: Returns the personalized marketing advice for a given customer.

👤 Author
--------
The Bank Marketing Strategy Advisor was developed by [Vedant Kalal].

📝 License
--------
The Bank Marketing Strategy Advisor is licensed under the MIT License. See the LICENSE file for more information.
