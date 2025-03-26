# SANSO ZOKA 

## Overview

The **Dissolved Oxygen Prediction System** by **SANSO ZOKA** integrates **Machine Learning** and **Electrocoagulation Treatment Systems** to analyze and optimize water quality. This application predicts **Dissolved Oxygen (DO) levels** in water using **Random Forest Regression**, providing a powerful tool for environmental monitoring and water treatment enhancements.

## Features

- **Advanced Machine Learning Model**: Utilizes Random Forest Regression for DO prediction.
- **Real-time Interactive UI**: Users can input water quality parameters and receive immediate predictions.
- **Persistent Data Storage**: Predictions are saved in a CSV database for long-term analysis.
- **Data Visualization (Upcoming)**: Interactive graphs and trend analysis for better insights.
- **Optimized Sidebar Navigation**: Custom-styled for an enhanced user experience.

## Tech Stack

- **Framework**: Streamlit (Python-based UI framework)
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Storage**: CSV-based persistent data storage
- **Styling**: Custom CSS for sidebar enhancements

## Installation

### Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### Clone the Repository

```bash
git clone https://github.com/your-username/dissolved-oxygen-prediction.git
cd dissolved-oxygen-prediction
```

### Create and Activate a Virtual Environment

It is recommended to run the application inside a virtual environment.

#### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run evaluation.py
```

## Project Structure

```
├── EDA_Model/
│   ├── Model_test_data/
│   │   ├── DOE.csv  # Dataset file
│   ├── model.py  # Machine learning model training script
│
├── static/
│   ├── styles.css  # Custom sidebar styles
│
├── app.py  # Main Streamlit application
├── requirements.txt  # Required dependencies
├── README.md  # Project documentation
```

## Usage

1. Navigate to the **Home** page to enter water quality parameters.
2. Adjust sliders for different input values.
3. Click **Predict DO Level** to generate a machine-learning prediction.
4. View stored predictions in the **Database** section.
5. (Upcoming) Analyze water quality trends in the **Stats** section.

## Roadmap

-

## Contributing

Contributions are welcome. Please **fork** the repository, create a branch, and submit a **pull request**.

## License

This project is licensed under the **MIT License**.

---

**Developed by:** **SANSO ZOKA** - Advancing Water Quality with AI & Electrocoagulation

