# Assignment-16  
## Polynomial Regression for Advertising Sales Prediction (Streamlit App)

### Status
Attempted

---

## Objective

Build a **Polynomial Regression model** to predict product sales based on advertising budgets for:

- TV
- Radio
- Newspaper

The model should capture **non-linear relationships** between advertising spend and sales, specifically highlighting **diminishing returns at higher spending levels**.

---

## Dataset

Use the **Advertising Dataset**, which contains:

- TV advertising budget
- Radio advertising budget
- Newspaper advertising budget
- Sales (target variable)

---

## Tasks

### 1. Data Preparation
- Load the Advertising dataset
- Perform exploratory data analysis (EDA)
- Visualize relationships between:
  - TV vs Sales
  - Radio vs Sales
  - Newspaper vs Sales

---

### 2. Model Development
- Implement **Polynomial Regression**
- Transform input features using polynomial feature expansion
- Train the model on the dataset
- Evaluate performance using:
  - R² Score
  - Mean Squared Error (MSE)

---

### 3. Capturing Non-Linearity
- Demonstrate how sales growth slows at higher advertising budgets
- Highlight **diminishing returns** in predictions
- Compare linear vs polynomial regression performance

---

### 4. Streamlit Web Application

Create an interactive **Streamlit app** that allows users to:

#### Inputs:
- TV budget
- Radio budget
- Newspaper budget

#### Output:
- Predicted sales value (instant result)

#### Features:
- User-friendly UI sliders or input boxes
- Real-time prediction updates
- Optional visualization of prediction trends

---

## Key Learning Outcomes

This project helps users:

- Optimize advertising budgets for maximum ROI
- Identify overspending and avoid diminishing returns
- Allocate budgets effectively across multiple channels
- Make fast, data-driven marketing decisions
- Understand non-linear relationships in real-world sales data

---

## Tech Stack

- Python
- NumPy / Pandas
- Scikit-learn
- Matplotlib / Seaborn (optional visualization)
- Streamlit (for web app)

---

## Expected Output

A working Streamlit application that:
- Accepts advertising budgets as input
- Predicts expected sales using a trained polynomial regression model
- Demonstrates non-linear effects of advertising spend on sales
