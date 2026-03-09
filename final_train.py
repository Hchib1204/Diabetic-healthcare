{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3d062906-9444-46b5-a995-a0280f43bfe6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Successfully saved synchronized scaler.pkl and diabetes_model.pkl\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# 1. Load data\n",
    "df = pd.read_csv('diabetes.csv')\n",
    "\n",
    "# 2. Synchronized Feature Engineering\n",
    "def prepare_data(data):\n",
    "    d = data.copy()\n",
    "    # Handle zeros in clinical columns using medians\n",
    "    cols_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']\n",
    "    for col in cols_fix:\n",
    "        d[col] = d[col].replace(0, np.nan).fillna(d[col].median())\n",
    "    \n",
    "    # Create new features\n",
    "    d[\"Insulin_Glucose_Ratio\"] = d[\"Insulin\"] / d[\"Glucose\"].replace(0, np.nan).fillna(1)\n",
    "    d[\"BMI_Class\"] = pd.cut(d[\"BMI\"], bins=[0, 18.5, 25.0, 30.0, float(\"inf\")], labels=[0, 1, 2, 3]).astype(float)\n",
    "    d[\"Age_Glucose\"] = d[\"Age\"] * d[\"Glucose\"]\n",
    "    \n",
    "    # MANDATORY ORDER: This must match app.py exactly\n",
    "    cols = [\"Pregnancies\", \"Glucose\", \"BloodPressure\", \"SkinThickness\", \"Insulin\", \n",
    "            \"BMI\", \"DiabetesPedigreeFunction\", \"Age\", \"Insulin_Glucose_Ratio\", \n",
    "            \"BMI_Class\", \"Age_Glucose\"]\n",
    "    return d[cols]\n",
    "\n",
    "# 3. Process and Train\n",
    "X = prepare_data(df.drop('Outcome', axis=1))\n",
    "y = df['Outcome']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(X_train_scaled, y_train)\n",
    "\n",
    "# 4. Save Fresh Files\n",
    "joblib.dump(scaler, 'scaler.pkl')\n",
    "joblib.dump(model, 'diabetes_model.pkl')\n",
    "print(\"✅ Successfully saved synchronized scaler.pkl and diabetes_model.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07fc8d2b-751a-490d-aae6-4761a22b7ee9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
