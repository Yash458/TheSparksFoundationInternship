{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# The Sparks Foundation "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Task 1 - Prediction using Supervised ML"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Predicting the percentage of an student based on number of study hours"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### step 1: Importing libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd \n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data imported successfully\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url = \"http://bit.ly/w-data\"\n",
    "df = pd.read_csv(url)\n",
    "print(\"Data imported successfully\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25 entries, 0 to 24\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Hours   25 non-null     float64\n",
      " 1   Scores  25 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 528.0 bytes\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualizing the Data Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot(x='Hours',y='Scores', style = 'o')\n",
    "plt.title('Hours Vs Percentage')\n",
    "plt.xlabel('Hours Studied')\n",
    "plt.ylabel('Percentage Score')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Preparing the Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.iloc[:,:-1].values\n",
    "y = df.iloc[:,1].values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Splitting data into training and testing sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Training the Data Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Completed\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "l_reg = LinearRegression()\n",
    "l_reg.fit(X_train,y_train)\n",
    "print(\"Training Completed\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept =  2.018160041434683\n",
      "Coefficient =  [9.91065648]\n"
     ]
    }
   ],
   "source": [
    "print(\"Intercept = \",l_reg.intercept_)\n",
    "print(\"Coefficient = \",l_reg.coef_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### plotting the Regression Line"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "line = l_reg.coef_*X+l_reg.intercept_\n",
    "\n",
    "#plotting for the test data\n",
    "plt.scatter(X,y)\n",
    "plt.plot(X,line)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Making Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Hours = \n",
      " [[1.5]\n",
      " [3.2]\n",
      " [7.4]\n",
      " [2.5]\n",
      " [5.9]]\n",
      "Predicted Scores = \n",
      " [16.88414476 33.73226078 75.357018   26.79480124 60.49103328]\n"
     ]
    }
   ],
   "source": [
    "print(\"Hours = \\n\",X_test)\n",
    "y_pred = l_reg.predict(X_test)\n",
    "print(\"Predicted Scores = \\n\",y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Comparing Actual and Predicted Scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>16.884145</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>27</td>\n",
       "      <td>33.732261</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>69</td>\n",
       "      <td>75.357018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>30</td>\n",
       "      <td>26.794801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>62</td>\n",
       "      <td>60.491033</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Actual  Predicted\n",
       "0      20  16.884145\n",
       "1      27  33.732261\n",
       "2      69  75.357018\n",
       "3      30  26.794801\n",
       "4      62  60.491033"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_comp = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})\n",
    "df_comp"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### As per given task, we have to predict score of student if he/she studied for 9.25 hrs/day"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of hours = 9.25\n",
      "Predicted Score = 93.69173248737538\n"
     ]
    }
   ],
   "source": [
    "hours = 9.25\n",
    "pred = l_reg.predict([[hours]])\n",
    "print(\"Number of hours = {}\".format(hours))\n",
    "print(\"Predicted Score = {}\".format(pred[0]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Evaluating using Mean Absolute Error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error:  21.5987693072174\n"
     ]
    }
   ],
   "source": [
    "print(\"Mean Squared Error: \", metrics.mean_squared_error(y_test,y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error:  4.183859899002975\n"
     ]
    }
   ],
   "source": [
    "print(\"Mean Absolute Error: \", metrics.mean_absolute_error(y_test,y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "r2_score:  0.9454906892105356\n"
     ]
    }
   ],
   "source": [
    "print(\"r2_score: \", r2_score(y_test,y_pred))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
