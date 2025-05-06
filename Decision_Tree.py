{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMqHo6RHJ7MwqQgj8Lk2D/T",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/VedTiwari278/Machine_Learning_Programs/blob/main/Decision_Tree.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fX-lnFbNq6xw"
      },
      "outputs": [],
      "source": [
        "#All import statement\n",
        "\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.datasets import load_iris\n",
        "\n",
        "#load statement\n",
        "\n",
        "iris=load_iris()\n",
        "X=iris.data\n",
        "y=iris.target\n",
        "\n",
        "# Split\n",
        "\n",
        "X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.8,random_state=4)\n",
        "\n",
        "#Initialize\n",
        "\n",
        "dt_model=DecisionTreeClassifier()\n",
        "dt_model.fit(X_train,Y_train)\n",
        "\n",
        "#prediction\n",
        "\n",
        "y_pred=dt_model.predict(X_test)\n",
        "\n",
        "print(\"The Accuracy : \",accuracy_score(Y_test,y_pred))"
      ]
    }
  ]
}