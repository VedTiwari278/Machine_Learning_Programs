{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP1edqe7lN8vvVCDOeH6sTk",
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
        "<a href=\"https://colab.research.google.com/github/VedTiwari278/Machine_Learning_Programs/blob/main/KNN.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
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
        "from sklearn.datasets import load_iris\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "X=load_iris().data\n",
        "Y=load_iris().target\n",
        "\n",
        "X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=92,random_state=40)\n",
        "\n",
        "knn_model=KNeighborsClassifier(n_neighbors=4)\n",
        "\n",
        "knn_model.fit(X_train,Y_train)\n",
        "\n",
        "y_pred=knn_model.predict(X_test)\n",
        "\n",
        "print(\"The accuracy of Knn : \",accuracy_score(Y_test,y_pred))\n"
      ]
    }
  ]
}