{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import numpy as np\n",
    "import os\n",
    "from sklearn.externals import joblib\n",
    "\n",
    "\n",
    "def init():\n",
    "    global model\n",
    "    model_path = os.path.join(os.getenv('AzureML'), 'labor-prediction.pkl')\n",
    "    model = joblib.load(model_path)\n",
    "\n",
    "def run(data):\n",
    "    try:\n",
    "        data = np.array(json.loads(data))\n",
    "        result = model.predict(data)\n",
    "        # You can return any data type, as long as it is JSON serializable.\n",
    "        return result.tolist()\n",
    "    except Exception as e:\n",
    "        error = str(e)\n",
    "        return error"
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
