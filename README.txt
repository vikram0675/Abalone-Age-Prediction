Abalone Age Prediction - Ready-to-run Project
=================================================

Contents:
- training/Code.ipynb          : Jupyter notebook to train models and save the best model.
- training/train_and_save.py   : Python script that trains Decision Tree and Random Forest, compares and saves the best model as abalone.pkl.
- flask_app/app.py             : Flask web application (runs on port 8000).
- flask_app/templates/         : home.html, predict.html, output.html (Bootstrap-based professional UI).
- flask_app/static/            : CSS and JS assets.
- dataset/                     : if you have abalone.csv, place it here. If not, the training script will try to download the UCI dataset automatically.
- abalone.pkl                  : (not included) model produced after running training scripts.

How to run:
1. Open the project in VS Code.
2. (Optional) Create a virtual environment and install requirements:
   pip install -r requirements.txt
3. To train models & save best model:
   python training/train_and_save.py
   This will create abalone.pkl at project root.
4. To run the Flask app:
   cd flask_app
   python app.py
   Open http://127.0.0.1:8000 in your browser.

Notes:
- The model predicts 'Rings'. Final age = Rings + 1.5 as per project guideline.
- The training script will try to download the UCI Abalone dataset if dataset/abalone.csv is missing.
