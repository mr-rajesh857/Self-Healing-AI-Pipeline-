from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import warnings
import os
import tempfile
from collections import Counter
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression, SelectKBest, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from boruta import BorutaPy
from sklearn.utils import check_X_y
import google.generativeai as genai

warnings.filterwarnings('ignore')

# Configure Gemini AI (replace with your API key)
genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")

app = Flask(__name__)
CORS(app)

# Global variables to store session data
session_data = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        file.save(temp_file.name)
        
        # Read CSV
        df = pd.read_csv(temp_file.name)
        
        # Store in session
        session_data['df'] = df
        session_data['temp_file'] = temp_file.name
        
        return jsonify({
            'success': True,
            'columns': list(df.columns),
            'shape': df.shape,
            'preview': df.head().to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        data = request.json
        target_column = data['target_column']
        task_type = data['task_type']
        
        df = session_data['df']
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Drop missing values
        X = X.dropna(axis=0)
        y = y.loc[X.index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task_type == 'classification' else None
        )
        
        # Store split data
        session_data['X_train'] = X_train
        session_data['X_test'] = X_test
        session_data['y_train'] = y_train
        session_data['y_test'] = y_test
        session_data['target_column'] = target_column
        session_data['task_type'] = task_type
        
        # Preprocessing
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ])
        
        preprocessor.fit(X_train)
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        
        X_train_arr = X_train_trans.toarray() if hasattr(X_train_trans, "toarray") else X_train_trans
        X_test_arr = X_test_trans.toarray() if hasattr(X_test_trans, "toarray") else X_test_trans
        
        session_data['preprocessor'] = preprocessor
        session_data['X_train_processed'] = X_train_arr
        session_data['X_test_processed'] = X_test_arr
        
        # Drift detection
        drift_results = calculate_drift(X_train, X_test)
        
        # Train base model
        if task_type == 'classification':
            base_model = RandomForestClassifier(random_state=42)
        else:
            base_model = RandomForestRegressor(random_state=42)
        
        base_model.fit(X_train_arr, y_train)
        base_performance = evaluate_model(task_type, base_model, X_test_arr, y_test)
        
        session_data['base_model'] = base_model
        
        return jsonify({
            'success': True,
            'drift_results': drift_results,
            'base_model_performance': base_performance
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/train', methods=['POST'])
def train_models():
    try:
        data = request.json
        feature_method = data['feature_method']
        
        X_train = session_data['X_train']
        X_test = session_data['X_test']
        y_train = session_data['y_train']
        y_test = session_data['y_test']
        task_type = session_data['task_type']
        
        # Feature selection
        X_train_encoded = encode_categoricals(X_train)
        X_selected = apply_feature_selection(feature_method, X_train_encoded, y_train, task_type)
        selected_features = list(X_selected.columns)
        
        # Prepare selected features data
        X_train_fs = X_train[selected_features]
        X_test_fs = X_test[selected_features]
        
        # Preprocessing for selected features
        cat_cols = X_train_fs.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X_train_fs.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        preprocessor_fs = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ])
        
        preprocessor_fs.fit(X_train_fs)
        X_train_fs_trans = preprocessor_fs.transform(X_train_fs)
        X_test_fs_trans = preprocessor_fs.transform(X_test_fs)
        
        X_train_fs_arr = X_train_fs_trans.toarray() if hasattr(X_train_fs_trans, "toarray") else X_train_fs_trans
        X_test_fs_arr = X_test_fs_trans.toarray() if hasattr(X_test_fs_trans, "toarray") else X_test_fs_trans
        
        # Apply SMOTE if needed
        X_train_model, y_train_model = apply_smote_if_needed(X_train_fs_arr, y_train, task_type)
        
        # Train multiple models
        models = get_models(task_type)
        model_performances = {}
        best_model = None
        best_score = -np.inf if task_type == 'regression' else 0
        
        for name, model in models.items():
            model.fit(X_train_model, y_train_model)
            performance = evaluate_model(task_type, model, X_test_fs_arr, y_test)
            model_performances[name] = performance
            
            # Determine best model
            score = get_model_score(performance, task_type)
            if (task_type == 'regression' and score > best_score) or (task_type == 'classification' and score > best_score):
                best_score = score
                best_model = {
                    'name': name,
                    'model': model,
                    'score': score,
                    'feature_method': feature_method,
                    'features_count': len(selected_features),
                    'selected_features': selected_features
                }
        
        session_data['best_model'] = best_model
        
        return jsonify({
            'success': True,
            'selected_features': selected_features,
            'model_performances': model_performances,
            'best_model': {
                'name': best_model['name'],
                'score': best_model['score'],
                'feature_method': best_model['feature_method'],
                'features_count': best_model['features_count']
            } if best_model else None
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate-code', methods=['POST'])
def generate_code():
    try:
        data = request.json
        model_name = data['model']
        feature_method = data['feature_method']
        task_type = data['task_type']
        target_column = data['target_column']
        
        best_model = session_data.get('best_model')
        if not best_model:
            return jsonify({'success': False, 'error': 'No best model found'})
        
        # Generate code using Gemini AI
        prompt = f"""
        Generate a complete HTML, CSS, and JavaScript web application for a machine learning model recommendation system.
        
        Model Details:
        - Best Model: {model_name}
        - Feature Selection Method: {feature_method}
        - Task Type: {task_type}
        - Target Column: {target_column}
        - Selected Features: {len(best_model['selected_features'])} features
        
        Requirements:
        1. Create a beautiful, modern web interface
        2. Include a form for input features
        3. Add prediction functionality
        4. Use modern CSS with gradients and animations
        5. Make it responsive and professional
        6. Include model information and performance metrics
        7. Add data visualization if possible
        
        Generate complete, production-ready code with proper structure.
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        generated_code = response.text
        
        return jsonify({
            'success': True,
            'code': generated_code
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Helper functions
def calculate_drift(X_train, X_test):
    drift_results = []
    X_train_encoded = encode_categoricals(X_train)
    X_test_encoded = encode_categoricals(X_test)
    
    for col in X_train_encoded.columns:
        try:
            psi = calculate_psi(X_train_encoded[col], X_test_encoded[col])
            if psi > 0.2:
                drift_results.append({'column': col, 'psi': psi})
        except:
            continue
    
    return drift_results

def calculate_psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)
    if np.std(expected) == 0 or np.std(actual) == 0:
        return 0
    
    breakpoints = np.linspace(0, 1, buckets + 1)
    scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
    expected_scaled = scale(expected)
    actual_scaled = scale(actual)
    
    expected_bins = np.histogram(expected_scaled, bins=breakpoints)[0] / len(expected)
    actual_bins = np.histogram(actual_scaled, bins=breakpoints)[0] / len(actual)
    
    psi_values = []
    for e, a in zip(expected_bins, actual_bins):
        if e == 0: e = 1e-4
        if a == 0: a = 1e-4
        psi_values.append((e - a) * np.log(e / a))
    
    return np.sum(psi_values)

def encode_categoricals(df):
    df_encoded = df.copy()
    encoder = OrdinalEncoder()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        df_encoded[cat_cols] = encoder.fit_transform(df_encoded[cat_cols].astype(str))
    return df_encoded

def apply_feature_selection(method, X, y, task_type):
    if method == 'f_test':
        score_func = f_regression if task_type == 'regression' else f_classif
        selector = SelectKBest(score_func=score_func, k='all')
        selector.fit(X, y)
        pvalues = selector.pvalues_
        selected_cols = X.columns[np.where(pvalues < 0.05)[0]].tolist()
    
    elif method == 'mutual_info':
        score_func = mutual_info_regression if task_type == 'regression' else mutual_info_classif
        selector = SelectKBest(score_func=score_func, k='all')
        selector.fit(X, y)
        scores = selector.scores_
        selected_cols = X.columns[np.where(scores > 0.01)[0]].tolist()
    
    elif method == 'boruta':
        model = RandomForestRegressor(n_jobs=-1, random_state=42) if task_type == 'regression' else RandomForestClassifier(n_jobs=-1, random_state=42)
        X_array, y_array = check_X_y(X, y)
        X_array = pd.DataFrame(X_array, columns=X.columns)
        boruta_selector = BorutaPy(model, n_estimators='auto', verbose=0, random_state=42)
        boruta_selector.fit(X_array.values, y_array)
        selected_cols = X.columns[boruta_selector.support_].tolist()
    
    elif method == 'rfe':
        estimator = LinearRegression() if task_type == 'regression' else LogisticRegression(solver='liblinear')
        selector = RFE(estimator, n_features_to_select=int(X.shape[1] * 0.5))
        selector.fit(X, y)
        selected_cols = X.columns[selector.support_].tolist()
    
    return X[selected_cols]

def apply_smote_if_needed(X_train, y_train, task_type):
    if task_type != 'classification':
        return X_train, y_train
    
    class_counts = Counter(y_train)
    total = sum(class_counts.values())
    imbalance = any((count / total) < 0.2 for count in class_counts.values())
    
    if imbalance:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res
    
    return X_train, y_train

def get_models(task_type):
    if task_type == 'classification':
        return {
            'RandomForest': RandomForestClassifier(random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier()
        }
    else:
        return {
            'RandomForest': RandomForestRegressor(random_state=42),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'SVM': SVR(),
            'KNN': KNeighborsRegressor()
        }

def evaluate_model(task_type, model, X_test, y_test):
    y_pred = model.predict(X_test)
    if task_type == 'regression':
        return {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
    else:
        return {
            'Accuracy': accuracy_score(y_test, y_pred)
        }

def get_model_score(performance, task_type):
    if task_type == 'regression':
        return performance['R2']
    else:
        return performance['Accuracy']

if __name__ == '__main__':
    app.run(debug=True, port=5000)