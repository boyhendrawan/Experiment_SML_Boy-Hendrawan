from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump
import os


def getData(path):
    data1 = pd.read_csv(f"{path}/test.csv")
    data2 = pd.read_csv(f"{path}/train.csv")
    return pd.concat([data1, data2], verify_integrity=True, ignore_index=True)


def preprocess_data(data, target_column, processor_save_path, clean_train_dataset_save_path, clean_test_dataset_save_path):
    # Pisahkan target dari data
    y = data[target_column]
    X = data.drop(target_column, axis=1)

    # Konversi target ke numerik jika kategorikal
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        # Save the label encoder for later use
        os.makedirs(os.path.dirname(processor_save_path), exist_ok=True)
        dump(le, processor_save_path.replace('.joblib', '_label_encoder.joblib'))

    # Pilih fitur numerik dan kategorikal dari X (tanpa target)
    numeric_features = X.select_dtypes(include='number').columns.tolist()
    categoric_features = X.select_dtypes(include='object').columns.tolist()

    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    # Pipeline untuk fitur kategorikal
    categoric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabungkan ke dalam ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categoric_transformer, categoric_features)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit dan transformasi data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Buat nama kolom hasil encoding
    column_names = preprocessor.get_feature_names_out()

    # Buat dataframe hasil
    train_df = pd.DataFrame(X_train.toarray() if hasattr(X_train, "toarray") else X_train, columns=column_names)
    train_df[target_column] = y_train

    test_df = pd.DataFrame(X_test.toarray() if hasattr(X_test, "toarray") else X_test, columns=column_names)
    test_df[target_column] = y_test

    print(test_df.isna().sum())
    print(train_df.isna().sum())

    # Simpan hasil dan pipeline
    os.makedirs(os.path.dirname(clean_train_dataset_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(processor_save_path), exist_ok=True)

    train_df.to_csv(clean_train_dataset_save_path, index=False)
    test_df.to_csv(clean_test_dataset_save_path, index=False)
    dump(preprocessor, processor_save_path)


# Paths
raw_dataset_path = '../attrition_raw'
clean_train_path = './clean_dataset/clean_train.csv'
clean_test_path = './clean_dataset/clean_test.csv'
preprocessing_pipeline_path = './pipeline/preprocessing_pipeline.joblib'

# Load data
dt = getData(raw_dataset_path)
dt = dt.drop(['Employee ID'], axis=1,errors="ignore")

# Run
preprocess_data(dt, 'Attrition', preprocessing_pipeline_path, clean_train_path, clean_test_path)
