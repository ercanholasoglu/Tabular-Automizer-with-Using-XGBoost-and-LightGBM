
import numpy as np
import pandas as pd
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler
from sklearn import pipeline
from category_encoders import TargetEncoder, OneHotEncoder, HashingEncoder, BackwardDifferenceEncoder, BinaryEncoder
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

import optuna
import xgboost as xgb
import lightgbm as lgbm
import logging
import warnings
import shap
from optuna.integration import XGBoostPruningCallback

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import RFECV as SklearnRFECV

from lime.lime_tabular import LimeTabularExplainer
import pickle
import shap
import lime
import lime.lime_tabular
from xgboost import XGBClassifier
import joblib
import traceback
from sklearn.utils.validation import check_is_fitted




cols_to_drop = [col for col in df.columns if col.lower() == 'id']

if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"Şu sütun elendi: {cols_to_drop}")
else:
    print("Hiç 'id' sütunu bulunamadı (büyük/küçük harf farketmeksizin).")

"""Global Variables:"""

##Depended Variable :

dependent_variable = 'Churn' ##Mutlaka değiştirilmeli

random_state = 42

treshold_for_eleminating_null_columns = 75

max_iter_for_iterative_imputing = 10

threshold_for_correlation_between_numerical_columns = 0

treshold_for_vif_analysis =1000

chi_squared_test_alpha = 0.5

correlation_between_categorical_columns_treshold = 0.8

test_size=0.2 ##for seperating test and train

number_of_required_feature = 1000 ##for rfecv

learning_rate_for_rfecv= 0.1

n_splits_for_rfe = 10 ##cv for rfecv

min_feature_selected_for_rfe = 20


initial_learning_rate_for_xgb_before_optuna = 0.01
initial_learning_rate_for_lgbm_before_optuna = 0.01

max_lr_for_xgb_optuna =0.1
min_lr_for_xgb_optuna =0.001

max_num_leaves_for_xgb_optuna = 256
min_num_leaves_for_xgb_optuna = 2

max_max_depth_for_xgb_optuna =8
min_max_depth_for_xgb_optuna = 1

min_subsample_for_xgb_optuna = 0.5
max_subsample_for_xgb_optuna = 1.0

max_colsample_bytree_for_xgb_optuna =1
min_colsample_bytree_for_xgb_optuna =0.1

max_n_estimators_for_xgb_optuna =10000
min_n_estimators_for_xgb_optuna =100

max_min_child_weight_for_xgb_optuna = 100
min_min_child_weight_for_xgb_optuna = 1

max_class_weight_for_xgb_optuna = 30
min_class_weight_for_xgb_optuna = 1

max_lr_for_lgbm_optuna =0.1
min_lr_for_lgbm_optuna =0.001

max_num_leaves_for_lgbm_optuna = 256
min_num_leaves_for_lgbm_optuna = 2

max_max_depth_for_lgbm_optuna =8
min_max_depth_for_lgbm_optuna = 1

min_subsample_for_lgbm_optuna = 0.5
max_subsample_for_lgbm_optuna = 1.0

max_colsample_bytree_for_lgbm_optuna =1
min_colsample_bytree_for_lgbm_optuna =.01

max_n_estimators_for_lgbm_optuna =10000
min_n_estimators_for_lgbm_optuna =100

max_class_weight_for_lgbm_optuna = 0
min_class_weight_for_lgbm_optuna = 0

min_child_samples_for_lgbm_optuna = 5
max_child_samples_for_lgbm_optuna = 100

class_weights_for_lgbm = {0: 1, 1: 5}

n_trials_for_optuna_optimization = 20

sample_number_for_shap_graph = 1000
num_features_for_lime= 30
sample_number_for_lime_graph = 1000


sample_size_for_pickle= 100

def checking_null_number_and_ratio_of_columns(df):
    null_count = df.isna().sum()
    null_ratio = (null_count / len(df)) * 100
    result_df = pd.DataFrame({
        'Kolon': null_count.index,
        'Null Sayısı': null_count.values,
        'Null Oranı (%)': null_ratio.round(2)
    })

    if result_df.empty:
        print("Veri setinde sütun bulunamadı.")
    else: ('Eksik Veri Analizi NULL_Result_DF e kaydedildi')

    return result_df

NULL_Result_DF = checking_null_number_and_ratio_of_columns(df)

class Preprocess_1:
    def __init__(self, df):
        self.df = df
        print('Preprocessing Öncesi Data Framedeki Satır ve Sütun Sayısı =', df.shape)

    def eliminating_null_columns(self, dependent_variable, treshold_for_eleminating_null_columns):
        df = self.df.copy()

        # Bağımlı değişkeni silinecek sütunlar listesinden çıkar
        initial_null_columns = (df.isnull().sum() / len(df))
        columns_to_drop = initial_null_columns[initial_null_columns > treshold_for_eleminating_null_columns].index.tolist()

        if dependent_variable in columns_to_drop:
            columns_to_drop.remove(dependent_variable)

        df.drop(columns=columns_to_drop, inplace=True)
        self.df = df

        return self.df

        print("Yüksek Oranda Null Oranı Olduğu İçin Elediğimiz Sütunlar:", list(dropped_for_missing_columns))
        print("Yüksek Oranda Null Oranı Olduğu İçin Elenen Sütunlar Sütun Sayısı:", len(dropped_for_missing_columns))
        print("NULL elemesi Sonrası Df'deki Satır ve Sütun Sayısı:", self.df.shape)

        end_time_elc = time.time()
        print(f"eliminating_null_columns çalışma süresi: {end_time_elc - start_time_elc :.4f} Saniyede Çalıştı")

    def separating_numerical_and_non_numerical_columns(self):
        start_time_separating = time.time()
        numerical_columns = self.df.select_dtypes(include="number")
        non_numerical_columns = self.df.select_dtypes(exclude="number")
        end_time_separating = time.time()
        print(f"separating_numerical_and_non_numerical_columns çalışma süresi: {end_time_separating - start_time_separating :.4f} Saniyede Çalıştı")

        print("Nümerik Kolon Sayısı:", numerical_columns.shape[1])
        print("Nümerik Kolonlar:", list(numerical_columns.columns))
        print("Nümerik Olmayan Kolon Sayısı:", non_numerical_columns.shape[1])
        print("Nümerik Olmayan Kolonlar:", list(non_numerical_columns.columns))
        return numerical_columns, non_numerical_columns

class Handling_Missing_Values:
    ##Bu fonksiyonlar Missing Valueları Handle edeceğiz, tahmine dayalı birer atama methodu ve

    def __init__(self,df):
        self.df = df
        self.numerical_columns = self.df.select_dtypes(include="number").columns.tolist()
        self.non_numerical_columns = self.df.select_dtypes(exclude="number").columns.tolist()

   ##IterativeImputer ile tahmine dayalı bir atama gerçekleştirebiliriz:
    def iterative_imputing(self):
        start_time_ii = time.time()
        self.it_imputer = IterativeImputer(max_iter=max_iter_for_iterative_imputing, random_state=42)
        self.df[self.numerical_columns] = self.it_imputer.fit_transform(self.df[self.numerical_columns])
        end_time_ii = time.time()
        print(f"Nümerik Değişkenler IterativeImputer Yöntemi: {end_time_ii - start_time_ii:.4f} Saniyede Çalıştı")
        print("Nümerik Değişkenler IterativeImputer Yöntemi ile Dolduruldu")

    ##İkinci bir yöntem nümerik kolonları 0 ile doldurmak
    def imputing_with_0 (self, numerical_columns):
        start_time_iw0 = time.time()
        self.df[numerical_columns] = self.df[numerical_columns].fillna(0)
        end_time_iw0 = time.time()
        print(f"Nümerik Değişkenleri 0 ile Doldurma Yöntemi: {end_time_iw0 - start_time_iw0:.4f} Saniyede Çalıştı")
        print("Nümerik Değişkenler 0 Değeri ile Doldurma İşlemi Başarıyla Çalıştı")

    ##Medyan ile doldurmak da başka bir seçenek

    def imputing_with_median (self, numerical_columns):
        start_time_iwm = time.time()
        self.median_imputer = SimpleImputer (strategy= "median")
        self.median_imputer.fit(self.df[numerical_columns])
        self.df[numerical_columns] = self.median_imputer.transform(self.df[numerical_columns])
        end_time_iwm = time.time()

        print(f"Nümerik Değişkenleri Medyan ile Doldurma Yöntemi: {end_time_iwm - start_time_iwm:.4f} Saniyede Çalıştı")
        print("Nümerik Değişkenler Her Sütundaki Medyan Değerleriyle Doldurma İşlemi Başarıyla Çalıştı")

    ##Kategorik  değişkenler için yöntemler:
    ##1 Unknown veya benzeri bir değere atama yapmak

    def imputing_cat_var_with_unknown (self, non_numerical_columns):
        start_time_icvwu = time.time()
        self.df[non_numerical_columns] = self.df[non_numerical_columns].fillna('Unknown')
        end_time_icvwu = time.time()

        print(f"Kateorik Değişkenleri Unknown ile Doldurma İşlemi: {end_time_icvwu - start_time_icvwu:.4f} Saniyede Çalıştı")
        print("Kategorik Değişkenler Unknown ile İşlemi Başarıyla Çalıştı")

    ##2. Yöntem en fazla tekrar eden değer ile doldurmak

    def imputing_cat_var_with_mostfreq(self):
        start_time_icvwm = time.time()
        self.most_frequent_imputer = SimpleImputer(strategy="most_frequent")
        self.df[self.non_numerical_columns] = self.most_frequent_imputer.fit_transform(self.df[self.non_numerical_columns])
        end_time_icvwm = time.time()

        print(f"Kategorik Değişkenler En Çok Tekrar Eden Değerler ile Doldurma İşlemi: {end_time_icvwm - start_time_icvwm:.4f} Saniyede Çalıştı")
        print("Kategorik Değişkenler En Çok Tekrar Eden Değerler ile Dolduruldu")

    def encode_and_impute(self):
        start_time_eae = time.time()
        for col in self.non_numerical_columns:
            le = LabelEncoder()
            self.df[col] = self.df[col].astype(str)
            self.df[col] = le.fit_transform(self.df[col])

        it_imputer_cat = IterativeImputer(max_iter=10, random_state=42)
        self.df[self.non_numerical_columns] = it_imputer_cat.fit_transform(self.df[self.non_numerical_columns])
        end_tine_eae = time.time()

        print(f"Kategorik Değişkenleri Label Encode Etme ve Sonrasında Iterative Impute İşlemi:  {end_tine_eae - start_time_eae:.4f} Saniyede Çalıştı")
        print("Kategorik Değişkenler Label Encoding Yapıldı ve Sonrasında Iterative Imputer Başarıyla Çalıştı")

    def get_transformed_df(self):
        return self.df

class Exploraty_Data_Analysis_1:  # Korelasyon ve Aykırı Değer Analizi
    def __init__(self, df):
        self.df = df
        self.numerical_columns = df.select_dtypes(include="number").columns
        self.outliers_info = {}

    def Correlation_Matrix(self):
        start_time_cm = time.time()
        plt.figure(figsize=(20, 15))
        correlation = self.df[self.numerical_columns].corr()
        sns.heatmap(correlation, vmin=-1, vmax=1, center=0, cmap='coolwarm', annot=True, fmt=".2f",
                    linewidths=1, linecolor='black')
        plt.title('Korelasyon Matrisi', fontsize=20)
        plt.show()
        end_time_cm = time.time()
        print(f"Korelasyon Matrisi {end_time_cm - start_time_cm:.4f} saniyede çizildi.")

    def Box_Plot(self):
        start_time_bp = time.time()
        for col in self.numerical_columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, x=col, palette='Set2')
            plt.title(f'{col} Kutu Grafiği', fontsize=16)
            plt.xlabel(col, fontsize=14)
            plt.ylabel('Değer', fontsize=14)
            plt.show()
        end_time_bp = time.time()
        print(f"Kutu Grafiği {end_time_bp - start_time_bp:.4f} saniyede çizildi.")

    def Outlier_Detection(self):
        start_time_od = time.time()
        for col in self.numerical_columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            IQR = q3 - q1
            upper_boundary = q3 + 1.5 * IQR
            lower_boundary = q1 - 1.5 * IQR

            # Aykırı değer bilgilerini outliers_info sözlüğüne kaydet
            self.outliers_info[col] = {
                'lower_boundary': lower_boundary,
                'upper_boundary': upper_boundary,
                'outliers': self.df[(self.df[col] < lower_boundary) | (self.df[col] > upper_boundary)]
            }
        end_time_od = time.time()
        print(f"Aykırı değerler {end_time_od - start_time_od:.4f} saniyede tespit edildi.")

    def Supressing_Outliers(self):
        start_time_so = time.time()
        for col in self.numerical_columns:
            if col in self.outliers_info:
                lower_boundary = self.outliers_info[col]['lower_boundary']
                upper_boundary = self.outliers_info[col]['upper_boundary']

                # Aykırı değerleri alt/üst sınıra baskılama
                self.df[col] = np.where(self.df[col] > upper_boundary, upper_boundary,
                                        np.where(self.df[col] < lower_boundary, lower_boundary, self.df[col]))
        end_time_so = time.time()
        print(f"Aykırı değerler {end_time_so - start_time_so:.4f} saniyede baskılandı.")

    def Changing_Outliers_With_Median(self):
        start_time_com = time.time()
        for col in self.numerical_columns:
            if col in self.outliers_info:
                lower_boundary = self.outliers_info[col]['lower_boundary']
                upper_boundary = self.outliers_info[col]['upper_boundary']

                # Her sütun için medyan hesaplama
                median = self.df[col].median()

                # Aykırı değerleri medyan ile değiştirme
                self.df[col] = np.where(self.df[col] > upper_boundary, median,
                                        np.where(self.df[col] < lower_boundary, median, self.df[col]))
        end_time_com = time.time()
        print(f"Aykırı değerler {end_time_com - start_time_com:.4f} saniyede medyan ile değiştirildi.")

class Exploratory_Data_Analysis_2:
    def __init__(self, df, dependent_variable):
        self.df = df.copy()
        self.dependent_variable = dependent_variable
        self.numerical_correlation_results = []
        self.dropped_for_no_variance = []
        self.dropped_for_high_correlation = []

    def Correlation_Between_Numerical_Columns(self, threshold=threshold_for_correlation_between_numerical_columns):
        start_time = time.time()
        numerical_columns = self.df.select_dtypes(include="number").columns.tolist()

        # Bağımlı değişkeni eleme listesinden çıkar
        if self.dependent_variable in numerical_columns:
            numerical_columns.remove(self.dependent_variable)

        if not numerical_columns:
            print("Hata: Sayısal sütun bulunamadı.")
            return

        correlation_matrix = self.df[numerical_columns].corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1))

        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

        # Bağımlı değişken ile en yüksek korelasyona sahip olanı koru
        for col_to_drop in list(to_drop):
            correlated_with_dependent = [col for col in self.df.columns if col in numerical_columns and col != self.dependent_variable]
            correlations = {col: self.df[col].corr(self.df[self.dependent_variable]) for col in correlated_with_dependent}

            if col_to_drop in correlations:
                del correlations[col_to_drop]

            if len(correlations) > 0 and col_to_drop in to_drop and col_to_drop in numerical_columns:
                max_corr_col = max(correlations, key=correlations.get)
                if correlations[max_corr_col] > self.df[col_to_drop].corr(self.df[self.dependent_variable]):
                    if col_to_drop in self.df.columns:
                        to_drop.remove(col_to_drop)

        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.dropped_for_high_correlation.extend(to_drop)
            print(f"{len(to_drop)} adet sütun yüksek korelasyon nedeniyle kaldırıldı.")
        else:
            print("Kaldırılacak sütun bulunamadı.")

    def get_transformed_df(self):
        return self.df

    def get_dropped_columns(self):
        return self.dropped_for_high_correlation

class VIF_Analysis:
    def __init__(self, df, dependent_variable, threshold):
        self.df = df.copy()
        self.dependent_variable = dependent_variable
        self.threshold = threshold
        self.Eliminated_For_VIF_Value = []
        self.vif_data = None

    def calculate_vif(self):
        # Bağımlı değişkeni VIF hesaplaması dışındaki tüm sayısal sütunları al
        numerical_cols_for_vif = [col for col in self.df.select_dtypes(include="number").columns if col != self.dependent_variable]
        temp_df = self.df[numerical_cols_for_vif].replace([np.inf, -np.inf], np.nan).dropna()

        vif_data = pd.DataFrame()
        vif_data["Feature"] = temp_df.columns
        vif_data["VIF"] = [
            variance_inflation_factor(temp_df.values, i)
            for i in range(temp_df.shape[1])
        ]
        return vif_data

    def VIF(self):
        start_time_vif = time.time()

        df_for_vif = self.df.copy()

        while True:
            # Bağımlı değişkeni işlem dışı bırak
            numerical_cols_for_vif = [col for col in df_for_vif.select_dtypes(include="number").columns if col != self.dependent_variable]
            temp_df = df_for_vif[numerical_cols_for_vif].replace([np.inf, -np.inf], np.nan).dropna()

            if temp_df.empty:
                print("Tüm özellikler elendi. VIF analizi sonlandırıldı.")
                break

            self.vif_data = self.calculate_vif()
            self.vif_data = self.vif_data.sort_values(by="VIF", ascending=False)

            if not self.vif_data.empty and self.vif_data["VIF"].iloc[0] > self.threshold:
                feature_to_eliminate = self.vif_data["Feature"].iloc[0]
                self.Eliminated_For_VIF_Value.append(feature_to_eliminate)
                df_for_vif.drop(columns=[feature_to_eliminate], inplace=True)
                print(f"{feature_to_eliminate} column was eliminated due to high VIF value ({self.vif_data['VIF'].iloc[0]:.4f}).")
            else:
                print("All features have VIF values below the threshold.")
                break

        end_time_vif = time.time()
        elapsed_time_vif = end_time_vif - start_time_vif
        print("VIF Analysis Execution Time:", f"{elapsed_time_vif:.4f} seconds")
        print("Eliminated Features:", self.Eliminated_For_VIF_Value)

        self.df_final = self.df.copy()
        print(f"{self.Eliminated_For_VIF_Value} değişkenleri VIF değerinden dolayı elendi")
        print("Güncel DF Boyu:", self.df_final.shape)

        if self.vif_data is not None:
            print(self.vif_data)
        else:
            print("No VIF data available due to early termination.")

        return self.vif_data

class Exploratory_Data_Analysis_3:
    def __init__(self, df, dependent_variable):
        self.df = df
        self.dependent_variable = dependent_variable
        self.categorical_columns_for_correlation = [
            col for col in df.select_dtypes(include='object').columns
            if col != self.dependent_variable
        ]
        self.cramers_v_results = []

    def cramers_v(self, x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        return np.sqrt(phi2 / min(k-1, r-1))

    def correlation_between_categorical_columns(self, threshold = correlation_between_categorical_columns_treshold):
        start_time_cramerv = time.time()
        high_corr_pairs = []

        for col1, col2, cramers_v_value in high_corr_pairs:
            if col1 in self.df.columns:
                correlation_col1 = self.df[col1].astype('category').cat.codes.corr(self.df[self.dependent_variable])
            else:
                correlation_col1 = None

            if col2 in self.df.columns:
                correlation_col2 = self.df[col2].astype('category').cat.codes.corr(self.df[self.dependent_variable])
            else:
                correlation_col2 = None

            # Bağımlı değişkeni drop listesine ekleme
            if correlation_col1 is not None and (correlation_col2 is None or correlation_col1 >= correlation_col2):
                if col1 in self.df.columns and col1 != self.dependent_variable:
                    self.df = self.df.drop(columns=[col1], errors='ignore')
                    print(f"{col1} sütunu, {col2} sütunu ile yüksek Cramér's V ({threshold} üstü) korelasyonuna sahip olduğu için düşürüldü. Bağımlı değişkenle korelasyonu: {correlation_col1:.4f}")
            else:
                if col2 in self.df.columns and col2 != self.dependent_variable:
                    self.df = self.df.drop(columns=[col2], errors='ignore')
                    print(f"{col2} sütunu, {col1} sütunu ile yüksek Cramér's V ({threshold} üstü) korelasyonuna sahip olduğu için düşürüldü. Bağımlı değişkenle korelasyonu: {correlation_col2:.4f}")

            self.cramers_v_results.append((col1, col2, cramers_v_value, correlation_col1, correlation_col2))


        end_time_cramerv = time.time()
        elapsed_time_cramer = end_time_cramerv - start_time_cramerv
        print(f"{elapsed_time_cramer:.4f} saniyede Cramer Testine Göre Eleminasyon Yapıldı")
        print(f"Cramér's V Elemesi Sonrası Güncel DataFrame Boyutu: {self.df.shape}")
        return self.df

def chi_squared_test(df, dependent_variable, alpha=chi_squared_test_alpha):
    non_numerical_columns = [col for col in df.select_dtypes(include=['object']).columns.tolist() if col != dependent_variable]
    to_drop = []

    for i in range(len(non_numerical_columns)):
        for j in range(i+1, len(non_numerical_columns)):
            col1 = non_numerical_columns[i]
            col2 = non_numerical_columns[j]
            contingency_table = pd.crosstab(df[col1], df[col2])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            if p > alpha:
                corr_col1 = pd.crosstab(df[col1], df[dependent_variable], normalize='index').apply(lambda r: r.max() - r.min(), axis=1).mean()
                corr_col2 = pd.crosstab(df[col2], df[dependent_variable], normalize='index').apply(lambda r: r.max() - r.min(), axis=1).mean()
                if corr_col1 > corr_col2:
                    to_drop.append(col2)
                else:
                    to_drop.append(col1)
    to_drop = list(set(to_drop))
    df = df.drop(columns=to_drop)

    print(f"Elenecek değişkenler: {to_drop}")
    return df

class Variable_Monitoring:
    def __init__(self, oot_data, current_data):
        self.oot_data = oot_data
        self.current_data = current_data

    def calculate_psi(self, expected, actual, bins=10):
        expected_percents = np.histogram(expected, bins=bins, density=True)[0]
        actual_percents = np.histogram(actual, bins=bins, density=True)[0]
        expected_percents = np.clip(expected_percents, 1e-10, 1)
        actual_percents = np.clip(actual_percents, 1e-10, 1)
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi

    def calculate_csi(self, expected, actual, bins=10):
        expected_percents = np.histogram(expected, bins=bins, density=True)[0]
        actual_percents = np.histogram(actual, bins=bins, density=True)[0]
        expected_percents = np.clip(expected_percents, 1e-10, 1)
        actual_percents = np.clip(actual_percents, 1e-10, 1)
        csi = np.sum(np.abs(actual_percents - expected_percents))
        return csi

    def filter_stable_features(self, features, psi_threshold=0.25, csi_threshold=0.1, bins=10):
        stable_features = []
        psi_results = {}
        csi_results = {}

        for feature in features:
            if feature in self.oot_data.columns and feature in self.current_data.columns:
                psi_value = self.calculate_psi(self.oot_data[feature], self.current_data[feature], bins)
                csi_value = self.calculate_csi(self.oot_data[feature], self.current_data[feature], bins)
                psi_results[feature] = psi_value
                csi_results[feature] = csi_value

                if psi_value <= psi_threshold and csi_value <= csi_threshold:
                    stable_features.append(feature)

        return stable_features, psi_results, csi_results

class EncodingMethods:

    def __init__(self, df):
        self.df = df
        self.categorical_columns = df.select_dtypes(exclude= "number").columns.tolist()

    ##Ordinal encoding, sıralama (order) bilgisinin önemli olduğu durumlarda kullanılır
    def OrdinalEncoding(self, ordinal_categorical_columns):
        start_time_ordinal_encoding = time.time()
        ordinal_encoder = OrdinalEncoder()
        self.df[ordinal_categorical_columns] = ordinal_encoder.fit_transform(self.df[ordinal_categorical_columns])
        end_time_ordinal_encoding = time.time()
        elapsed_time_ordinal_encoding = end_time_ordinal_encoding - start_time_ordinal_encoding
        print("Ordinal Encoding Çalışma Zamanı:" f"{elapsed_time_ordinal_encoding:.4f}")
        print("Ordinal Encoding Sonrası Data Frame Boyutu:", self.df.shape)
        return self.df

    ## One-Hot Encoding sıralı olmayan sütunlar için uygundur
    def OneHotEncoding(self):
        start_time_ohe = time.time()
        for col in self.categorical_columns:
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
            self.df = pd.concat([self.df, dummies], axis=1)
            self.df.drop(col, axis=1, inplace=True)
        end_time_ohe = time.time()
        elapsed_time_ohe = end_time_ohe - start_time_ohe
        print("One Hot Encoding Çalışma Süresi:", f"{elapsed_time_ohe:.4f}")
        print("One-Hot Encoding sonrası Data Frame boyutu:", self.df.shape)
        return self.df

    ## Binary Encoding, sıralamanın önemli olmadığı durumlarda kullanılabilir
    def BinaryEncoding(self, categorical_columns):
        start_time_bin_encoding = time.time()
        binary_encoder = BinaryEncoder()
        self.df[categorical_columns] = binary_encoder.fit_transform(self.df[categorical_columns])
        end_time_bin_encoding = time.time()
        elapsed_time_bin_encoding = end_time_bin_encoding - start_time_bin_encoding
        print("Binary Encoding Çalışma Süresi:" f"{elapsed_time_bin_encoding:.4f}")
        print("Binary Encoding Sonrası Df Boyutu:", self.df.shape)
        return self.df

    ## BackwardDifferenceEncoding, sıralı değerler arasında anlamlı farklar olduğunda kullanılır
    def BackwardDifferenceEncoding(self, ordinal_categorical_columns):
        start_time_bwd_encoding = time.time()
        backward_diff_encoder = BackwardDifferenceEncoder()
        self.df[ordinal_categorical_columns] = backward_diff_encoder.fit_transform(self.df[ordinal_categorical_columns])
        end_time_bwd_encoding = time.time()
        elapsed_time_bwd_encoding = end_time_bwd_encoding - start_time_bwd_encoding
        print("Backward Difference Encoding Çalışma Zamanı:", f"{elapsed_time_bwd_encoding:.4f}")
        print("Backward Difference Encoding Sonrası Df Boyutu:", self.df.shape)
        return self.df

    ## Hash Encoding, çok fazla kategori olduğunda kullanılır
    def HashingEncoding(self, categorical_columns, n_components):
        start_time_hashing_encoding = time.time()
        hashing_encoder = HashingEncoder(n_components=n_components)
        self.df = hashing_encoder.fit_transform(self.df[categorical_columns])
        end_time_hashing_encoding = time.time()
        elapsed_time_hashing_encoding = end_time_hashing_encoding - start_time_hashing_encoding
        print("Hashing Encoding Çalışma Süresi:", f"{elapsed_time_hashing_encoding:.4f}")
        print("Hashing Encoding Sonrası Data Frame Boyutu:", self.df.shape)
        return self.df

    ## Label Encoding
    def LabelEncoding(self):
        start_time_le = time.time()
        le = LabelEncoder()
        for col in self.categorical_columns:
            self.df[col] = le.fit_transform(self.df[col].astype(str))
        end_time_le = time.time()
        elapsed_time_le = end_time_le - start_time_le
        print("Label Encoding Çalışma Süresi:" f"{elapsed_time_le:.4f}")
        print("Label Encoding sonrası Data Frame boyutu:", self.df.shape)
        return self.df

    ## Target Encoding
    def TargetEncoding(self, dependent_variable):
        start_time_te = time.time()
        target_encoder = TargetEncoder(smoothing=1.0, random_state=42)
        self.df[self.categorical_columns] = target_encoder.fit_transform(self.df[self.categorical_columns], self.df[dependent_variable])
        end_time_te = time.time()
        elapsed_time_te = end_time_te - start_time_te
        print("Target Encoding Çalışma Süresi:", f"{elapsed_time_te:.4f}")
        print("Target Encoding sonrası Data Frame boyutu:", self.df.shape)
        return self.df

    ## kFold Target Encoding
    def kFoldTargetEncoding(self, dependent_variable, n_splits=5):
        start_time_kfte = time.time()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        encoder = TargetEncoder(smoothing=1.0)
        encoded_df = self.df.copy()

        for train_index, test_index in kf.split(self.df):
            train, test = self.df.iloc[train_index], self.df.iloc[test_index]
            X_train = train[self.categorical_columns]
            y_train = train[dependent_variable]
            X_test = test[self.categorical_columns]
            encoder.fit(X_train, y_train)
            encoded_values = encoder.transform(X_test)
            encoded_df.loc[test_index, self.categorical_columns] = encoded_values

        end_time_kfte = time.time()
        elapsed_time_kfte = end_time_kfte - start_time_kfte

        print("kFold Target Encoding sonrası Data Frame boyutu:", encoded_df.shape)
        print(f"kFold Target Encoding Çalışma Süresi: {elapsed_time_kfte:.4f} saniye")

        return encoded_df

class FeaureScalingMethods:
    def __init__(self, df):
        self.df = df

    def standardize(self, columns):
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df

    def robust_scale(self, columns):
        scaler = RobustScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df

    def normalize(self, columns):
        normalizer = Normalizer()
        self.df[columns] = normalizer.fit_transform(self.df[columns])
        return self.df

    def min_max_scale(self, columns):
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df

class DataTypeFixer:
    def __init__(self, df):
        self.df = df

    def fix_data_types(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    self.df[col] = self.df[col].astype(float)
                    print(f"'{col}' kolonu başarıyla 'float' veri tipine dönüştürüldü.")
                except (ValueError, TypeError):  # Bu satırı değiştirdik
                    print(f"'{col}' kolonu 'float' veri tipine çevrilemedi.")
        return self.df

class SeperatingTrainTest:
    def __init__(self, df, dependent_variable):
        self.df = df
        self.dependent_variable = dependent_variable

    def train_test_splitter(self, test_size=test_size):
        X = self.df.drop(self.dependent_variable, axis=1)
        y = self.df[self.dependent_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

class SklearnXGBClassifier(XGBClassifier, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _more_tags(self):
        tags = super()._more_tags()
        tags["estimator_type"] = "classifier"
        return tags


class RFECV_Class:
    def __init__(self, X_train, y_train, X_test, y_test, cv):
        if isinstance(y_train, pd.Series):
            self.y_train = y_train.values.flatten()
        elif isinstance(y_train, np.ndarray) and y_train.ndim > 1:
            self.y_train = y_train.flatten()
        else:
            self.y_train = y_train

        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv

        self.estimator = SklearnXGBClassifier(
            learning_rate=0.1,
            enable_categorical=True,
            eval_metric="logloss"
        )

    def select_features_with_rfecv(self, min_features_to_select=1):
        rfecv = SklearnRFECV(
            estimator=self.estimator,
            step=1,
            scoring=make_scorer(f1_score, average="binary"),
            cv=self.cv,
            min_features_to_select=min_features_to_select,
            n_jobs=-1
        )

        rfecv.fit(self.X_train, self.y_train)
        print(f"Optimal number of features: {rfecv.n_features_}")

        # Seçilen feature isimleri
        if hasattr(self.X_train, "columns"):
            selected_features = self.X_train.columns[rfecv.support_].tolist()
        else:
            selected_features = [f"Feature_{i}" for i in range(self.X_train.shape[1])]
            selected_features = [selected_features[i] for i in range(len(rfecv.support_)) if rfecv.support_[i]]

        if hasattr(self.X_train, "columns"):
            X_train_selected = self.X_train[selected_features]
            X_test_selected = self.X_test[selected_features]
        else:
            X_train_selected = self.X_train[:, rfecv.support_]
            X_test_selected = self.X_test[:, rfecv.support_]

        self.estimator.fit(X_train_selected, self.y_train)


        try:
            feature_importance = self.estimator.get_booster().get_score(importance_type="gain")
            importance_df = pd.DataFrame(
                sorted(feature_importance.items(), key=lambda item: item[1], reverse=True),
                columns=["Feature", "Gain Importance"]
            )
        except:
            importance_df = pd.DataFrame({
                "Feature": selected_features,
                "Gain Importance": self.estimator.feature_importances_
            }).sort_values(by="Gain Importance", ascending=False)

        print("Selected Features and Their Gain Importance:")
        print(importance_df)

        self.X_train = X_train_selected
        self.X_test = X_test_selected

        return self.X_train, self.X_test, rfecv, importance_df

class Hyper_Parameter_Tuning_With_Optuna:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.study_lgbm = None
        self.study_xgb = None

    def lgbm_objective(self, trial):
        learning_rate = trial.suggest_float("learning_rate", min_lr_for_lgbm_optuna,max_lr_for_lgbm_optuna)
        num_leaves = trial.suggest_int("num_leaves", min_num_leaves_for_lgbm_optuna , max_num_leaves_for_lgbm_optuna )
        max_depth = trial.suggest_int("max_depth",min_max_depth_for_lgbm_optuna ,max_max_depth_for_lgbm_optuna )
        min_child_samples = trial.suggest_int("min_child_samples",min_child_samples_for_lgbm_optuna,max_child_samples_for_lgbm_optuna)
        subsample = trial.suggest_float("subsample", min_subsample_for_lgbm_optuna, max_subsample_for_lgbm_optuna)
        colsample_bytree = trial.suggest_float("colsample_bytree", min_colsample_bytree_for_lgbm_optuna,max_colsample_bytree_for_lgbm_optuna)
        n_estimators = trial.suggest_int("n_estimators", min_n_estimators_for_lgbm_optuna,max_n_estimators_for_lgbm_optuna)
        class_weight = trial.suggest_categorical("class_weight", ["balanced", class_weights_for_lgbm])

        lgbm_model = lgbm.LGBMClassifier(
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=42,
            verbose = -1
        )

        lgbm_model.fit(self.X_train, self.y_train)
        y_pred = lgbm_model.predict(self.X_test)
        return f1_score(self.y_test, y_pred)

    def xgb_objective(self, trial):
        learning_rate = trial.suggest_float("learning_rate", min_lr_for_xgb_optuna,max_lr_for_xgb_optuna)
        max_depth = trial.suggest_int("max_depth",min_max_depth_for_xgb_optuna,max_max_depth_for_xgb_optuna)
        min_child_weight = trial.suggest_int("min_child_weight", min_min_child_weight_for_xgb_optuna, max_min_child_weight_for_xgb_optuna)
        subsample = trial.suggest_float("subsample",min_subsample_for_xgb_optuna,max_subsample_for_xgb_optuna )
        colsample_bytree = trial.suggest_float("colsample_bytree", min_colsample_bytree_for_xgb_optuna, max_colsample_bytree_for_xgb_optuna)
        n_estimators = trial.suggest_int("n_estimators", min_n_estimators_for_xgb_optuna,max_n_estimators_for_xgb_optuna)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", min_class_weight_for_xgb_optuna, max_class_weight_for_xgb_optuna)

        xgb_model = xgb.XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight,
            enable_categorical=True,
            random_state=42
        )

        xgb_model.fit(self.X_train, self.y_train)
        y_pred = xgb_model.predict(self.X_test)
        return f1_score(self.y_test, y_pred)

    def tune_hyperparameters(self,n_trials):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings('ignore')

        start_time_lgbm = time.time()
        self.study_lgbm = optuna.create_study(direction="maximize")
        self.study_lgbm.optimize(self.lgbm_objective, n_trials=n_trials)
        end_time_lgbm = time.time()
        lgbm_duration = end_time_lgbm - start_time_lgbm

        print("\nLightGBM Tuning:")
        print("Best LightGBM Parameters:", self.study_lgbm.best_params)
        print("Best LightGBM F1 Score:", self.study_lgbm.best_value)
        print(f"LightGBM Tuning Süresi: {lgbm_duration:.2f} saniye")

        start_time_xgb = time.time()
        self.study_xgb = optuna.create_study(direction="maximize")
        self.study_xgb.optimize(self.xgb_objective, n_trials=n_trials)
        end_time_xgb = time.time()
        xgb_duration = end_time_xgb - start_time_xgb

        print("\nXGBoost Tuning:")
        print("Best XGBoost Parameters:", self.study_xgb.best_params)
        print("Best XGBoost F1 Score:", self.study_xgb.best_value)
        print(f"XGBoost Tuning Süresi: {xgb_duration:.2f} saniye")

class Final_Model_Builder:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def build_lgbm_final_model(self, best_params_lgbm):
        lgbm_final_model = lgbm.LGBMClassifier(
            learning_rate=best_params_lgbm['learning_rate'],
            num_leaves=best_params_lgbm['num_leaves'],
            max_depth=best_params_lgbm['max_depth'],
            min_child_samples=best_params_lgbm['min_child_samples'],
            subsample=best_params_lgbm['subsample'],
            colsample_bytree=best_params_lgbm['colsample_bytree'],
            n_estimators=best_params_lgbm['n_estimators'],
            class_weight=best_params_lgbm['class_weight'],
            random_state=random_state,
            verbose=-1
        )

        lgbm_final_model.fit(self.X_train, self.y_train)
        y_pred = lgbm_final_model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred)
        print(f"LightGBM Final Model F1 Score: {f1:.4f}")

        return lgbm_final_model

    def build_xgb_final_model(self, best_params_xgb):

        scale_pos_weight = best_params_xgb['scale_pos_weight']
        if scale_pos_weight == 0.0:
            scale_pos_weight = 1.0

        xgb_final_model = xgb.XGBClassifier(
            learning_rate=best_params_xgb['learning_rate'],
            max_depth=best_params_xgb['max_depth'],
            min_child_weight=best_params_xgb['min_child_weight'],
            subsample=best_params_xgb['subsample'],
            colsample_bytree=best_params_xgb['colsample_bytree'],
            n_estimators=best_params_xgb['n_estimators'],
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            enable_categorical=True
        )

        xgb_final_model.fit(self.X_train, self.y_train)
        y_pred = xgb_final_model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred)
        print(f"XGBoost Final Model F1 Score: {f1:.4f}")

        return xgb_final_model

class Model_Explainability:
    def __init__(self, model, training_data, feature_names, class_names):
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None

    def _get_explainer(self, explainer_type):
        if explainer_type == 'shap':
            # SHAP için uygun explainer'ı seçin
            if 'xgboost' in str(type(self.model)):
                self.explainer = shap.TreeExplainer(self.model)
            elif 'lightgbm' in str(type(self.model)):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = shap.Explainer(self.model.predict, self.training_data)
        elif explainer_type == 'lime':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                self.training_data.values,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification'
            )

    def explain_with_shap(self, data_sample):
        self._get_explainer('shap')
        shap_values = self.explainer.shap_values(data_sample)
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values, data_sample, plot_type="bar", feature_names=self.feature_names)
        else:
            shap.summary_plot(shap_values, data_sample, feature_names=self.feature_names)

    def explain_with_lime(self, instance, num_features=30):
        self._get_explainer('lime')
        exp = self.explainer.explain_instance(
            instance, self.model.predict_proba, num_features=num_features
        )
        exp.as_pyplot_figure()
        plt.show()

def save_and_export_model(model, model_name, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    pkl_path = os.path.join(model_dir, f"{model_name}.pkl")

    try:
        joblib.dump(model, pkl_path)
        print(f"✅ {model_name} pickled: {pkl_path}")
    except Exception as e:
        print(f"Pickle kaydetme hatası: {e}")

class Pipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, func, *args, **kwargs):
        self.steps.append((func, args, kwargs))

    def run(self, initial_data=None):
        import os
        os.makedirs("models", exist_ok=True)

        data = initial_data
        for func, args, kwargs in self.steps:
            try:
                if data is not None:
                    data = func(data, *args, **kwargs)
                else:
                    data = func(*args, **kwargs)
                print(f"{func.__name__} başarıyla çalıştı")
            except Exception as e:
                print(f"{func.__name__} sırasında hata oluştu: {e}")
        return data

pipeline = Pipeline()
pipeline_start_time = time.time()

NULL_Result_DF = checking_null_number_and_ratio_of_columns(df)
print(NULL_Result_DF)

Preprocess1 = Preprocess_1(df)

try:
    Preprocess1.eliminating_null_columns(dependent_variable=dependent_variable, treshold_for_eleminating_null_columns=treshold_for_eleminating_null_columns)
    Preprocess1.separating_numerical_and_non_numerical_columns()
    print(df.shape)
    print("Preprocess1 Hatasız çalıştı")
except Exception as e:
    print("Hata Alındı", e)


Handling_Missing_Values = Handling_Missing_Values(df)

try:
    Handling_Missing_Values.iterative_imputing()
    missing_columns = df.columns[df.isna().any()]
    categorical_columns_with_missing = df[missing_columns].select_dtypes(include=['object', 'category']).columns

    if len(categorical_columns_with_missing) > 0:
        Handling_Missing_Values.imputing_cat_var_with_unknown(categorical_columns_with_missing)
    else:
        print("Veri Seti Nümeri Kolon İçermediği için doldurma yapılmadı")

    for col in missing_columns:
        print(col, df[col].isna().sum())

    print("Handling_Missing_Values Hatasız Çalıştı")

except Exception as e:
    print(e , "Hatası Alındı")

eda = Exploraty_Data_Analysis_1(df)

try:
    eda.Correlation_Matrix()
    eda.Outlier_Detection()
    eda.Supressing_Outliers()
    print("ExploratyDataAnalysis1 Hatasız Çalıştı; Korelasyon Matrisi Yukarıdadır, Outlierlar Tespit Edildi ve Baskılandı")

except Exception as e:
    print(e , "Hatası Alındı" )

try:
    eda2 = Exploratory_Data_Analysis_2(df, dependent_variable=dependent_variable)
    eda2.Correlation_Between_Numerical_Columns(threshold=threshold_for_correlation_between_numerical_columns)


    df = eda2.get_transformed_df()

    print("Korelasyon Analizi Başarıyla Tamamlandı")
    print(f"Veri çerçevesinin boyutu (işlemden sonra): {df.shape}")
    dropped_columns = eda2.get_dropped_columns()
    if dropped_columns:
        print("Kaldırılan Sütunlar:", dropped_columns)
except ValueError as ve:
    print(f"Hata: {ve}")
except Exception as e:
    print(f"Hata: {e}")

vif_analysis = VIF_Analysis(df, dependent_variable=dependent_variable, threshold=treshold_for_vif_analysis)
try:
    vif_df = vif_analysis.VIF()
    print("VIF Analizi Başarıyla Tamamlandı")
except Exception as e:
    print(e, "Hatası Alındı")


eda3 = Exploratory_Data_Analysis_3(df, dependent_variable=dependent_variable)

try:
    df = eda3.correlation_between_categorical_columns(
        threshold=correlation_between_categorical_columns_treshold
    )
    print("Exploraty_Data_Analysis_3 Başarıyla Çalıştı")
    print("Cramer V testi sonrası güncel Data Frame Boyutu:", df.shape)
except Exception as e:
    print(e, "Hatası Alındı")



encoding_methods = EncodingMethods(df)
df = encoding_methods.LabelEncoding()

try:
    fixer = DataTypeFixer(df)
    fixer.fix_data_types()
except Exception as e:
    print(e , "Veri Seti Dönüştürülemedi")


print(df.columns)
splitter = SeperatingTrainTest(df, dependent_variable=dependent_variable)
X_train, X_test, y_train, y_test = splitter.train_test_splitter(test_size=test_size)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


try:
    cv = KFold(n_splits=n_splits_for_rfe,
               shuffle=True,
               random_state=random_state)
    RFECV = RFECV_Class(X_train, y_train, X_test, y_test, cv)
    X_train_selected, X_test_selected, rfecv_model, feature_importance = RFECV.select_features_with_rfecv(min_features_to_select=min_feature_selected_for_rfe)

    if hasattr(X_train_selected, 'columns'):
        selected_feature_names = X_train_selected.columns.tolist()
    else:
        selected_feature_names = [f"f{i}" for i in range(X_train_selected.shape[1])]

except Exception as e:
    print(e, "Hatası Alındı")

try:
    XGB_basic = xgb.XGBClassifier(learning_rate=initial_learning_rate_for_xgb_before_optuna
                                ,enable_categorical=True
                                ,random_state = random_state)
    XGB_basic.fit(X_train, y_train)
    XGB_basic_pred = XGB_basic.predict(X_test)
    print("XGBoost Model without hyper parameter tuning:")
    print(classification_report(y_test, XGB_basic_pred))
    print(f"F1 Skoru: {f1_score(y_test, XGB_basic_pred)}")


except Exception as e:
    print(e ,"Hatası Alındı")

try:
    lgbm_basic = lgbm.LGBMClassifier(learning_rate=initial_learning_rate_for_lgbm_before_optuna
                                    , verbose = -1
                                    ,random_state= random_state)
    lgbm_basic.fit(X_train, y_train)
    lgbm_basic_pred = lgbm_basic.predict(X_test)
    print("LightGBM Model without hyper parameter tuning:")
    print(classification_report(y_test, lgbm_basic_pred))
    print(f"F1 Skoru: {f1_score(y_test, lgbm_basic_pred)}")

except Exception as e:
    print(e, "LightGBM Model Hatası Alındı")


try:
    hyperparameter_tuner = Hyper_Parameter_Tuning_With_Optuna(X_train, y_train, X_test, y_test)
    n_trials = n_trials_for_optuna_optimization
    hyperparameter_tuner.tune_hyperparameters(n_trials)
    best_params_lgbm = hyperparameter_tuner.study_lgbm.best_params
    best_params_xgb = hyperparameter_tuner.study_xgb.best_params

    final_model_builder = Final_Model_Builder(X_train, y_train, X_test, y_test)
    lgbm_final_model = final_model_builder.build_lgbm_final_model(best_params_lgbm)
    xgb_final_model = final_model_builder.build_xgb_final_model(best_params_xgb)

    print("Optuna tarafından seçilen en iyi LightGBM parametreler:", best_params_lgbm)
    print("LightGBM final modeli başarıyla oluşturuldu.")
    print("Optuna tarafından seçilen en iyi XGBoost parametreler:", best_params_xgb)
    print("XGBoost final modeli başarıyla oluşturuldu.")

 ##Pickle
    save_and_export_model(lgbm_final_model, 'lgbm_classifier')
    save_and_export_model(xgb_final_model, 'xgb_classifier')

except Exception as e:
    print(f"Hyperparameter tuning hatası: {e}")
    lgbm_final_model = lgbm_basic
    xgb_final_model = XGB_basic
    print("Hata nedeniyle varsayılan modeller kullanılacak.")

try:

    sample_size = min(len(X_train), 1000)
    feature_names = X_train.columns.tolist()
    class_names = [str(x) for x in sorted(y_train.unique())]

    X_sample = X_train.sample(n=sample_size, random_state=random_state)


    print("\n📊 LightGBM Açıklanabilirlik Analizi...")
    lgbm_explainer = Model_Explainability(
        model=lgbm_final_model,
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names
    )
    lgbm_explainer.explain_with_shap(X_sample)
    lgbm_explainer.explain_with_lime(X_test.iloc[0])


    print("\n📊 XGBoost Açıklanabilirlik Analizi...")
    xgb_explainer = Model_Explainability(
        model=xgb_final_model,
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names
    )
    xgb_explainer.explain_with_shap(X_sample)
    xgb_explainer.explain_with_lime(X_test.iloc[0])

except Exception as e:
    print(f"Açıklanabilirlik analizi hatası: {str(e)}")

pipeline_end_time = time.time()
elapsed_time_pipeline = pipeline_end_time - pipeline_start_time
print(f"\n✅ Pipeline Çalışma Zamanı: {elapsed_time_pipeline:.4f} saniye")

