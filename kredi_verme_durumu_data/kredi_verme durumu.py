
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Loan_default.csv")

print("Veri seti boyutu:", df.shape)
print("\nİlk 5 satır:")
print(df.head())
print(df.info())

missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    "missing_count":missing_count,
    "missing_percent":missing_percent,
}).sort_values(by="missing_percent",ascending=False)

print(missing_df)

plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(),cbar=False)
plt.title("eksik değer ısı haritası")
plt.show()
"""
# Eksik olma durumunda ne yapılırdı ?

1)Az eksik değer 
df[ıncome] = df["ıncome"].fillna(df[ıncome].mean())

2)orta seviye eksik değer
örneğin gelir eksik ama müşterinin yaş grubu biliniyor 
df["ıncome"] = df.grouby("ageGroup")["ıncome"].transform(lambda x: x:fillna(x.mean()))

veya

Orta seviyede eksik varsa en iyi yöntemdir.

KNN ile doldurma:
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

3)çok fazla eksik varsa

df.drop("somecolumn",axis = 1,inplace=true)
"""


num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object","category"]).columns

print("sayısal kolonlar:\n",num_cols)
print("kategorik kolonlar:\n",cat_cols)

plt.figure(figsize=(5,4))
df["Default"].value_counts().plot(kind ="bar")
plt.title("DEFAULT DAĞILIMI")
plt.xlabel("Default (0 = Ödemiş, 1 = Ödeyememiş)")
plt.ylabel("Adet")
plt.show()
print(df['Default'].value_counts(normalize=True))

df[num_cols].hist(figsize=(25,20),bins = 30)
plt.suptitle("sayısal kolonların dağılımı")
plt.show()

for col in cat_cols:
    plt.figure(figsize=(7,4))
    df[col].value_counts().head(10).plot(kind="bar")
    plt.title(f"{col} frekans dağılımı")
    plt.show()


for col in num_cols:
    plt.figure(figsize=(25,20))
    sns.boxplot(x="Default" , y = col , data=df)
    plt.title(f"{col} - default ilşikisi")
    plt.show()


plt.figure(figsize=(15,12))
sns.heatmap(df[num_cols].corr(),annot=False,cmap="coolwarm")
plt.title("korelasyon matrisi")
plt.show()

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 *IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col]< lower) | (df[col]> upper)]
    print(f"{col}:{len(outliers)} adet aykırı değer ")


for col in cat_cols:
    plt.figure(figsize=(7,4))
    sns.countplot(data=df, x=col, hue='Default')
    plt.xticks(rotation=45)
    plt.title(f"{col} - Default İlişkisi")
    plt.show()
   

df = df.drop("LoanID",axis=1)
binary_cols = ["HasMortgage","HasDependents","HasCoSigner"]
binary_map = {"yes":1 , "no":0}
for col in binary_cols:
    df[col]=df[col].map(binary_map)

education_map = {
    "High School": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "PhD": 4
}
df["Education"] = df["Education"].map(education_map) 

nominal_cols = ['EmploymentType', 'MaritalStatus', 'LoanPurpose']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)


print("Sonuç DataFrame'inin ilk 5 satırı:")
print(df.head())
print("\nSonuç DataFrame'inin bilgi özeti:")
print(df.info())


from sklearn.model_selection import train_test_split

X = df.drop("Default", axis = 1)
y = df["Default"]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(" 5. Adım Tamamlandı: Train/Test Split")
print("---------------------------------------")
print(f"X (Öznitelikler) Boyutu: {X.shape}")
print(f"y (Hedef Değişken) Boyutu: {y.shape}")
print("---------------------------------------")
print(f"Eğitim Kümesi (X_train) Boyutu: {X_train.shape}")
print(f"Test Kümesi (X_test) Boyutu: {X_test.shape}")
print("---------------------------------------")
print("Hedef Değişken Oran Kontrolü (Stratified Sampling):")
print(f"Tüm Veri Oranı:\n{y.value_counts(normalize=True).round(4)}")
print(f"Eğitim Verisi Oranı:\n{y_train.value_counts(normalize=True).round(4)}")
print(f"Test Verisi Oranı:\n{y_test.value_counts(normalize=True).round(4)}")


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import(classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)

Log_model_tuned  = LogisticRegression(random_state=42 , solver="liblinear" , class_weight="balanced")
Log_model_tuned.fit(X_train , y_train)

y_pred_tuned = Log_model_tuned.predict(X_test)
y_pred_proba_tuned = Log_model_tuned.predict_proba(X_test)[:,1]

print("-" * 50)
print("8. Adım: Hyperparameter Tuning Sonuçları (class_weight='balanced')")

print("\nSınıflandırma Raporu:")
print(classification_report(y_test , y_pred_tuned))

roc_auc_tuned = roc_auc_score(y_test , y_pred_proba_tuned)
print(f"\nROC AUC Skoru: {roc_auc_tuned:.4f}")

param_grid = {
    "C":[0.001,0.01,0.1,1,10,100],
    "penalty":["l1","l2"]
}

lr = LogisticRegression(random_state=42 , solver="liblinear", class_weight="balanced")

grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv = 5,
    verbose = 1,
    n_jobs=-1
)

grid_search.fit(X_train,y_train)

best_lr = grid_search.best_estimator_

print("-" * 50)
print("9. Adım: GridSearchCV Sonuçları")
print(f"En İyi Parametreler: {grid_search.best_params_}")
print(f"En İyi Çapraz Doğrulama AUC Skoru: {grid_search.best_score_:.4f}")

y_pred_final = best_lr.predict(X_test)
y_proba_final = best_lr.predict_proba(X_test)[:,1]

print("\nNihai Test Kümesi Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_final))

roc_auc_final = roc_auc_score(y_test, y_proba_final)
print(f"\nNihai ROC AUC Skoru: {roc_auc_final:.4f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("-" * 50)
print("9. Adım: Gelişmiş Model Eğitimi (Random Forest)")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(X_train,y_train)

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:,1]

print("\nRandom Forest Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_rf))

roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
print(f"\nRandom Forest ROC AUC Skoru: {roc_auc_rf:.4f}")


cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nKarışıklık Matrisi:")
print(cm_rf)

feature_importances = rf_model.feature_importances_

feature_names = X_train.columns 

importance_df = pd.DataFrame(
    {
        "feature":feature_names,
        "importance":feature_importances
    }
).sort_values(by = "importance",ascending=False)

print("-" * 50)
print("10. Adım: Random Forest Öznitelik Önem Düzeyleri")
print(importance_df.head(10))

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=importance_df.head(10), palette="viridis")
plt.title('Random Forest - En Önemli 10 Öznitelik')
plt.xlabel('Önem Düzeyi')
plt.ylabel('Öznitelik')
plt.savefig('random_forest_feature_importance.png')
plt.show()

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
df = df.fillna(0)

numerical_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                  'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education']

scaler = StandardScaler()
X_train[numerical_cols]=scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols]=scaler.fit_transform(X_test[numerical_cols])

print("-" * 50)
print("11. Adım: Nihai Model Kurulumu (XGBoost)")

neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight = neg_count / pos_count

print(f"sınıf oranı (negatif/pozitif: {scale_pos_weight}")

xgb_model = xgb.XGBClassifier(
    objective = "binary:logistic",
    n_estimators = 100,
    random_state=42,
    scale_pos_weight = scale_pos_weight,
    use_label_encoder = False,
    eval_metric = "logloss"
)
xgb_model.fit(X_train ,y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb= xgb_model.predict_proba(X_test)[:,1]

print("\nXGBoost Sınıflandırma Raporu:")
print(classification_report(y_test , y_pred_xgb))

roc_auc_xgb = roc_auc_score(y_test , y_pred_xgb)
print(f"\nXGBoost ROC AUC Skoru: {roc_auc_xgb:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.linear_model import LogisticRegression


log_model_tuned = LogisticRegression(random_state=42, solver="liblinear", class_weight="balanced")
log_model_tuned.fit(X_train, y_train)
y_pred_proba_tuned = log_model_tuned.predict_proba(X_test)[:, 1]


precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_tuned)


fscore = (2 * precision * recall) / (precision + recall)
ix = np.argmax(fscore)
best_f1_threshold = thresholds[ix]


ix_70 = np.where(recall >= 0.70)[0][::-1][0] 
best_recall_threshold = thresholds[ix_70]


print("-" * 50)
print(" Nihai Analiz: Karar Eşiği Ayarı (Tuned LR Modeli)")


y_pred_f1 = (y_pred_proba_tuned >= best_f1_threshold).astype(int)
print(f"1. F1-Skoru'nu Maksimize Eden Eşik: {best_f1_threshold:.4f}")
print("F1-Skoru Maksimizasyon Sonucu:")
print(classification_report(y_test, y_pred_f1))


y_pred_recall = (y_pred_proba_tuned >= best_recall_threshold).astype(int)
print(f"2. Recall >= 0.70 Hedefine Ulaşan Eşik: {best_recall_threshold:.4f}")
print("Recall Hedefleme Sonucu:")
print(classification_report(y_test, y_pred_recall))


plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.scatter(best_f1_threshold, precision[ix], marker='o', color='red', label=f'Best F1 Threshold: {best_f1_threshold:.4f}')
plt.title('Precision ve Recall Eşiğe Bağlı Değişimi')
plt.xlabel('Karar Eşiği (Threshold)')
plt.ylabel('Değer')
plt.legend()
plt.savefig('precision_recall_threshold_tuning.png')
plt.show()