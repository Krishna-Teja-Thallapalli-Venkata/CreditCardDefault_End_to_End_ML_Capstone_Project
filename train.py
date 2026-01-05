import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/credit_card_default.csv")

# Normalize column names
df.columns = (
    df.columns
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(".", "_")
)

# Explicit target rename
df.rename(
    columns={"default_payment_next_month": "target"},
    inplace=True
)

# Fail fast if wrong
assert "target" in df.columns, f"Target column not found. Columns: {df.columns.tolist()}"

# Split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train["target"]
del df_train["target"]

# Vectorize
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train.to_dict(orient="records"))

# Model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=1,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save
joblib.dump((dv, model), "model.joblib")

print("Model trained and saved as model.joblib")