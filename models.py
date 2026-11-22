import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    PolynomialFeatures
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score


def create_encoder():

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def get_feature_types(X):

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    return numeric_features, categorical_features


def create_preprocessor(
        X,
        scale=True
):

    numeric_features, categorical_features = (
        get_feature_types(X)
    )

    transformers = []

    # Numeric preprocessing

    if numeric_features:

        numeric_steps = [
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]

        if scale:
            numeric_steps.append(
                (
                    "scaler",
                    StandardScaler()
                )
            )

        numeric_pipeline = Pipeline(
            numeric_steps
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    # Categorical preprocessing

    if categorical_features:

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    create_encoder()
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def select_simple_linear_feature(
        X,
        y
):
    """
    Selects the numerical feature having the
    strongest absolute correlation with target.
    """

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_features:
        return None

    correlation_df = X[numeric_features].copy()

    correlation_df["target"] = y.values

    correlations = (
        correlation_df.corr()["target"]
        .drop("target")
        .abs()
        .sort_values(ascending=False)
    )

    if len(correlations) == 0:
        return None

    return correlations.index[0]


def train_selected_models(
        df,
        target,
        selected_models,
        output_dir
):
    """
    Trains all selected regression models.
    """

    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(columns=[target])
    y = df[target]

    if X.shape[1] == 0:
        raise ValueError(
            "No predictor columns available."
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    results = {}

    # -------------------------------------------------
    # SIMPLE LINEAR REGRESSION
    # -------------------------------------------------

    if "Simple Linear Regression" in selected_models:

        feature = select_simple_linear_feature(
            X_train,
            y_train
        )

        if feature is None:
            raise ValueError(
                "Simple Linear Regression requires "
                "at least one numerical predictor."
            )

        simple_X_train = X_train[[feature]]
        simple_X_test = X_test[[feature]]

        preprocessor = create_preprocessor(
            simple_X_train,
            scale=True
        )

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LinearRegression()
                )
            ]
        )

        pipeline.fit(
            simple_X_train,
            y_train
        )

        train_pred = pipeline.predict(
            simple_X_train
        )

        test_pred = pipeline.predict(
            simple_X_test
        )

        results["Simple Linear Regression"] = {
            "model": pipeline,
            "train_r2": r2_score(
                y_train,
                train_pred
            ),
            "test_r2": r2_score(
                y_test,
                test_pred
            ),
            "y_test": y_test,
            "predictions": test_pred,
            "feature": feature
        }

    # -------------------------------------------------
    # MULTIPLE LINEAR REGRESSION
    # -------------------------------------------------

    if "Multiple Linear Regression" in selected_models:

        preprocessor = create_preprocessor(
            X_train,
            scale=True
        )

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LinearRegression()
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        results["Multiple Linear Regression"] = {
            "model": pipeline,
            "train_r2": r2_score(
                y_train,
                train_pred
            ),
            "test_r2": r2_score(
                y_test,
                test_pred
            ),
            "y_test": y_test,
            "predictions": test_pred
        }

    # -------------------------------------------------
    # POLYNOMIAL REGRESSION
    # -------------------------------------------------

    if "Polynomial Regression" in selected_models:

        numeric_features = X_train.select_dtypes(
            include=np.number
        ).columns.tolist()

        if not numeric_features:
            raise ValueError(
                "Polynomial Regression requires "
                "numerical predictor columns."
            )

        # Select top features to prevent huge feature expansion

        correlation_df = X_train[numeric_features].copy()
        correlation_df["target"] = y_train.values

        correlations = (
            correlation_df.corr()["target"]
            .drop("target")
            .abs()
            .sort_values(ascending=False)
        )

        poly_features = correlations.head(4).index.tolist()

        poly_preprocessor = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "polynomial",
                    PolynomialFeatures(
                        degree=2,
                        include_bias=False
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        [
                            (
                                "poly",
                                poly_preprocessor,
                                poly_features
                            )
                        ],
                        remainder="drop"
                    )
                ),
                (
                    "model",
                    LinearRegression()
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        results["Polynomial Regression"] = {
            "model": pipeline,
            "train_r2": r2_score(
                y_train,
                train_pred
            ),
            "test_r2": r2_score(
                y_test,
                test_pred
            ),
            "y_test": y_test,
            "predictions": test_pred,
            "features": poly_features
        }

    # -------------------------------------------------
    # RIDGE REGRESSION
    # -------------------------------------------------

    if "Ridge Regression" in selected_models:

        preprocessor = create_preprocessor(
            X_train,
            scale=True
        )

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    Ridge(alpha=1.0)
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        results["Ridge Regression"] = {
            "model": pipeline,
            "train_r2": r2_score(
                y_train,
                train_pred
            ),
            "test_r2": r2_score(
                y_test,
                test_pred
            ),
            "y_test": y_test,
            "predictions": test_pred
        }

    # -------------------------------------------------
    # LASSO REGRESSION
    # -------------------------------------------------

    if "Lasso Regression" in selected_models:

        preprocessor = create_preprocessor(
            X_train,
            scale=True
        )

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    Lasso(
                        alpha=0.01,
                        max_iter=10000
                    )
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        results["Lasso Regression"] = {
            "model": pipeline,
            "train_r2": r2_score(
                y_train,
                train_pred
            ),
            "test_r2": r2_score(
                y_test,
                test_pred
            ),
            "y_test": y_test,
            "predictions": test_pred
        }

    # -------------------------------------------------
    # DECISION TREE REGRESSION
    # -------------------------------------------------

    if "Decision Tree Regression" in selected_models:

        preprocessor = create_preprocessor(
            X_train,
            scale=False
        )

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=5,
                        random_state=42
                    )
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        results["Decision Tree Regression"] = {
            "model": pipeline,
            "train_r2": r2_score(
                y_train,
                train_pred
            ),
            "test_r2": r2_score(
                y_test,
                test_pred
            ),
            "y_test": y_test,
            "predictions": test_pred
        }

    # -------------------------------------------------
    # RANDOM FOREST REGRESSION
    # -------------------------------------------------

    if "Random Forest Regression" in selected_models:

        preprocessor = create_preprocessor(
            X_train,
            scale=False
        )

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        results["Random Forest Regression"] = {
            "model": pipeline,
            "train_r2": r2_score(
                y_train,
                train_pred
            ),
            "test_r2": r2_score(
                y_test,
                test_pred
            ),
            "y_test": y_test,
            "predictions": test_pred
        }

    if not results:
        raise ValueError(
            "Please select at least one regression model."
        )

    # -------------------------------------------------
    # Create comparison dataframe
    # -------------------------------------------------

    comparison = []

    for model_name, result in results.items():

        comparison.append(
            {
                "Model": model_name,
                "Train R2": round(
                    result["train_r2"],
                    4
                ),
                "Test R2": round(
                    result["test_r2"],
                    4
                ),
                "R2 Gap": round(
                    result["train_r2"]
                    - result["test_r2"],
                    4
                )
            }
        )

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df.sort_values(
        by="Test R2",
        ascending=False,
        inplace=True
    )

    comparison_path = os.path.join(
        output_dir,
        "model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    return results, comparison_df, comparison_path