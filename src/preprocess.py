from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def get_preprocessor():

    categorical_features = [
        "type"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    return preprocessor