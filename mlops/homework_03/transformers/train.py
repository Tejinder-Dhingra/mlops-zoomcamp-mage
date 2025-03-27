from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    """
    Args:
        df: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        DictVectorizer and LinearRegressor model
    """
    dv = DictVectorizer()

    categorical = ["PULocationID", "DOLocationID"]
    train_dicts = df[categorical].to_dict(orient='records')

    X_train = dv.fit_transform(train_dicts)
    Y_train = df['duration'].values

    lr = LinearRegression()
    lr.fit(X_train, Y_train)

    print(lr.intercept_)


    return dv, lr
