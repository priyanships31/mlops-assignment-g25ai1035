from sklearn.tree import DecisionTreeRegressor

from misc import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model
)


def main():

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = DecisionTreeRegressor(
        random_state=42
    )

    model = train_model(
        model,
        X_train,
        y_train
    )

    mse, r2 = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("=" * 50)
    print("Decision Tree Regressor")
    print(f"Average Test MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()