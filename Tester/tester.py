

def test(model, data):
    X_test, Y_true = data.X_test[0], data.Y_test[0]  # (in, batch), (out, batch)
    Y_pred = model.get_output(X_test)  # (out, batch)

    loss = model.loss.get_avg_loss(Y_pred, Y_true)

    print(f"{model.name} loss : {loss}")

    return Y_pred
