from pipeline.predict import predict, predict_from_byte_input


def test_pred():
    y_pred = predict("../tests/trained_epoch50_trial.pt", "tests/prediction_test/test_embedding_esterase.p",
                     "tests/prediction_test/test_encoding_esterase.p", {})
    print(y_pred)


def test_pred_byte():
    encoding = open("../tests/prediction_test/test_encoding_esterase.p", "rb").read()
    embedding = open("../tests/prediction_test/test_embedding_esterase.p", "rb").read()
    y_pred = predict_from_byte_input("../tests/trained_epoch50_trial.pt", embedding, encoding, params={})
    print(y_pred)

if __name__ == '__main__':
    test_pred()