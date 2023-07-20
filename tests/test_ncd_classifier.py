from ncd_classifier import NCDClassifier, COMPRESSORS
from sklearn.metrics import accuracy_score


def get_data():
    X_train = ["hello world", "hello", "world", "what's up?"]
    y_train = [0, 0, 0, 1]
    X_test = ["hello", "world"]
    y_text = [0, 0]
    return X_train, y_train, X_test, y_text


def get_list_int_data():
    X_train = [[1, 2, 3], [1, 2], [1, 2, 3, 4], [100, 200, 300, 400, 500]]
    y_train = [0, 0, 0, 1]
    X_test = [[1, 2], [1, 2, 3, 4]]
    y_text = [0, 0]
    return X_train, y_train, X_test, y_text


def test_ncd_classifier():
    X_train, y_train, X_test, y_text = get_data()

    classifier = NCDClassifier(n_jobs=2, k=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    assert accuracy_score(y_text, y_pred) == 1

    assert len(y_pred) == len(X_test)
    assert all(isinstance(y, int) for y in y_pred)


def test_ncd_classifier_list_int():
    X_train, y_train, X_test, y_text = get_list_int_data()

    classifier = NCDClassifier(n_jobs=2, k=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    assert accuracy_score(y_text, y_pred) == 1

    assert len(y_pred) == len(X_test)
    assert all(isinstance(y, int) for y in y_pred)


def test_ncd_classifier_all_comporessors():
    X_train, y_train, X_test, y_text = get_data()

    for name, fn in COMPRESSORS.items():
        classifier = NCDClassifier(n_jobs=2, k=3, compress_len_fn=fn)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        if name == "lzma":
            # skip
            continue
        assert accuracy_score(y_text, y_pred) == 1
        assert len(y_pred) == len(X_test)
        assert all(isinstance(y, int) for y in y_pred)
