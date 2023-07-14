from src.npc_classifier.npc_classifier import NPCClassifier


def test_npc_classifier():
    X_train = ["text1", "text2", "text3", "text4"]
    y_train = [0, 0, 1, 1]
    X_test = ["text1", "text3"]

    classifier = NPCClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    assert len(y_pred) == len(X_test)
    assert all(isinstance(y, int) for y in y_pred)
