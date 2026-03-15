from sklearn.metrics import f1_score

def compute_f1(preds, labels):
    preds = preds.argmax(axis=1)
    return f1_score(labels, preds)