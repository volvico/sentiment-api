import numpy as np
from typing import List, Tuple, Dict

def top_tokens_by_coef(model, vect, k=10) -> Dict[str, List[Tuple[str, float]]]:
    """If the model is linear (e.g., LogisticRegression), use coef_ to get top +/- features.
    Returns dict with 'pos' and 'neg' lists of (token, weight).
    """
    if not hasattr(model, "coef_"):
        return {"pos": [], "neg": []}
    if hasattr(vect, "get_feature_names_out"):
        feats = vect.get_feature_names_out()
    else:
        feats = np.array(sorted(vect.vocabulary_, key=vect.vocabulary_.get))
    coef = model.coef_
    # handle binary vs multiclass (one-vs-rest)
    if coef.ndim == 2 and coef.shape[0] == 1:
        c = coef[0]
        top_pos_idx = np.argsort(c)[-k:][::-1]
        top_neg_idx = np.argsort(c)[:k]
        pos = [(feats[i], float(c[i])) for i in top_pos_idx]
        neg = [(feats[i], float(c[i])) for i in top_neg_idx]
    else:
        # multiclass: take max abs across classes
        c = np.max(np.abs(coef), axis=0)
        idx = np.argsort(c)[-k:][::-1]
        pos = [(feats[i], float(c[i])) for i in idx]
        neg = []
    return {"pos": pos, "neg": neg}
