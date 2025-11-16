import pandas as pd

def create_submissions(
    test_df,
    COL_ID,
    COL_TARGET,
    test_pred_log,
    test_pred_weighted,
    test_pred_stack,
    threshold: float = 0.5
):

    # SUBMISSION FOR LOGISTIC MODEL
    submission_log = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (test_pred_log >= threshold).astype(int)
    })
    submission_log.to_csv("submissionlog.csv", index=False)
    print(f"\nSubmission saved: submissionlog.csv")
    print(f"  Shape: {submission_log.shape}")

    # SUBMISSION FOR WEIGHTED ENSEMBLE MODEL
    submission_w = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (test_pred_weighted >= threshold).astype(int)
    })
    submission_w.to_csv("submissionw.csv", index=False)
    print(f"\nSubmission saved: submissionw.csv")
    print(f"  Shape: {submission_w.shape}")

    # SUBMISSION FOR META LEARNER MODEL
    submission_meta = pd.DataFrame({
        COL_ID: test_df[COL_ID],
        COL_TARGET: (test_pred_stack >= threshold).astype(int)
    })
    submission_meta.to_csv("submissionfinal.csv", index=False)
    print(f"\nSubmission saved: submissionfinal.csv")
    print(f"  Shape: {submissionfinal.shape}")

 

    return {
        "log": submission_log,
        "weighted": submission_w,
        "meta": submissionfinal
    }

