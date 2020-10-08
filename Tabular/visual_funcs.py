from scipy.stats import chi2_contingency,ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def chisq_of_df_cols(df, c1, c2):
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    return(chi2_contingency(ctsum.fillna(0))[1])

def plot_binaries(data, binary_var):
    for i in binary_var:
        test = pd.DataFrame(data.groupby([i, "DEFAULT"]).count()['LOAN_AMNT']).unstack()
        test.columns = test.columns.droplevel()  # Drop `count` label.
        test = test.div(test.sum())
        # Null hypothesis: Assumes that there is no association between the two variables. Reject Null if p < .05
        test.T.plot(kind='bar', stacked=True, rot=1, figsize=(8, 8), colors=['g','r'],
               title=i + ": P-Value for Chi-Square Independence Test: " + str(chisq_of_df_cols(data, i, "DEFAULT")))

def plot_boxplot(data_pd, continuous):
    sub_pos = data_pd[data_pd["DEFAULT"] == 1]
    sub_neg = data_pd[data_pd["DEFAULT"] == 0]
    for i in continuous:
        p_value = ttest_ind(sub_pos[i], sub_neg[i], equal_var = False)[1]
        fig, axs = plt.subplots(nrows= 1, figsize=(13, 5))
        sns.boxplot(x = "DEFAULT", y = i, data = data_pd, palette="Set1")
        if p_value < .05:
            plt.title(i + "\n P value:" + str(p_value) + "\n The differences between the two groups are significant!" + "\n Default: mean/std.: " + str(sub_pos[i].describe()[1]) + "/" + str(sub_pos[i].describe()[2]) + "\n non-Default: mean/std.: " + str(sub_neg[i].describe()[1]) + "/" + str(sub_neg[i].describe()[2]))
        else:
            plt.title(i + "\n P value:" + str(p_value) + "\n collsion: mean/std.: " + str(sub_pos[i].describe()[1]) + "/" + str(sub_pos[i].describe()[2]) + "\n non-Default: mean/std.: " + str(sub_neg[i].describe()[1]) + "/" + str(sub_neg[i].describe()[2]))
