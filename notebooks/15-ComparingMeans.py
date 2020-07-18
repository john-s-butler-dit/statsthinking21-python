# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Comparing means

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# %%
# load the NHANES data library
from nhanes.load import load_NHANES_data
nhanes_data = load_NHANES_data()
adult_nhanes_data = nhanes_data.query('AgeInYearsAtScreening > 17')
## Need to remove duplicates
#nhanes_data=nhanes_data.drop_duplicates(subset ="ID",  keep = 'first', inplace = True)


# %% [markdown]
# We have already encountered a number of cases where we wanted to ask questions about the mean of a sample.  
# In this chapter, we will delve deeper into the various ways that we can compare means.
#
#
# ## Testing the value of a single mean {#single-mean}
#
# The simplest question we might want to ask of a mean is whether it has a specific value.  Let's say that we want to test whether the mean BMI value in adults from the NHANES dataset is above 25, which is the lower cutoff for being overweight according to the US Centers for Disease Control.  We take a sample of 200 adults in order to ask this question.

# %%
adult_nhanes_data_na=adult_nhanes_data.dropna(subset=['BodyMassIndexKgm2']) # Removing NA's
adult_nhames_sample=adult_nhanes_data_na.sample(n=200, random_state=1)
fig, ax= plt.subplots(1, 1, figsize=(6,4))
ax=adult_nhames_sample['BodyMassIndexKgm2'].hist()
ax.axvline(25,color='g',linewidth=4)

adult_nhames_sample['BMIdiff']=adult_nhames_sample['BodyMassIndexKgm2']>25

# %% [markdown]
# One simple way to test for this difference is using a test called the *sign test*, 
# which asks whether the proportion of positive differences between the actual value and the hypothesized value is different than what we would expect by chance.  To do this, we take the differences between each data point and the hypothesized mean value and compute their sign.  In our sample, we see that `r I(sprintf('%0.1f',mean(NHANES_sample$BMIdiff)*100))` percent of individuals have a BMI greater than 25.  We can then use a binomial test to ask whether this proportion of positive differences is greater than 0.5, using the `binom_test()` function in Python:

# %%
# compute sign test for differences between first and second measurement
npos = np.sum(adult_nhames_sample['BMIdiff'])
bt=stats.binom_test(npos, n=adult_nhames_sample.shape[0], p=0.5, alternative='greater')
print(bt)

# %% [markdown]
# Here we see that the proportion of individuals with positive signs would be *very* surprising under the null hypothesis of $p=0.5$. 
#
# We can also ask this question using Student's t-test, which you have already encountered earlier in the book.  We will refer to the mean as $\bar{X}$ and the hypothesized population mean as $\mu$.  Then, the t test for a single mean is:
#
# $$
# t = \frac{\bar{X} - \mu}{SEM}
# $$
# where SEM (as you may remember from the chapter on sampling) is defined as:
#
# $$
# SEM = \frac{\hat{\sigma}}{\sqrt{n}}
# $$
#
# In essence, the t statistic asks how large the deviation of the sample mean from the hypothesized quantity is with respect to the sampling variability of the mean.
#
# We can compute this for the NHANES dataset using the ```t.ttest_1samp()``` function in Python:

# %%
stats.ttest_1samp(adult_nhames_sample['BodyMassIndexKgm2'],5.0)
a=3

# %% [markdown]
# This shows us that the mean BMI in the dataset  is significantly larger than the cutoff for overweight.
#
# ## Comparing two means {#comparing-two-means}
#
# A more common question that often arises in statistics is whether there is a difference between the means of two different groups.  Let's say that we would like to know whether regular marijuana smokers watch more television.  We can ask this question using the NHANES dataset; let's take a sample of 200 individuals from the dataset and test whether the number of hours of television watching per day is related to regular marijuana use.  The left panel of Figure \@ref(fig:PotTVViolin) shows these data using a violin plot.

# %%
# In the R code it was TVHrsDay

adult_nhanes_data=adult_nhanes_data.dropna(subset=['EverUsedMarijuanaOrHashish','SleepHoursWeekdaysOrWorkdays']) # Removing NA's
adult_nhames_sample=adult_nhanes_data.sample(n=200, random_state=0)
adult_nhames_sample['GenderNum']=adult_nhames_sample['Gender']=='Female'
adult_nhames_sample=adult_nhames_sample[['GenderNum','Gender','SleepHoursWeekdaysOrWorkdays','EverUsedMarijuanaOrHashish']]
ax = sns.violinplot(x=adult_nhames_sample['Gender'],y=adult_nhames_sample['SleepHoursWeekdaysOrWorkdays'])

# %%
Group1 = adult_nhames_sample.loc[adult_nhames_sample['Gender']=='Female']
Group2 = adult_nhames_sample.loc[adult_nhames_sample['Gender']=='Male']

#Group1 =adult_nhames_sample.query('GenderNum==True').sample(200, random_state=2)
#Group2 =adult_nhames_sample.query('Gender==Male').sample(200, random_state=2)


# clean up smoking variables
#adult_nhanes_data.loc[adult_nhanes_data['SmokedAtLeast100CigarettesInLife'] == 0, 'DoYouNowSmokeCigarettes'] = 'Not at all'
#adult_nhanes_data.loc[:, 'SmokeNow'] = adult_nhanes_data['DoYouNowSmokeCigarettes'] != 'Not at all'

# Create average alcohol consumption variable between the two dietary recalls
#adult_nhanes_data.loc[:, 'AvgAlcohol'] = adult_nhanes_data[['AlcoholGm_DR1TOT', 'AlcoholGm_DR2TOT']].mean(1)
#adult_nhanes_data = adult_nhanes_data.dropna(subset=['AvgAlcohol'])

#sample_size_per_group = 150

stats.ttest_ind(Group1['SleepHoursWeekdaysOrWorkdays'], Group2['SleepHoursWeekdaysOrWorkdays'])

# %% [markdown]
# We can also use Student's t test to test for differences between two groups of independent observations (as we saw in an earlier chapter); we will turn later in the chapter to cases where the observations are not independent.  As a reminder, the t-statistic for comparison of two independent groups is computed as:
#
# $$
# t = \frac{\bar{X_1} - \bar{X_2}}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}}
# $$
#
# where $\bar{X}_1$ and $\bar{X}_2$ are the means of the two groups, $S^2_1$ and $S^2_2$ are the variances for each of the groups, and $n_1$ and $n_2$ are the sizes of the two groups.  Under the null hypothesis of no difference between means, this statistic is distributed according to a t distribution with n-2 degrees of freedom (since we have computed two parameter estimates, namely the means of the two groups).  We can compute the independent t-test in Python using the ```ttest_ind()``` function. In this case, we started with the specific hypothesis that smoking marijuana is associated with greater TV watching, so we will use a one-tailed test.  Since the t.test function orders the conditions alphabetically, the "No" group comes first, and thus we need to test the alternative hypothesis of whether the first group is less than the second ("Yes") group; for this reason, we specify 'less' as our alternative.

# %%
#```{r echo=FALSE,warning=FALSE}
# compute t test for tv watching as function of marijuana use
#ttresult
#```

# %% [markdown]
# In this case we see that there is a statistically significant difference between groups, in the expected direction - regular pot smokers watch more TV.
#
# ## The t-test as a linear model {#ttest-linear-model}
#
# The t-test is often presented as a specialized tool for comparing means, but it can also be viewed as an application of the general linear model.  In this case, the model would look like this:
#
# $$
# \hat{TV} = \hat{\beta_1}*Marijuana + \hat{\beta_0}
# $$
# However, smoking is a binary variable, so we treat it as a *dummy variable* like we discussed in the previous chapter, setting it to a value of 1 for smokers and zero for nonsmokers.  In that case, $\hat{\beta_1}$ is simply the difference in means between the two groups, and $\hat{\beta_0}$ is the mean for the group that was coded as zero.  We can fit this model using the ```lm()``` function, and see that it gives the same t statistic as the t-test above:

# %%
from statsmodels.formula.api import ols
ols_model = ols(formula='SleepHoursWeekdaysOrWorkdays~ GenderNum + 1', data=adult_nhames_sample)
ols_result = ols_model.fit()
ols_result.summary()

# %% [markdown]
# We can also view the linear model results graphically (see the right panel of Figure \@ref(fig:PotTVViolin)).  In this case, the predicted value for nonsmokers is $\hat{\beta_0}$ (`r I(lm_summary$coefficients[1,1])`) and the predicted value for smokers is $\hat{\beta_0} +\hat{\beta_1}$ (`r I(lm_summary$coefficients[1,1] + lm_summary$coefficients[2,1])`).  
#
# To compute the standard errors for this analysis, we can use exactly the same equations that we used for linear regression -- since this really is just another example of linear regression.  In fact, if you compare the p-value from the t-test above with the p-value in the linear regression analysis for the marijuana use variable, you will see that the one from the linear regression analysis is exactly twice the one from the t-test, because the linear regression analysis is performing a two-tailed test.  
#
# ### Effect sizes for comparing two means
#
# The most commonly used effect size for a comparison between two means is Cohen's d, which (as you may remember from Chapter \@ref(ci-effect-size-power)) is an expression of the effect in terms of standard error units.  For the t-test estimated using the general linear model outlined above (i.e. with a single dummy-coded variable), this is expressed as:
#
# $$
# d = \frac{\hat{beta_1}}{SE_{residual}}
# $$
# We can obtain these values from the analysis output above, giving us a d = `r I(lm_summary$coefficients[2,1]/lm_summary$sigma)`, which we would generally interpret as a medium sized effect.
#
# We can also compute $R^2$ for this analysis, which tells us how much variance in TV watching is accounted for.  This value (which is reported in the summary of the lm() analysis) is `r I(lm_summary$r.squared)`, which tells us that while the effect may be statistically significant, it accounts for relatively little of the variance in TV watching.
#
# ## Bayes factor for mean differences
#
# As we discussed in the chapter on Bayesian analysis, Bayes factors provide a way to better quantify evidence in favor or against the null hypothesis of no difference.  In this case, we want to specifically test against the null hypothesis that the difference is greater than zero - because the difference is computed by the function between the first group ('No') and the second group ('Yes'). Thus, we specify a "null interval" going from zero to infinity, which means that the alternative is less than zero.

# %%
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()
#import rpy2.robjects.packages as rpackages

# %%
# import the BayesFactor package
BayesFactor = importr('BayesFactor')

# %%
# import the data frames into the R workspace
robjects.globalenv["Group1"] = Group1
robjects.globalenv["Group2"] = Group2

# %%
# perform the standard t-test
ttest_output = r('print(t.test(Group1$SleepHoursWeekdaysOrWorkdays, Group2$SleepHoursWeekdaysOrWorkdays, alternative="greater"))')

# compute the Bayes factor
r('bf = ttestBF(y=Group1$SleepHoursWeekdaysOrWorkdays, x=Group2$SleepHoursWeekdaysOrWorkdays, nullInterval = c(0, Inf))')
r('summary(bf)')


# %% [markdown]
# This shows us that the evidence against the null hypothesis is moderately strong.
#
# ## Comparing paired observations {#paired-ttests}

# %%
adult_nhanes_data_na=adult_nhanes_data.dropna(subset=['SystolicBloodPres1StRdgMmHg','SystolicBloodPres2NdRdgMmHg']) # Removing NA's
adult_nhames_sample=adult_nhanes_data_na.sample(n=200, random_state=50)
BloodPressure=adult_nhames_sample[['SystolicBloodPres1StRdgMmHg','SystolicBloodPres2NdRdgMmHg']]
BloodPressure.loc[:,'ID']=np.arange(0,200)
BP=BloodPressure.melt(id_vars=['ID'],value_vars = ['SystolicBloodPres1StRdgMmHg','SystolicBloodPres2NdRdgMmHg'],var_name='Order',value_name='Pressure')
BP.head()

# %%
ax = sns.violinplot(x=BP['Order'],y=BP['Pressure'])
ax = sns.swarmplot(x="Order", y="Pressure", data=BP, color=".25")

# %%
BloodPressure.loc[:,'BPSdiff']=BloodPressure.loc[:,'SystolicBloodPres1StRdgMmHg']-BloodPressure.loc[:,'SystolicBloodPres2NdRdgMmHg']
#BloodPressure['meanBPS']=(BloodPressure['SystolicBloodPres1StRdgMmHg']+BloodPressure['SystolicBloodPres2NdRdgMmHg'])/2
BloodPressure.loc[:,'meanBPS']=adult_nhanes_data[['SystolicBloodPres1StRdgMmHg', 'SystolicBloodPres2NdRdgMmHg']].mean(1)
BloodPressure.loc[:,'diffPos']=BloodPressure.loc[:,'BPSdiff']>0

# %% [markdown]
# In experimental research, we often use *within-subjects* designs, in which we compare the same person on multiple measurements.  The measurement that come from this kind of design are often referred to as *repeated measures*. For example, in the NHANES dataset blood pressure was measured three times. Let's say that we are interested in testing whether there is a difference in mean blood pressure between the first and second measurement (Figure  \@ref(fig:BPfig)). We see that there does not seem to be much of a difference in mean blood pressure between time points (about one point). First let's test for a difference using an independent samples t-test, which ignores the fact that pairs of data points come from the the same individuals.   

# %%
stats.ttest_ind(BloodPressure['SystolicBloodPres1StRdgMmHg'], BloodPressure['SystolicBloodPres2NdRdgMmHg'])

# %% [markdown]
# This analysis shows no significant difference. However, this analysis is inappropriate since it assumes that the two samples are independent, when in fact they are not, since the data come from the same individuals.  We can plot the data with a line for each individual to show this (see Figure \@ref(fig:BPLinePlot)).
#
# In this analysis, what we really care about is whether the blood pressure for each person changed in a systematic way between the two measurements, so another way to represent the data is to compute the difference between the two timepoints for each individual, and then analyze these difference scores rather than analyzing the individual measurements. In Figure \@ref(fig:BPDiffHist), we show a histogram of these difference scores, with a blue line denoting the mean difference.

# %%
ax=BloodPressure['BPSdiff'].hist(bins=20)
ax.axvline(BloodPressure['BPSdiff'].mean(),color='b',linewidth=4)
plt.show()

# %% [markdown]
# ### Sign test
#
# One simple way to test for differences is using the *sign test*. To do this, we take the differences and compute their sign, and then we use a binomial test to ask whether the proportion of positive signs differs from 0.5.

# %%
npos = np.sum(BloodPressure['diffPos'])
bt=stats.binom_test(npos, n=BloodPressure.shape[0], p=0.5, alternative='greater')
print(bt)

# %% [markdown]
# Here we see that the proportion of individuals with positive signs (`r I(bt$estimate)`) is not large enough to be surprising under the null hypothesis of $p=0.5$. However, one problem with the sign test is that it is throwing away information about the magnitude of the differences, and thus might be missing something.
#
# ### Paired t-test
# A more common strategy is to use a *paired t-test*. We can compute this using the ```ttest_rel()``` function in Python.  

# %%
stats.ttest_rel(BloodPressure['SystolicBloodPres1StRdgMmHg'], BloodPressure['SystolicBloodPres2NdRdgMmHg'])

# %% [markdown]
# This is equivalent to a one-sample t-test ```ttest_1samp()``` for whether the mean difference between the measurements is zero.  

# %%
stats.ttest_1samp(BloodPressure['BPSdiff'],popmean=0)

# %% [markdown]
# With this analyses we see that there is in fact a significant difference between the two measurements. Let's compute the Bayes factor to see how much evidence is provided by the result:

# %%
robjects.globalenv["BloodPressure"] = BloodPressure
# compute the Bayes factor
r('bf = ttestBF(x=BloodPressure$SystolicBloodPres1StRdgMmHg, y=BloodPressure$SystolicBloodPres2NdRdgMmHg, paired = TRUE, nullInterval = c(0, Inf))')
r('summary(bf)')

# %% [markdown]
# This shows us that although the effect was significant in a paired t-test, it actually provides very little evidence in favor of the alternative hypothesis. 
#
# The paired t-test can also be defined in terms of a linear model; see the Appendix for more details on this.
#
# ## Comparing more than two means
#
# Often we want to compare more than two means to determine whether any of them differ from one another.  Let's say that we are analyzing data from a clinical trial for the treatment of high blood pressure.  In the study, volunteers are randomized to one of three conditions: Drug 1, Drug 2 or placebo.  Let's generate some data and plot them (see Figure \@ref(fig:DrugTrial))

# %%
nPerGroup= 36


# %%
np.random.seed(0)
nPerGroup= 36
noiseSD = 10
meanSysBP = 140
effectSize = 0.8

group=['placebo']*nPerGroup+['drug1']*nPerGroup+['drug2']*nPerGroup
sysBP1=np.random.normal(meanSysBP, noiseSD, nPerGroup)
sysBP2=(np.random.normal(meanSysBP-noiseSD*effectSize, noiseSD, nPerGroup))
sysBP3=(np.random.normal(meanSysBP, noiseSD, nPerGroup))
sysBP=[[sysBP1],[sysBP2],[sysBP3]]
sysBP=np.concatenate((sysBP1,sysBP2,sysBP3), axis=0)
df=pd.DataFrame({'group': group,'sysBP':sysBP})
ax = sns.boxplot(x="group", y="sysBP", data=df)

# %% [markdown]
# ### Analysis of variance {#ANOVA}
#
# We would first like to test the null hypothesis that the means of all of the groups are equal -- that is, neither of the treatments had any effect. We can do this using a method called *analysis of variance* (ANOVA). This is one of the most commonly used methods in psychological statistics, and we will only scratch the surface here.  The basic idea behind ANOVA is one that we already discussed in the chapter on the general linear model, and in fact ANOVA is just a name for a specific implementation of such a model.
#
# Remember from the last chapter that we can partition the total variance in the data ($SS_{total}$) into the variance that is explained by the model ($SS_{model}$) and the variance that is not ($SS_{error}$).  We can then compute a *mean square* for each of these by dividing them by their degrees of freedom; for the error this is $N - p$ (where $p$ is the number of means that we have computed), and for the model this is $p - 1$:
#
# $$
# MS_{model} =\frac{SS_{model}}{df_{model}}= \frac{SS_{model}}{p-1}
# $$
#
# $$
# MS_{error} = \frac{SS_{error}}{df_{error}} = \frac{SS_{error}}{N - p}
# $$
#
# With ANOVA, we want to test whether the variance accounted for by the model is greater than what we would expect by chance, under the null hypothesis of no differences between means.  Whereas for the t distribution the expected value is zero under the null hypothesis, that's not the case here, since sums of squares are always positive numbers.  Fortunately, there is another standard distribution that describes how ratios of sums of squares are distributed under the null hypothesis: The *F* distribution (see figure \@ref(fig:FDist)). This distribution has two degrees of freedom, which correspond to the degrees of freedom for the numerator (which in this case is the model), and the denominator (which in this case is the error).

# %%
from scipy.stats import f
x=np.arange(0.1,10,0.1)
ax=sns.lineplot(x=x,y=f.pdf(x, 1, 1))
ax=sns.lineplot(x=x,y=f.pdf(x, 50, 1))
ax=sns.lineplot(x=x,y=f.pdf(x, 50, 10))
ax.set(xlabel='F Values', ylabel='Density')
plt.show()

# %% [markdown]
# To create an ANOVA model, we extend the idea of *dummy coding* that you encountered in the last chapter. Remember that for the t-test comparing two means, we created a single dummy variable that took the value of 1 for one of the conditions and zero for the others.  Here we extend that idea by creating two dummy variables, one that codes for the Drug 1 condition and the other that codes for the Drug 2 condition.  Just as in the t-test, we will have one condition (in this case, placebo) that doesn't have a dummy variable, and thus represents the baseline against which the others are compared; its mean defines the intercept of the model. Let's create the dummy coding for drugs 1 and 2.

# %%
#```{r echo=FALSE}
# create dummy variables for drug1 and drug2
#df <-
#  df %>%
#  mutate(
#    d1 = as.integer(group == "drug1"), # 1s for drug1, 0s for all other drugs
#    d2 = as.integer(group == "drug2")  # 1s for drug2, 0s for all other drugs
#  )
#```

# %% [markdown]
# Now we can fit a model using the same approach that we used in the previous chapter:

# %%

import statsmodels.api as sm
ols_model = ols(formula='sysBP~ group', data=df)
ols_result = ols_model.fit()
aov_table = sm.stats.anova_lm(ols_result)
aov_table 

# %%
#```{r echo=FALSE}
# fit ANOVA model
#lmResultANOVA <- lm(sysBP ~ d1 + d2, data = df)
#summary(lmResultANOVA)
#```

ols_model = ols(formula='sysBP~ group + 1', data=df)
ols_result = ols_model.fit()
ols_result.summary()

# %% [markdown]
# The output from this command provides us with two things.  First, it shows us the result of a t-test for each of the dummy variables, which basically tell us whether each of the conditions separately differs from placebo; it appears that Drug 1 does whereas Drug 2 does not.  However, keep in mind that if we wanted to interpret these tests, we would need to correct the p-values to account for the fact that we have done multiple hypothesis tests; we will see an example of how to do this in the next chapter.
#
# Remember that the hypothesis that we started out wanting to test was whether there was any difference between any of the conditions; we refer to this as an *omnibus* hypothesis test, and it is the test that is provided by the F statistic. The F statistic basically tells us whether our model is better than a simple model that just includes an intercept.  In this case we see that the F test is highly significant, consistent with our impression that there did seem to be differences between the groups (which in fact we know there were, because we created the data).

# %%
```{r echo=FALSE}
# Add section on post-hoc tests using emmeans
```

# %% [markdown]
# ## Learning objectives
#
# After reading this chapter, you should be able to:
#
# * Describe the rationale behind the sign test
# * Describe how the t-test can be used to compare a single mean to a hypothesized value
# * Compare the means for two paired or unpaired groups using a two-sample t-test
#
#
# ## Appendix
#
# ### The paired t-test as a linear model
#
# We can also define the paired t-test in terms of a general linear model.  To do this, we include all of the measurements for each subject as data points (within a tidy data frame).  We then include in the model a variable that codes for the identity of each individual (in this case, the ID variable that contains a subject ID for each person). This is known as a *mixed model*, since it includes effects of independent variables as well as effects of individuals.  The standard model fitting procedure ```lm()``` can't do this, but we can do it using the ```lmer()``` function from a popular R package called *lme4*, which is specialized for estimating mixed models.  The ```(1|ID)``` in the formula tells `lmer()` to estimate a separate intercept (which is what the ```1``` refers to) for each value of the ```ID``` variable (i.e. for each individual in the dataset), and then estimate a common slope relating timepoint to BP.

# %%
#```{r,messages=FALSE}
# compute mixed model for paired test
#lmrResult <- lmer(BPsys ~ timepoint + (1 | ID), 
#                  data = NHANES_sample_tidy)
#summary(lmrResult)
#```

# %% [markdown]
# You can see that this shows us a p-value that is very close to the result from the paired t-test computed using the ```t.test()``` function.

# %%
