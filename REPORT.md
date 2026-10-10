# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Santiago Llorca
- **Student ID**: 19952 
- **Email**: sllorca.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

The data comes from the public Lending Club loan dataset on Kaggle
(`accepted_2007_to_2018Q4`, accepted loans only). Each row is one consumer
loan, with the borrower's application-time information and the final outcome
of the loan.

From the full file (~2M loans) I kept only 36-month loans issued between
January 2012 and December 2014. A 36-month loan issued by the end of 2014 had
reached its final outcome by the time the data was closed in 2018, so the
target is reliable. I then drew a random sample of 18,000 loans (seed 42) so
the file is small enough to commit and retrain from scratch (2.9 MB). The
result has 18,000 rows and 26 columns, with a default rate of 13.4%.

I picked it because it has a natural business decision (approve or reject),
a target I have to define myself, a clear time dimension, and a mix of
numeric and categorical variables where feature engineering matters.

## Business / real-life framing

**Scenario.** A consumer lender must decide, for each new 36-month loan
application, whether to approve or reject it. The model estimates the
probability that the loan will default, and the loan is approved if that
probability is below a threshold.

**Target definition.** `default = 1` if the loan ended as "Charged Off" and
`0` if "Fully Paid". Loans in other statuses (e.g. Current, Late) are dropped
because their outcome is not final.

**Only application-time information.** The model must only use what the lender
knows at the moment of the decision. Post-origination columns (payments
received, recoveries, last payment date) are excluded because they leak the
outcome. I also exclude `int_rate`, `grade` and `sub_grade`: these are
produced by Lending Club's own scoring model, so using them would mean
inheriting its answer instead of building an independent assessment of the
applicant.

**Asymmetric costs, which set the threshold.**
- Approving a loan that defaults (false negative) costs roughly
  LGD × loan amount, where LGD is the share of the principal not recovered.
- Rejecting a loan that would have been repaid (false positive) costs the net
  interest margin I would have earned.

Assumptions (not taken from the data, to be varied in the dashboard):
LGD = 65% and net margin = 10% of the loan amount. A loan is worth
approving when (1 − p) × margin > p × LGD, i.e. when p < margin /
(margin + LGD) ≈ 13%. So the threshold is chosen by **minimizing expected
cost on the validation set**, not fixed at 0.5. Because the default rate is
13.4%, accuracy is misleading (a model that approves everyone scores 86.6%),
so I also report ROC-AUC and PR-AUC, which do not depend on the threshold.

**Time-based split.** A random split would train on loans issued after some of
the loans I evaluate on, leaking the future. The split is by issue date:
train 2012–2013, validation January–June 2014 (to choose regularization,
learning rate and threshold), test July–December 2014 (used once at the end).

**Baselines.** (1) Approve every loan. (2) A single-variable rule on FICO
score.

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
 