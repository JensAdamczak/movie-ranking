# Movie ranking

Multiple ranking methods are currently supported.

## Baseline - Average win ratio
*rank_baseline*

As a baseline, movies are ranked according to their win ratio, i.e. the number of wins divided by the number of competitions.

## Direct method - Perron-Frobenius scheme
*rank_directly*

The direct methods calculates a ranking based on the [Perron-Frobenius](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem) theorem.

The ranking is based on a preference matrix A that needs to be irreducible. For paired comparisons (0 for loss, 1 for win), the matrix is irreducible if there is no partition into two sets *S* and *T* such that no team in *S* plays any team in *T* or every game between one team from *S* and one team from *T* resulted in a victory for the team in *S*. In particular, for this preference matrix to be irreducible, there can be now winless teams.

We have winless teams, or better movies. Therefore the power method might not converge. We can pick n=2. This gives the interpretation of calculating the ranking of team *i* according to the average winning percentage of teams that theam *i* defeated.

## Bradley-Terry model
*rank_bradley_terry*

The [Bradley-Terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) model assumes that a ranking vector $r$ can be determined with the following properties:

$$p_{ij} = \frac{r_i}{r_i + r_j}$$

The model can be expressed in the $logit$ linear form

$$\log \frac{p_{ij}}{1 - p_{ji}} = \lambda_i - \lambda_j$$
with $$\lambda_i = \log r_i$$ for all $$i$$.

## Regularization check
*reg_check*

The Bradley-Terry model as used here has a few parameter - the regularization of the model. To chose the parameter we will evaluate how accurate the model can predict the number of times a movie will win a competition in a hold out set. For that purpose we will train different models with varying regularization on a training set of a subset of competitions.

## LinearSVC
*rank_svc*

The LinearSVC ranking methos is based on ([Herbrich 1999](https://www.mendeley.com/catalogue/a2709c3c-0705-3058-89db-28aeab2161f2/). If we consider linear ranking functions, the ranking problem can be transformed into a two-class classification problem. For this, we form the difference of all comparable elements such that our data is transformed into 
$$(x′_k, y′_k)=(x_i−x_j, \text{sign}(y_i−y_j))$$ 
for all comparable pairs.

We need to transform our feature (movie) values to account for the difference in hotels. In light of better knowledge, let's take 1 and -1.

A SVM machine is a maximum margin classifier. In the case of completely separable classes, if minimizes the L2-norm. For cases, in which classes cannot be sepearate completely, it additionally adds a hinge-loss to penalize misclassification. 

The difference between a (regularized) logistic regression model and a linearSVC model is therefore logloss vs. hinge-loss.
