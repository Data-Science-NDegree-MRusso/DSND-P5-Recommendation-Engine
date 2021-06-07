# Recommendation Engine Project
The goal of this project is to analyze the interactions that users have with articles on the IBM Watson Studio platform, and then explore several recommendation strategies to make suggestion to users about new articles they might like.  

The projecte is completed as a Jupyter notebook available [here](./Recommendations_with_IBM.ipynb): in it we can distinguish V different sections, that will be described in more detail here in the following.  
The notebook was originally provided by Udacity, and each section included a list of questions and tasks that have been now completed. The original template also 
included some test functions to be called while progressing, to check on the results.

## Sect. I - Exploratory Data Analysis
The data to work with consist of two different dataframes:

* One called simply `df`, including information about the usage of each article available in the platform, with their title and an id of a user that has interacted with it. Users were initially identified through their email, but this gets later changed in a more practical and simpler `user_id`.
* Another one called `df_content`, that lists all the articles, with more detailed information like a description and the body content, but without data on their actual usage.

In this section a few questions are asked in terms of data analysis: number of unique articles, unique users etc. As a quick summary:

 _Questions_    | _Answers_     |
| ------------- |:-------------:| 
| # of unique articles that have an interaction with a user | 714 | 
| # of unique articles in the dataset | 1051 |
| # of unique users in the dataset | 5148 |
| Total # of user-article interaction | 45993 |
| `id` of the most viewed article | `1429.0` |
| # of views for the most viewed article | 937 |

To get some more insight on the distribution of user-article interactions we can take a look at the histogram:

<p align="center">
  <img src="./pictures/user-interaction-hist.png">
</p>

More quantitatively, the analysis showed that:

* 75% of the users interacted with less than 10 articles;
* Isolated peaks are possible, with a max of 364 interactions counted from a single user.


## Sect. II - Rank-Based Recommendations
The first kind of recommendations that we try to provide are based on _rank_: given any user, we can provide them a list of the highest ranked items in our dataset.  
In this case, rank is measured through user-article interactions: the higher the number of interactions is, the higher the rank of the article will be considered to be.  

As an example of the outcome of this section we can see the **top 7** ranked articles:

<p align="center">
  <img width="850" height="160" src="./pictures/ranked-recs-example.png">
</p>

## Sect. III - User-User Based Collaborative Filtering
The second family of recommendations is based on a _collaborative_ approach, in which we investigate the dataset to identify _similarities_ between users.

* First of all, starting from the `df` dataframe we obtain a user-item matrix in which, for each unique user, we identify the article that they interacted with:

<p align="center">
  <img width="850" height="225" src="./pictures/user-item-matrix.png">
</p>

* Then we implement a similarity metric based on the number of seen articles that two users have in common:  the highr that number, the more simialr the users are considered to be.   
In the notebook there are actually two differnt proposals to measure this kind of similarity: one makes use of the dot product and another of the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

* Based on the information above it is possible to implement a function that, given an input a `user_id` and a integer `m` is capable of returning the first `m` article read by the users that are the most similar to `user_id`, and that `user_id` has not yet interacted to.
* This function is than further refined to take into account also the absolute number of interactions that characterizes an article.

As an example of the outcome of this section we can see the **top 10** recommendations for **user 20**:

<p align="center">
  <img width="850" height="255" src="./pictures/collaborative-recs-example.png">
</p>
 

## Sect. IV - Content-Based Recommendations (OPTIONAL)


## Sect. V - Matrix Factorization


