# Part 1 - Hello Machine Learning

Original Youtube Video: https://youtu.be/cKxRvEZd3Mw

This part is an introduction to machine learning. In this first part we will build a really simple classified learning.
We will create a classifier to determine whether a fruit is an orange or and apple based on two features: weight, and textures.

## Data Sample

Here's the data sample we'll use as traning data.

| Weight | Texture | Label |
| ------ | ------- | ----- |
| 150g   | Bumpy   | Orange|
| 170g   | Bumpy   | Orange|
| 140g   | Smooth  | Apple |
| 130g   | Smooth  | Apple |

Weight and texture will be considered as features, while the label is the name of fruits (Apple or Orange).
And based on the data above, we can learn that orange tend to be heavier with bumpy texture, while apples are lighter and smooth.

Based on the data sample, we can define `features` and `labels` variables.

```python
features = [[150, 'bumpy'], [170, 'bumpy'], [140, 'smooth'], [130, 'smooth']]
labels = [['orange'], ['orange'], ['apple'], ['apple']]

```

The code above can be simplified, by changing the texture values to `0` for `'smooth'` and `1` for `'bumpy'`.
We can do the same for the label, so we will change `'apple'` to `'0'`, and `'orange'` to `'1'`.
So, the code will look like this:

```python
# 0 - smooth, 1 - bumpy
features = [[150, 1], [170, 1], [140, 0], [130, 0]]
# 0 - apple, 1 - orange
labels = [[1], [1], [1], [1]]
```

## Using classifier for the first time

After getting the data sample for the training ready, we start preparing the classifier. In this first part we will use `DecisionTreeClassifier` from `sklearn`.

So, the first thing we need to do is calling the `tree` from `sklearn` package. Here's the code:
```python
from sklearn import tree
```
More detailed explanation about tree module from sklearn documentation can be found [here](https://scikit-learn.org/stable/modules/tree.html).

After calling the tree module, we can access `DecisionTreeClassifier` from it, and assign it to variable. Then, we can add the data sample to `fit()` method for the training.
`fit()` method will build a decision tree classifier from the data training set (X = features, y = labels).

More detailed explanation about how `DecisionTreeClasssifier` works under the hood, will be talked about in another part.

[More about fit method](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.fit
)
```python
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
```
The last thing we can do is trying to use the classifier to do some predictions.
Here's the example:

```python
# weight 165g, texture bumpy, expected output: 1 - orange
print(clf.predict([165, 1]))
# weight 150g, texture smooth, expected output: 0 - apple
print(clf.predict([150, 0]))
```

The final full code will look like this:

```python
from sklearn import tree

# 0 - smooth, 1 - bumpy
features = [[150, 1], [170, 1], [140, 0], [130, 0]]
# 0 - apple, 1 - orange
labels = [[1], [1], [1], [1]]

# calling the DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()
# clf.fit build the classifier based on traning dataset
clf = clf.fit(features, labels)

# weight 165g, texture bumpy, expected output: 1 - orange
print(clf.predict([165, 1]))
# weight 150g, texture smooth, expected output: 0 - apple
print(clf.predict([150, 0]))

```

That's it, our first simple classifier with scikit-learn. The next part will talk about how we can visualize the DecisionTreeClassifier tree.