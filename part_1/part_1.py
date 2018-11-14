from sklearn import tree

# weight and textures
# 0 smooth - 1 bumpy
features = [[130, 0], [140, 0], [150, 1], [155, 1], [160, 0], [170, 1]]
# expected output 0 apple - 1 orange
labels = [0, 0, 1, 1, 0, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[170, 0]]))
print(clf.predict([[160, 1]]))
