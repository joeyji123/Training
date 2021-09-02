from ep1OOP import ep1OOP

test = ep1OOP("branin_data.csv", ["x1", "x1"], [-5, 10, 0, 15], 2)
#test.show()
clf = test.curveFit(2)
print(clf.predict(test.polyTransform(2, [[5, 10]])))
print(test.R2(2))