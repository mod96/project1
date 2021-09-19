

def dec(n_feature):
    def function(hp):
        print(n_feature)
        print(hp)
        return
    return function


a = dec(2)
print(a)
print(a(4))