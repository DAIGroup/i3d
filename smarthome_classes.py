labels = ["Cook", "Cook.Cleandishes", "Cook.Cleanup", "Cook.Cut", "Cook.Stir", "Cook.Usestove", "Cutbread", "Drink",
         "Drink.Frombottle", "Drink.Fromcan", "Drink.Fromcup", "Drink.Fromglass", "Eat.Attable", "Eat.Snack", "Enter",
         "Getup", "Laydown", "Leave", "Makecoffee", "Makecoffee.Pourgrains", "Makecoffee.Pourwater",
         "Maketea.Boilwater", "Maketea.Insertteabag", "Pour.Frombottle", "Pour.Fromcan", "Pour.Fromcup",
         "Pour.Fromkettle", "Readbook", "Sitdown", "Takepills", "Uselaptop", "Usetablet", "Usetelephone", "Walk",
         "WatchTV"]

labels_cv = ["Cutbread", "Drink.Frombottle", "Drink.Fromcan", "Drink.Fromcup", "Drink.Fromglass", "Eat.Attable",
             "Eat.Snack", "Enter", "Getup", "Leave", "Pour.Frombottle", "Pour.Fromcan", "Readbook", "Sitdown",
             "Takepills", "Uselaptop", "Usetablet", "Usetelephone", "Walk"]

def name_to_int(name):
    integer = 0
    if name == "Cook":
        integer = 1
    elif name == "Cook.Cleandishes":
        integer = 2
    elif name == "Cook.Cleanup":
        integer = 3
    elif name == "Cook.Cut":
        integer = 4
    elif name == "Cook.Stir":
        integer = 5
    elif name == "Cook.Usestove":
        integer = 6
    elif name == "Cutbread":
        integer = 7
    elif name == "Drink":
        integer = 8
    elif name == "Drink.Frombottle":
        integer = 9
    elif name == "Drink.Fromcan":
        integer = 10
    elif name == "Drink.Fromcup":
        integer = 11
    elif name == "Drink.Fromglass":
        integer = 12
    elif name == "Eat.Attable":
        integer = 13
    elif name == "Eat.Snack":
        integer = 14
    elif name == "Enter":
        integer = 15
    elif name == "Getup":
        integer = 16
    elif name == "Laydown":
        integer = 17
    elif name == "Leave":
        integer = 18
    elif name == "Makecoffee":
        integer = 19
    elif name == "Makecoffee.Pourgrains":
        integer = 20
    elif name == "Makecoffee.Pourwater":
        integer = 21
    elif name == "Maketea.Boilwater":
        integer = 22
    elif name == "Maketea.Insertteabag":
        integer = 23
    elif name == "Pour.Frombottle":
        integer = 24
    elif name == "Pour.Fromcan":
        integer = 25
    elif name == "Pour.Fromcup":
        integer = 26
    elif name == "Pour.Fromkettle":
        integer = 27
    elif name == "Readbook":
        integer = 28
    elif name == "Sitdown":
        integer = 29
    elif name == "Takepills":
        integer = 30
    elif name == "Uselaptop":
        integer = 31
    elif name == "Usetablet":
        integer = 32
    elif name == "Usetelephone":
        integer = 33
    elif name == "Walk":
        integer = 34
    elif name == "WatchTV":
        integer = 35
    return integer