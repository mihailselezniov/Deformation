import json

lenLine = 2*10
stepLine = 2

lines = []
addKey = lambda key, obj: {key: obj}

for step in range(-1*lenLine/2, lenLine/2+1):
    line1, line2 = [], []
    for i in range(-1*lenLine, lenLine+1, stepLine):
        line1.append({"x": float(i), "y": float(abs((i/stepLine+step%2)%2*0.5)), "z": float(step*stepLine)})
        line2.append({"x": float(step*stepLine), "y": float(abs((i/stepLine+(step+1)%2)%2*0.5)), "z": float(i)})
    lines.append(addKey("points", line1))
    lines.append(addKey("points", line1))
data = addKey("lines", lines)

with open('data.json', 'w') as outfile:
    json.dump(data, outfile)
