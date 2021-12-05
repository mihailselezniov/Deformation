import numpy as np

N1 = 5#25
N2 = 5#201
strike_energy = np.linspace(0, 50, N2, endpoint=True)
strike_energy = list(map(lambda x: round(x, 5), strike_energy))

protection_thickness = np.linspace(2.0, 8.0, N1, endpoint=True)
protection_thickness = list(map(lambda x: round(x, 5), protection_thickness))

base_s = '0000111111111111111111111'
base_st_pr = {}
st_pr = []
i = 0
for s in strike_energy:
  for p in protection_thickness:
    st_pr.append([s, p])
    base_st_pr[(s, p)] = int(base_s[i])
    i += 1

print(st_pr[0], len(st_pr), st_pr[-1])
print(protection_thickness)
print(strike_energy)
print(base_st_pr)


N1 = 25
N2 = 201
strike_energy = np.linspace(0, 50, N2, endpoint=True)
strike_energy = list(map(lambda x: round(x, 5), strike_energy))

protection_thickness = np.linspace(2.0, 8.0, N1, endpoint=True)
protection_thickness = list(map(lambda x: round(x, 5), protection_thickness))

st_pr = []
base_ids = {}
i = 0
for s in strike_energy:
  for p in protection_thickness:
    st_pr.append([s, p])
    if (s, p) in base_st_pr:
      #print(i)
      base_ids[i] = base_st_pr[(s, p)]
    i += 1

print(st_pr[0], len(st_pr), st_pr[-1])
print(base_ids)
#print(protection_thickness)
#print(strike_energy)

"""
[0.0, 2.0] 25 [50.0, 8.0]
[2.0, 3.5, 5.0, 6.5, 8.0]
[0.0, 12.5, 25.0, 37.5, 50.0]
{(0.0, 2.0): 0, (0.0, 3.5): 0, (0.0, 5.0): 0, (0.0, 6.5): 0, (0.0, 8.0): 1, (12.5, 2.0): 1, (12.5, 3.5): 1, (12.5, 5.0): 1, (12.5, 6.5): 1, (12.5, 8.0): 1, (25.0, 2.0): 1, (25.0, 3.5): 1, (25.0, 5.0): 1, (25.0, 6.5): 1, (25.0, 8.0): 1, (37.5, 2.0): 1, (37.5, 3.5): 1, (37.5, 5.0): 1, (37.5, 6.5): 1, (37.5, 8.0): 1, (50.0, 2.0): 1, (50.0, 3.5): 1, (50.0, 5.0): 1, (50.0, 6.5): 1, (50.0, 8.0): 1}
[0.0, 2.0] 5025 [50.0, 8.0]
{0: 0, 6: 0, 12: 0, 18: 0, 24: 1, 1250: 1, 1256: 1, 1262: 1, 1268: 1, 1274: 1, 2500: 1, 2506: 1, 2512: 1, 2518: 1, 2524: 1, 3750: 1, 3756: 1, 3762: 1, 3768: 1, 3774: 1, 5000: 1, 5006: 1, 5012: 1, 5018: 1, 5024: 1}

"""