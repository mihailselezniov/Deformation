import numpy as np

N = 11#11 5
N_strike_energy = 5#201
N_protection_thickness = 5#25
N_strike_duration = 5

rangee = lambda start, stop, N: tuple(map(lambda x: round(x, 5), np.linspace(start, stop, N, endpoint=True)))

strike_energy = rangee(0, 50, N)
protection_thickness = rangee(1.0, 8.0, N)
strike_duration = rangee(0.5e-4, 2.5e-4, N)
material_strength_limit = rangee(1e7, 7e7, N)
contact_strength_limit = rangee(1e7, 7e7, N)
target_thickness = rangee(8.0, 16.0, N)
width = rangee(8.0, 16.0, N)


st_pr = []
i = 0
for se in strike_energy:
  for p in protection_thickness:
    for sd in strike_duration:
      for m in material_strength_limit:
        for c in contact_strength_limit:
          for t in target_thickness:
            for w in width:
              st_pr.append([se, p, sd, m, c, t, w])
              i += 1
  print(se)

print()
print(len(st_pr))
print(st_pr[0])
print(st_pr[-1])
print()
print(strike_energy)
print(protection_thickness)
print(strike_duration)
print(material_strength_limit)
print(contact_strength_limit)
print(target_thickness)
print(width)