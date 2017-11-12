import pandas

ff_path = "datasets\\forestfires-old.csv"
machine_path = 'datasets\\machine-old.csv'
ff_out = 'datasets\\converted\\forestfires.csv'
machine_out = 'datasets\\converted\\machine-fixed.csv'

df = pandas.read_csv(ff_path, header=None)
df = df.drop([2], axis=1)
df = df.drop([3], axis=1)
df.to_csv(ff_out, index=False, header=False)

df = pandas.read_csv(machine_path, header=None)
df = df.drop([0], axis=1)
df = df.drop([1], axis=1)
df = df.drop([9], axis=1)
df.to_csv(machine_out, index=False, header=False)