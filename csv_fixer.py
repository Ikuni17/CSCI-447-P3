import pandas

ff_path = "C:\\Users\\white\\OneDrive\\School\\CSCI-447 Machine Learning\\Project3\\CSCI447_P3\\datasets\\forestfires.csv"
machine_path = 'C:\\Users\\white\\OneDrive\\School\\CSCI-447 Machine Learning\\Project3\\CSCI447_P3\\datasets\\machine.csv'
ff_out = 'C:\\Users\\white\\OneDrive\\School\\CSCI-447 Machine Learning\\Project3\\CSCI447_P3\\datasets\\converted\\forestfires-fixed.csv'
machine_out = 'C:\\Users\\white\\OneDrive\\School\\CSCI-447 Machine Learning\\Project3\\CSCI447_P3\\datasets\\converted\\machine-fixed.csv'

df = pandas.read_csv(ff_path, header=None)
df = df.drop([2], axis=1)
df = df.drop([3], axis=1)
df.to_csv(ff_out, index=False, header=False)

df = pandas.read_csv(machine_path, header=None)
df = df.drop([0], axis=1)
df = df.drop([1], axis=1)
df = df.drop([9], axis=1)
df.to_csv(machine_out, index=False, header=False)
