import pandas as pd

abnormal = pd.read_csv("./openstack/openstack_abnormal.csv", index_col=None) # OpenStack dataset
normal1 = pd.read_csv("./openstack/openstack_normal1.csv", index_col=None) # OpenStack dataset
normal2 = pd.read_csv("./openstack/openstack_normal2.csv", index_col=None) # OpenStack dataset
pdlist = [abnormal, normal1, normal2]
# combine dataset
data = pd.concat(pdlist, axis=0)
#print(data)

data.dropna(inplace=True)
print(data)

data.to_csv("./openstack/openstack_full.csv", index=False)