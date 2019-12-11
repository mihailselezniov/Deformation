import gzip

for f_name in ['LogReg.j', 'KNN7.j']:
    f_in = open(f_name)
    f_out = gzip.open(f_name + '.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()